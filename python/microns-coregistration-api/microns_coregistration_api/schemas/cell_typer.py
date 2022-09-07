"""
DataJoint tables for cell typing.
"""

from imp import reload
import pandas as pd
from pathlib import Path
import datajoint_plus as djp
import datajoint as dj
from microns_utils.misc_utils import classproperty
from microns_utils.datetime_utils import current_timestamp
from microns_utils.version_utils import check_package_version

from microns_materialization_api.schemas import \
    minnie65_materialization as m65mat
from microns_morphology_api.schemas import \
    minnie65_auto_proofreading as m65auto

from microns_dashboard_api.schemas import \
    dashboard as db
import microns_dashboard_api as mdb

from ..config import cell_typer_config as config

config.register_externals()
config.register_adapters(context=locals())

schema = djp.schema(config.schema_name, create_schema=True)

@schema
class Tag(mdb.VersionLookup):
    package = 'microns-coregistration-api'
    attr_name = 'tag'

@schema
class User(djp.Lookup):
    definition = f"""
    user : varchar(128) # dashboard username
    ---
    timestamp=CURRENT_TIMESTAMP : timestamp
    """

    class MICrONSDashboard(djp.Part, dj.Lookup):
        definition = f"""
        -> master
        -> db.User
        ---
        {db.User.hash_name} : varchar({db.User.hash_len}) # {db.User.comment}
        """
        
        @classproperty
        def contents(cls):
            cls.insert(db.User, ignore_extra_fields=True, skip_duplicates=True, insert_to_master=True)
            return {}

@schema
class Protocol(djp.Lookup):
    hash_name = 'protocol_id'
    definition = """
    protocol_id : varchar(8)
    ---
    timestamp=CURRENT_TIMESTAMP : timestamp # 
    """

    @classmethod
    def get(cls, key, n=None, subtract=None):
        return cls.r1p(key, include_parts=cls.Info).get(key, n=n, subtract=subtract)

    class StaticSource(djp.Part):
        definition = """
        -> master
        idx : int unsigned
        -> m65mat.Segment
        ---
        cell_type=NULL : varchar(256)
        other_info=NULL : longblob
        """
    
    class Info(djp.Part, dj.Lookup):
        enable_hashing = True
        hash_name = 'protocol_id'
        hashed_attrs = 'protocol_name', Tag.attr_name
        definition = f"""
        -> master
        protocol_name : varchar(128)
        -> {Tag.class_name}
        ---
        source_type=NULL : varchar(128)
        description=NULL : longblob
        """

        @property
        def contents(cls):
            contents = [
                {
                    'protocol_name': 'training', 
                    Tag.attr_name: Tag.version, 
                    'source_type': 'dynamic',
                    'description': 'minnie65_materialization.Segment.Nucleus \
                        * minnie65_auto_proofreading.AutoProofreadNeuron & multiplicity=1 \
                        & baylor_cell_type_exc_probability_after_proof > 0.95 \
                        & baylor_cell_type_after_proof=external_cell_type',
                }
            ] + [
                { 
                    'protocol_name': f'validation_{n}', 
                    Tag.attr_name: Tag.version, 
                    'source_type': 'static',
                    'description': f'{n} random segments from training protocol with materialization version 343'} for n in [10, 20, 50]
            ] + [
                { 
                    'protocol_name': f'evaluation_{n}', 
                    Tag.attr_name: Tag.version, 
                    'source_type': 'static',
                    'description': f'{n} random segments from training protocol with materialization version 343'} for n in [10, 20, 50]
            ]
            cls.insert(contents, insert_to_master=True, skip_duplicates=True, ignore_extra_fields=True)
            protocol_id_source = djp.U('protocol_id') & (cls & {'source_type="static"'})
            static_source = djp.U('protocol_id') & cls.master.StaticSource()
            new_protocols = protocol_id_source - static_source
            if len(new_protocols) > 0:
                cls.master.Log('info', 'Found new static protocols. Adding...')
                cls.load_dependencies()
                cls.fill_static_protocols(new_protocols)
            return {}

        
        @classmethod
        def fill_static_protocols(cls, protocols):
            for key in protocols:
                protocol_id = key.get('protocol_id')
                protocol_name = (cls & key).fetch1('protocol_name')
                if 'validation' in protocol_name or 'evaluation' in protocol_name:
                    n = int(protocol_name.split('_')[1])
                    df = cls.master.get({'protocol_name': 'training', Tag.attr_name: Tag.version, 'ver': 343}, n=n).reset_index().rename(columns={'index': 'idx'})
                    df['protocol_id'] = protocol_id
                    cls.master.StaticSource.insert(df, ignore_extra_fields=True, skip_duplicates=True)

        def to_df(self, rel, n=None):
            if n is None:
                return pd.DataFrame(rel.fetch())
            else:
                return pd.DataFrame(rel.fetch(order_by='RAND()', limit=n, as_dict=True))

        def get(self, key, n=None, subtract=None):
            params = self.fetch1()
            assert params.get(Tag.attr_name) == Tag.version, 'Version mismatch'
            protocol_id = params.get('protocol_id')
            protocol_name = params.get('protocol_name')
            source_type = params.get('source_type')
            subtract = [] if subtract is None else subtract
            if source_type == 'static':
                return self.to_df(((self.master.StaticSource & {'protocol_id': protocol_id}) - subtract) & key, n=n)
            elif source_type == 'dynamic':
                if protocol_name == 'training':
                    ver = m65mat.Materialization.latest if key.get('ver') is None else {'ver': key.get('ver')}
                    segment_rel = m65mat.Segment.Nucleus & ver
                    autoproof_rel = (m65auto.AutoProofreadNeuron & 'multiplicity=1' & 'baylor_cell_type_after_proof=external_cell_type' & 'baylor_cell_type_exc_probability_after_proof > 0.95').proj(cell_type="baylor_cell_type_after_proof")
                    rel = djp.U('segment_id', 'nucleus_id', 'cell_type') & (segment_rel * autoproof_rel)
                    unique_segs = djp.U('segment_id').aggr(rel, n_seg='count(segment_id)') & 'n_seg=1'
                    sequestered_segs = self.master.StaticSource & (self.__class__ & ['protocol_name LIKE "validation%"', 'protocol_name LIKE "evaluation%"'])
                    final = rel & unique_segs.proj() - sequestered_segs.proj()
                    return self.to_df((final - subtract) & key, n=n)
                else:
                    raise AttributeError(f'protocol_name {protocol_name} not found.')
            else:
                raise AttributeError(f'source_type {source_type} not found.')

    class Manager(djp.Part, dj.Lookup):  
        hash_name = 'protocol_id'
        definition = """
        -> master.Info
        ---
        active=0 : tinyint # 1 if active, 0 if inactive
        ordering=NULL : tinyint # order of protocol from right to left, default NULL is placed at the end
        last_updated=CURRENT_TIMESTAMP : timestamp
        """
        @classproperty
        def contents(cls):
            cls.insert(cls.master.Info, ignore_extra_fields=True, skip_duplicates=True)
            return {}
            
@schema
class Event(mdb.EventLookup):
    basedir = Path(config.externals.get('cell_typer_files').get('location'))
    extra_secondary_attrs = f"""
    -> {Tag.class_name}
    """

    class Submission(mdb.Event):
        events = 'cell_type_submission', 'flagged_submission'
        constant_attrs = {Tag.attr_name: Tag.version}
        extra_primary_attrs = f"""
        -> {Tag.class_name}
        """
        extra_secondary_attrs = f"""
        -> {User.class_name}
        """

        def on_event(self, event):
            Submission.Add.populate({'event_id': event.id})

@schema
class EventHandler(mdb.EventHandlerLookup):
    @classmethod
    def run(cls, key):
        handler = cls.r1p(key)
        handler_event = handler.fetch1('event')
        handler_version = handler.fetch1(Tag.attr_name)
        cls.Log('info',  'Running %s', handler.class_name)
        cls.Log('debug', 'Running %s with key %s', handler.class_name, key)
        assert handler_event == key['event'], f'event in handler {handler_event} doesnt match event in key {key["event"]}'
        assert handler_version == Tag.version, f'version mismatch, event_handler version_id is {handler_version} but the current version_id is {Tag.version}'
        key = handler.run(key)
        cls.Log('info', '%s ran successfully.', handler.class_name)
        return key

    class Submission(mdb.EventHandler):
        hashed_attrs = 'event', Tag.attr_name
        constant_attrs = {Tag.attr_name: Tag.version}
        extra_primary_attrs = f"""
        -> {Tag.class_name}
        """

        @classproperty
        def contents(cls):
            for event in Event.Submission.events:
                key = {'event': event}
                key.update(cls.constant_attrs)
                cls.insert(key, ignore_extra_fields=True, skip_duplicates=True, insert_to_master=True)
            return {}

        def run(self, key):
            event = self.fetch1('event')
            data = key.get('data')
            if event in ['cell_type_submission']:
                required_attrs = ['protocol_id', 'segment_id', 'user_choice']
                assert data is not None, f'data must include: {required_attrs}'
                data.setdefault('cell_type_label', None)
                data.setdefault('note', None)
                key['protocol_id'] = data.pop('protocol_id')
                key['segment_id'] = data.pop('segment_id')
                key['user_choice'] = data.pop('user_choice')
                key['cell_type_label'] = data.pop('cell_type_label')
                key['note'] = data.pop('note')
                key['other_data'] = data if data else None
                return key
            
            if event in ['flagged_submission']:
                required_attrs = ['protocol_id', 'segment_id', 'note']
                assert data is not None, f'data must include: {required_attrs}'
                key['protocol_id'] = data.pop('protocol_id')
                key['segment_id'] = data.pop('segment_id')
                key['note'] = data.pop('note')
                key['other_data'] = data if data else None
                return key


@schema
class Submission(djp.Lookup):
    hash_name = 'submission_id'
    definition = """
    submission_id : varchar(10)
    ---
    timestamp=CURRENT_TIMESTAMP : timestamp # 
    """

    class Add(mdb.Maker):
        hash_name = 'submission_id'
        upstream = Event
        method = EventHandler
        @classproperty
        def key_source(cls):
            return (djp.U('event_id', 'event_handler_id') & (Event.Submission * EventHandler.Submission))

        def on_make(self, key):
            if key.get('event') in ['cell_type_submission']:
                self.master.CellType.insert1(key, ignore_extra_fields=True)

            elif key.get('event') in ['flagged_submission']:
                self.master.Flagged.insert1(key, ignore_extra_fields=True)

    class CellType(djp.Part):
        definition = f"""
        -> User
        -> Protocol
        -> m65mat.Segment
        -> master
        ---
        user_choice : varchar(128) # user selected choice
        cell_type_label=NULL : varchar(128) # cell type label (optional)
        note=NULL : longblob # user note (optional)
        other_data=NULL : longblob # other data (optional)
        """
    
    class Flagged(djp.Part):
        definition = f"""
        -> User
        -> Protocol
        -> m65mat.Segment
        -> master
        ---
        note=NULL : longblob # user note (optional)
        other_data=NULL : longblob # other data (optional)
        """

schema.spawn_missing_classes()
schema.connection.dependencies.load()