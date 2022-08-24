"""
DataJoint tables for cell typing.
"""

import json
import traceback
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
import microns_dashboard_api as mdb_api

from ..config import cell_typer_config as config

config.register_externals()
config.register_adapters(context=locals())

schema = djp.schema(config.schema_name, create_schema=True)

version_attr = """-> Version \n"""
user_attr = """user : varchar(128) # dashboard username \n"""


@schema
class Version(mdb_api.VersionLookup):
    package = 'microns-coregistration-api'


@schema
class Protocol(djp.Lookup):
    hash_name = 'protocol_id'
    definition = """
    protocol_id : varchar(8)
    """

    @classmethod
    def get(cls, key, n=None, subtract=None):
        return cls.r1p(key, include_parts=cls.Info).get(key, n=n, subtract=subtract)

    class Info(djp.Part, dj.Lookup):
        enable_hashing = True
        hash_name = 'protocol_id'
        hashed_attrs = 'protocol_name', 'version_id'
        definition = f"""
        -> master
        protocol_name : varchar(128)
        ---
        -> Version
        source_type=NULL : varchar(128)
        description=NULL : longblob
        ts_inserted=CURRENT_TIMESTAMP : timestamp
        """

        @property
        def contents(cls):
            contents = [
                {
                    'protocol_name': 'training', 
                    'version_id': Version.current_version_id, 
                    'source_type': 'dynamic',
                    'description': 'minnie65_materialization.Segment.Nucleus \
                        * minnie65_auto_proofreading.AutoProofreadNeuron & multiplicity=1 \
                        & baylor_cell_type_exc_probability_after_proof > 0.95 \
                        & baylor_cell_type_after_proof=external_cell_type',
                }
            ] + [
                { 
                    'protocol_name': f'validation_{n}', 
                    'version_id': Version.current_version_id, 
                    'source_type': 'static',
                    'description': f'{n} random segments from training protocol with materialization version 343'} for n in [50, 20, 10]
            ] + [
                { 
                    'protocol_name': f'evaluation_{n}', 
                    'version_id': Version.current_version_id, 
                    'source_type': 'static',
                    'description': f'{n} random segments from training protocol with materialization version 343'} for n in [50, 20, 10]
            ]
            cls.insert(contents, insert_to_master=True, skip_duplicates=True, ignore_extra_fields=True)
            return {}

        
        @classmethod
        def fill_static_protocols(cls):
            for key in cls:
                protocol_id = key.get('protocol_id')
                protocol_name = key.get('protocol_name')
                source_type = key.get('source_type')
                if source_type == 'static':
                    if 'validation' in protocol_name or 'evaluation' in protocol_name:
                        n = int(protocol_name.split('_')[1])
                        df = cls.master.get({'protocol_name': 'training', 'version_id': Version.current_version_id, 'ver': 343}, n=n).reset_index().rename(columns={'index': 'idx'})
                        df['protocol_id'] = protocol_id
                        cls.master.StaticSource.insert(df, ignore_extra_fields=True, skip_duplicates=True)

        def to_df(self, rel, n=None):
            if n is None:
                return pd.DataFrame(rel.fetch())
            else:
                return pd.DataFrame(rel.fetch(order_by='RAND()', limit=n, as_dict=True))

        def get(self, key, n=None, subtract=None):
            params = self.fetch1()
            assert params.get('version_id') == Version.current_version_id, 'Version mismatch'
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
                    evaluation_segs = self.master.StaticSource & (self.__class__ & 'protocol_name LIKE "evaluation%"')
                    final = rel & unique_segs.proj() - evaluation_segs.proj()
                    return self.to_df((final - subtract) & key, n=n)
                else:
                    raise AttributeError(f'protocol_name {protocol_name} not found.')
            else:
                raise AttributeError(f'source_type {source_type} not found.')

    class StaticSource(djp.Part):
        definition = """
        -> master
        idx : int unsigned
        -> m65mat.Segment
        ---
        cell_type=NULL : varchar(256)
        other_info=NULL : longblob
        """
    
    class Manager(djp.Part, dj.Lookup):  
        hash_name = 'protocol_id'
        definition = """
        -> master.Info
        ---
        active=0 : tinyint # 1 if active, 0 if inactive
        ordering=NULL : tinyint # order of protocol from right to left, default NULL is placed at the end
        """
        @classproperty
        def contents(cls):
            cls.insert(cls.master.Info, ignore_extra_fields=True, skip_duplicates=True)
            return {}
@schema
class Event(mdb_api.EventLookup):
    basedir = Path(config.externals.get('cell_typer_files').get('location'))
    additional_secondary_attrs = version_attr

    class Submission(mdb_api.EventType):
        event_types = 'cell_type_submission', 'flagged_submission'
        constant_attrs = {'version_id': Version.current_version_id}
        additional_primary_attrs = version_attr
        additional_secondary_attrs = user_attr

        def on_event(self, event):
            Submission.Maker.populate({'event_id': event.id})

@schema
class EventHandler(mdb_api.EventHandlerLookup):
    current_version_id = Version.current_version_id

    class Submission(djp.Part, dj.Lookup):
        enable_hashing = True
        hash_name = 'event_handler_id'
        hashed_attrs = 'version_id', 'event_type'
        definition = """
        -> master
        event_type : varchar(450) # type of event that method handles
        -> Version
        """

        @classproperty
        def contents(cls):
            for event_type in Event.Submission.event_types:
                cls.insert({'event_type': event_type, 'version_id': Version.current_version_id}, ignore_extra_fields=True, skip_duplicates=True, insert_to_master=True)
            return {}

        def run(self, key):
            event_type = self.fetch1('event_type')
            data = key.get('data')
            if event_type in ['cell_type_submission']:
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
            
            if event_type in ['flagged_submission']:
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
    """

    class Maker(djp.Part, dj.Computed):
        enable_hashing=True
        hash_name = 'submission_id'
        hashed_attrs = Event.primary_key + EventHandler.primary_key
        definition = """
        -> master
        -> Event
        -> EventHandler
        ---
        ts_inserted_maker=CURRENT_TIMESTAMP : timestamp # timestamp inserted to Maker
        """

        @classproperty
        def key_source(cls):
            return (djp.U('event_id', 'event_handler_id') & (Event.Submission * EventHandler.Submission)) - cls.master.Attempted

        def make(self, key):
            key[self.hash_name] = self.hash1(key)
            try:
                key.update(Event.get1(key))
                key.update(EventHandler.run(key))
                self.insert1(key, ignore_extra_fields=True, insert_to_master=True, insert_to_master_kws={'ignore_extra_fields': True, 'skip_duplicates': True}, skip_hashing=True)
                
                if key.get('event_type') in ['cell_type_submission']:
                    self.master.CellType.insert1(key, ignore_extra_fields=True)
                elif key.get('event_type') in ['flagged_submission']:
                    self.master.Flagged.insert1(key, ignore_extra_fields=True)
            except:
                key['traceback'] = traceback.format_exc()
                self.master.Attempted.insert1(key, insert_to_master=True, ignore_extra_fields=True)
                self.master.Log('exception', 'Error inserting Submission')
    
    class Attempted(djp.Part):
        definition = """
        -> master
        -> Event
        -> EventHandler
        ---
        traceback=NULL : longblob
        ts_inserted_attempted=CURRENT_TIMESTAMP : timestamp # timestamp inserted into Attempted
        """

    class CellType(djp.Part):
        definition = """
        -> db.UserInfo
        -> Protocol
        -> m65mat.Segment
        -> master
        ---
        user_choice : varchar(128) # user selected choice
        cell_type_label=NULL : varchar(128) # cell type label (optional)
        note=NULL : longblob # user note (optional)
        other_data=NULL : longblob # other data (optional)
        ts_inserted_store=CURRENT_TIMESTAMP : timestamp # timestamp inserted into store
        """
    
    class Flagged(djp.Part):
        definition = """
        -> db.UserInfo
        -> Protocol
        -> m65mat.Segment
        -> master
        ---
        note=NULL : longblob # user note (optional)
        other_data=NULL : longblob # other data (optional)
        ts_inserted_store=CURRENT_TIMESTAMP : timestamp # timestamp inserted into store
        """

schema.spawn_missing_classes()
schema.connection.dependencies.load()