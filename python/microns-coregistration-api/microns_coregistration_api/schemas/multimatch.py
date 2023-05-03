"""
DataJoint tables for multimatch.
"""

import pandas as pd
import datajoint_plus as djp
import datajoint as dj

from ..config import multimatch_config as config

from microns_utils.misc_utils import classproperty, wrap
import microns_utils.datajoint_utils as dju
from microns_utils.widget_utils import SlackForWidget
from microns_nda_api.schemas import \
    minnie_nda as nda
from microns_materialization_api.schemas import \
    minnie65_materialization as m65mat
from microns_coregistration_api.schemas import \
    minnie65_coregistration as m65crg
m6502 = djp.create_djp_module(schema_name='microns_minnie65_02')

config.register_externals()
config.register_adapters(context=locals())

schema = djp.schema(config.schema_name, create_schema=True)

slack_client = SlackForWidget(default_channel='#microns-dashboard')
user_attr = """user : varchar(128) # dashboard username"""


@schema
class Tag(dju.VersionLookup):
    package = 'microns-coregistration-api'
    attr_name = 'tag'


@schema
class Protocol(djp.Lookup):
    hash_name = 'protocol_id'
    definition = """
    protocol_id : varchar(6) # id of protocol
    ---
    timestamp=CURRENT_TIMESTAMP : timestamp
    """

    class M6502Unit(djp.Part):
        enable_hashing = True
        hash_name = 'protocol_id'
        hashed_attrs = 'protocol_name'
        definition = f"""
        -> master
        -> m6502.UnitMatchProtocol
        """
        @classproperty
        def key_source(cls):
            return m6502.UnitMatchProtocol()
        
        @classmethod
        def fill(cls):
            cls.insert(cls.key_source, insert_to_master=True, ignore_extra_fields=True, skip_duplicates=True, insert_to_master_kws={'ignore_extra_fields': True, 'skip_duplicates': True})
        
        def get(self, key):
            return self.key_source.get(key)

    class M6502Nucleus(djp.Part):
        enable_hashing = True
        hash_name = 'protocol_id'
        hashed_attrs = 'protocol_name'
        definition = f"""
        -> master
        -> m6502.NucleusMatchProtocol
        """
        @classproperty
        def key_source(cls):
            return m6502.NucleusMatchProtocol()
        
        @classmethod
        def fill(cls):
            cls.insert(cls.key_source, insert_to_master=True, ignore_extra_fields=True, skip_duplicates=True, insert_to_master_kws={'ignore_extra_fields': True, 'skip_duplicates': True})
        
        def get(self, key):
            return self.key_source.get(key)


@schema
class MatchInterface(djp.Lookup):
    definition = """
    interface : varchar(128) # type of matching interface
    """
    contents = [['classic'], ['multimatch']]


@schema
class MatchPriority(djp.Lookup):
    definition = """
    priority : varchar(128) # priority of match
    """
    contents = [['primary'], ['secondary']]


@schema
class Event(dju.EventLookup):
    extra_secondary_attrs = f"""
    -> {Tag.class_name}
    """

    class UserAccess(dju.Event):
        events = ['user_access']
        constant_attrs = {Tag.attr_name: Tag.version}
        extra_primary_attrs = f"""
        -> {Tag.class_name}
        """
        extra_secondary_attrs = f"""
        {user_attr}
        """
        def on_event(self, event):
            user = (self & {'event_id': event.id}).fetch1('user')
            if len(User & {'user': user}) == 0:
                event = self.master.log_event('user_add', {'user': user})
            slack_client.post_to_slack(f"```{user} accessed the multimatch tool```")

    class UserAdd(dju.Event):
        events = ['user_add']
        constant_attrs = {Tag.attr_name: Tag.version}
        extra_primary_attrs = f"""
        -> {Tag.class_name}
        """
        extra_secondary_attrs = f"""
        {user_attr}
        """
        def on_event(self, event):
            User.Add.populate({'event_id': event.id})
    
    class SubmissionM6502(dju.Event):
        events = 'unit_match_attempt_m65_02', 'nucleus_match_attempt_m65_02'
        hashed_attrs = 'event', 'm6502_submission_id'
        constant_attrs = {Tag.attr_name: Tag.version}
        extra_primary_attrs = f"""
        -> {Tag.class_name}
        """
        extra_secondary_attrs = f"""
        m6502_submission_id : varchar(16) # unique identifier for submission
        """
        @classmethod
        def fill(cls):
            nuc_df = pd.DataFrame(m6502.NucleusManualMatchAttempt.proj('timestamp', m6502_submission_id='nucleus_submission_id', event="'nucleus_match_attempt_m65_02'").fetch())
            unit_df = pd.DataFrame(m6502.UnitManualMatchAttempt.proj('timestamp', m6502_submission_id='unit_submission_id', event="'unit_match_attempt_m65_02'").fetch())
            nuc_df['timestamp'] = nuc_df.timestamp.astype('str')
            unit_df['timestamp'] = unit_df.timestamp.astype('str')
            cls.insert(nuc_df, ignore_extra_fields=True, constant_attrs=cls.constant_attrs, insert_to_master=True)
            cls.insert(unit_df, ignore_extra_fields=True, constant_attrs=cls.constant_attrs, insert_to_master=True)

        def on_event(self, event):
            Submission.AddM6502.populate({'event_id': event.id})

    class Submission(dju.Event):
        events = 'unit_match_attempt', 'nucleus_match_attempt'
        constant_attrs = {Tag.attr_name: Tag.version}
        extra_primary_attrs = f"""
        -> {Tag.class_name}
        """
        extra_secondary_attrs = f"""
        {user_attr}
        """

        def on_event(self, event):
            Submission.Add.populate({'event_id': event.id})


@schema
class EventHandler(dju.EventHandlerLookup):
    @classmethod
    def run(cls, key, force=False):
        handler = cls.r1p(key)
        if not force:
            handler_event = handler.fetch1('event')
            handler_version = handler.fetch1(Tag.attr_name)
            cls.Log('info',  'Running %s', handler.class_name)
            cls.Log('debug', 'Running %s with key %s', handler.class_name, key)
            assert handler_event == key['event'], f'event in handler {handler_event} doesnt match event in key {key["event"]}'
            assert handler_version == Tag.version, f'version mismatch, event_handler version_id is {handler_version} but the current version_id is {Tag.version}'
        key = handler.run(key)
        cls.Log('info', '%s ran successfully.', handler.class_name)
        return key

    class UserAdd(dju.EventHandler):
        hashed_attrs = 'event', Tag.attr_name
        constant_attrs = {Tag.attr_name: Tag.version}
        extra_primary_attrs = f"""
        -> {Tag.class_name}
        """
        @classproperty
        def contents(cls):
            for event in Event.UserAdd.events:
                key = {'event': event}
                key.update(cls.constant_attrs)
                cls.insert(key, ignore_extra_fields=True, skip_duplicates=True, insert_to_master=True)
            return {}

        def run(self, key):
            return key
    
    class SubmissionM6502(dju.EventHandler):
        hashed_attrs = 'event', Tag.attr_name
        constant_attrs = {Tag.attr_name: Tag.version}
        extra_primary_attrs = f"""
        -> {Tag.class_name}
        """

        @classproperty
        def contents(cls):
            for event in Event.SubmissionM6502.events:
                key = {'event': event}
                key.update(cls.constant_attrs)
                cls.insert(key, ignore_extra_fields=True, skip_duplicates=True, insert_to_master=True)
            return {}
        
        def run(self, key):
            event = self.fetch1('event')
            if event in ['unit_match_attempt_m65_02']:
                m6502_key = {'unit_submission_id': key.get('m6502_submission_id')}
                data = (m6502.UnitManualMatchAttempt & m6502_key).fetch1()
                match_data = (m6502.UnitNucleusManualMatch & m6502_key).fetch(as_dict=True)
                if match_data:
                    data.update(match_data[0])
                key['user'] = data.pop('user_id')
                protocol_name = data.pop('protocol_name')
                if not protocol_name:
                    key['protocol_id'] = 0
                else:
                    key['protocol_id'] = Protocol.r1p({'protocol_name': protocol_name}).fetch1('protocol_id')
                key['coregistration'] = (m65crg.Coregistration.AIBS & {'transform_id': data.pop('transform_id')}).fetch1('coregistration')
                key['user_choice'] = data.pop('unit_submission_type')
                key['animal_id'] = data.pop('animal_id')
                key['scan_session'] = data.pop('scan_session')
                key['scan_idx'] = data.pop('scan_idx')
                key['unit_id'] = data.pop('unit_id')
                key['field'] = data.pop('field')
                key['nucleus_id'] = data.get('nucleus_id')
                key['note'] = data.pop('note')
                key['interface'] = data.pop('interface')
                key['priority'] = data.pop('priority')
                key['other_data'] = data if data else None
                return key
            elif event in ['nucleus_match_attempt_m65_02']:
                m6502_key = {'nucleus_submission_id': key.get('m6502_submission_id')}
                data = (m6502.NucleusManualMatchAttempt & m6502_key).fetch1()
                match_data = (m6502.UnitNucleusManualMatch & m6502_key).fetch(as_dict=True)
                if match_data:
                    data.update(match_data[0])
                key['user'] = data.pop('user_id')
                protocol_name = data.pop('protocol_name')
                if not protocol_name:
                    key['protocol_id'] = 0
                else:
                    key['protocol_id'] = Protocol.r1p({'protocol_name': protocol_name}).fetch1('protocol_id')
                key['coregistration'] = (m65crg.Coregistration.AIBS & {'transform_id': data.pop('transform_id')}).fetch1('coregistration')
                key['user_choice'] = data.pop('nucleus_submission_type')
                key['nucleus_id'] = data.pop('nucleus_id')
                key['scan_session'] = data.get('scan_session')
                key['scan_idx'] = data.get('scan_idx')
                key['unit_id'] = data.get('unit_id')
                key['field'] = data.get('field')
                key['note'] = data.pop('note')
                key['interface'] = data.pop('interface')
                key['priority'] = data.pop('priority')
                key['other_data'] = data if data else None
                return key

    class Submission(dju.EventHandler):
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
            if event in ['unit_match_attempt']:
                required_attrs = ['protocol_id', 'scan_session', 'scan_idx', 'unit_id', 'user_choice']
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
            elif event in ['nucleus_match_attempt']:
                return key

@schema
class User(djp.Lookup):
    hash_name = 'make_id'
    definition = f"""
    {user_attr}
    ---
    make_id : varchar(10)
    timestamp=CURRENT_TIMESTAMP : timestamp
    """

    class Add(dju.Maker):
        hash_name = 'make_id'
        upstream = Event
        method = EventHandler
        definition = """
        -> master
        -> Event
        -> EventHandler
        make_id : varchar(10)
        """
        @classproperty
        def key_source(cls):
            return djp.U('event_id', 'event_handler_id') & (Event.UserAdd * EventHandler.UserAdd)

        def on_make(self, key):
            slack_client.post_to_slack(f"```{key.get('user')} added to multimatch users```")

    class M6502(djp.Part):
        enable_hashing = True
        hash_name = 'make_id'
        hashed_attrs = 'user'
        definition = """
        -> master
        make_id : varchar(10)
        """

        @classmethod
        def fill(cls):
            df = pd.DataFrame(((djp.U('user_id') & m6502.NucleusManualMatchAttempt()) + (djp.U('user_id') & m6502.UnitManualMatchAttempt())).proj(user='user_id').fetch())
            df['make_id'] = cls.hash(df)
            cls.insert(df, insert_to_master=True, ignore_extra_fields=True, skip_hashing=True)

@schema
class Submission(djp.Lookup):
    hash_name = 'submission_id'
    definition = """
    submission_id : varchar(10)
    ---
    timestamp=CURRENT_TIMESTAMP : timestamp # 
    """

    class Add(dju.Maker):
        hash_name = 'submission_id'
        upstream = Event
        method = EventHandler
        @classproperty
        def key_source(cls):
            return (djp.U('event_id', 'event_handler_id') & (Event.Submission * EventHandler.Submission))

        def on_make(self, key):
            if key.get('event') in ['unit_match_attempt']:
                self.master.UnitMatchAttempt.insert1(key, ignore_extra_fields=True)
            
            if key.get('event') in ['nucleus_match_attempt']:
                self.master.NucleusMatchAttempt.insert1(key, ignore_extra_fields=True)

    class M6502(dju.Maker):
        hash_name = 'submission_id'
        upstream = Event
        method = EventHandler
        force = True
        @classproperty
        def key_source(cls):
            return (djp.U('event_id', 'event_handler_id') & (Event.SubmissionM6502 * EventHandler.SubmissionM6502))

        def on_make(self, key):
            if key.get('event') in ['unit_match_attempt_m65_02']:
                self.master.UnitMatchAttempt.insert1(key, ignore_extra_fields=True)
            
            if key.get('event') in ['nucleus_match_attempt_m65_02']:
                self.master.NucleusMatchAttempt.insert1(key, ignore_extra_fields=True)

    class UnitMatchAttempt(djp.Part):
        definition = f"""
        -> User
        -> Protocol
        -> m65crg.Coregistration
        -> nda.UnitSource
        -> master
        ---
        user_choice : varchar(128) # user selected choice
        field=NULL                : smallint                     # field of the scan
        nucleus_id=NULL : varchar(128) # selected nucleus_id (optional)
        -> MatchPriority
        -> MatchInterface
        note=NULL : longblob # user note (optional)
        other_data=NULL : longblob # other data (optional)
        """
    
    class NucleusMatchAttempt(djp.Part):
        definition = f"""
        -> User
        -> Protocol
        -> m65crg.Coregistration
        -> m65mat.Nucleus
        -> master
        ---
        user_choice : varchar(128) # user selected choice
        scan_session=NULL         : smallint                     # scan session index for the mouse
        scan_idx=NULL             : smallint                     # id of the scan within the session
        field=NULL                : smallint                     # field of the scan
        unit_id=NULL              : int                          # unique per scan & segmentation method
        -> MatchPriority
        -> MatchInterface
        note=NULL : longblob # user note (optional)
        other_data=NULL : longblob # other data (optional)
        """
    

schema.spawn_missing_classes()
