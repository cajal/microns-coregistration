"""
DataJoint tables for Coregistration Dashboard.
"""
import os
import datajoint_plus as djp
import datajoint as dj
from microns_utils.datetime_utils import current_timestamp
from microns_utils.version_utils import check_package_version
from microns_utils.misc_utils import classproperty, unwrap
from ..utils.slack_utils import SlackForWidget
from ..config import dashboard_config as config

config.register_externals()
config.register_adapters(context=locals())

schema = djp.schema(config.schema_name, create_schema=True)

slack_client = SlackForWidget()

os.environ['DJ_LOGLEVEL'] ='WARNING'
logger = djp.getLogger(__name__, level='WARNING', update_root_level=True)

@schema
class Tag(djp.Lookup):
    definition = """
    tag : varchar(32) # github tag of repository
    """
    contents = [{'tag': check_package_version(package='microns-coregistration-api')}]


@schema
class User(djp.Lookup):
    definition = """
    user : varchar(450) # jupyterhub username
    """

    @classmethod
    def on_dashboard_access(cls, user):
        if len(cls & {'user': user}) == 0:
            cls.Log('info', f'Dashboard user {user} not found in database. Adding new user...')
            cls.add_user(user)
        UserEvent.DashboardAccess.add_event(user)

    @classmethod
    def add_user(cls, user):
        cls.insert1({'user': user})
        cls.Log('info', f'Dashboard user {user} added.')


@schema
class UserInputType(djp.Lookup):
    hash_name = 'user_input_type_id'
    definition = """
    user_input_type_id : varchar(12)
    """

    class Username(djp.Part, dj.Lookup):
        enable_hashing = True
        hash_name = 'user_input_type_id'
        hashed_attrs = 'user_input_type'
        definition = """
        -> master
        ---
        user_input_type : varchar(48)
        """

        @classproperty
        def contents(cls):
            contents = [
                {'user_input_type': 'slack_username'},
            ]
            for content in contents:
                cls.master.insert1({cls.master.hash_name: cls.hash1(content)}, ignore_extra_fields=True, skip_duplicates=True)
            return contents


@schema
class UserInput(djp.Lookup):
    hash_name = 'user_input_id'
    definition = """
    user_input_id : varchar(24) # hash of user input
    """
    @classmethod
    def get(cls, key):
        return cls.r1p(key).get(key)
        
    class Username(djp.Part):
        enable_hashing = True
        hash_name = 'user_input_id'
        hashed_attrs = 'user', 'user_input_type_id', 'ts_inserted'
        definition = """
        -> master
        ---
        -> User
        -> UserInputType
        user_input: varchar(450)
        ts_inserted : timestamp
        """

        def get(self, key):
            return unwrap((self & key).fetch('user', 'user_input_type_id', 'user_input', 'user_input_id', as_dict=True))

        @classmethod
        def on_input(cls, user, user_input_type, user_input):
            user_input_type_id = UserInputType.Username.hash1({'user_input_type': user_input_type})
            cls.insert1({
                'user': user, 
                'user_input_type_id': user_input_type_id,
                'ts_inserted': current_timestamp('US/Central', fmt="%Y-%m-%d_%H:%M:%S"),
                'user_input': user_input
                }, insert_to_master=True)
            UserEvent.Input.add_event(user, user_input_type_id, cls.get_latest_entries().fetch1('user_input_id'))
            Username.populate()

# METHODS

@schema
class UserInputHandler(djp.Lookup):
    hash_name = 'user_input_handler_id'
    definition = """
    user_input_handler_id : varchar(10) # hash of method of handling user input
    """

    @classmethod
    def run(cls, key):
        return cls.r1p(key).run(**key)

    class Username(djp.Part, dj.Lookup):
        enable_hashing = True
        hash_name = 'user_input_handler_id'
        hashed_attrs = 'tag'
        definition = """
        -> master
        ---
        -> Tag
        ts_inserted=CURRENT_TIMESTAMP : timestamp
        """
        @classproperty
        def contents(cls):
            contents = Tag.fetch(as_dict=True)
            for content in contents:
                cls.master.insert1({cls.master.hash_name: cls.hash1(content)}, ignore_extra_fields=True, skip_duplicates=True)
            return contents

        def run(self, **kwargs):
            params = self.fetch1()
            
            # VALIDATION
            method_tag = params['tag']
            current_tag = check_package_version('microns-coregistration-api')
            if method_tag != current_tag:
                msg = f"This method requires tag to be {method_tag}, but currently is {current_tag}. Create a new method."
                self.master.Log('error', msg)
                raise Exception(msg)


            user_input_type = UserInputType.r1pwh(kwargs['user_input_type_id']).fetch1('user_input_type')
            if user_input_type == 'slack_username':
                try:
                    return {'username': slack_client.get_slack_username(kwargs['user_input']), 'user': kwargs['user'], 'user_input_type': user_input_type}
                except:
                    self.master.Log('exception', f'Error getting slack username with user input: {kwargs["user_input"]}')
                    return {'username': ''}
                
            
# ENTITIES

@schema
class Username(djp.Lookup):
    hash_name = 'username_id'
    definition = """
    username_id : varchar(6) # hash of username
    """

    @classmethod
    def populate(cls):
        cls.Maker.populate()
    
    class Slack(djp.Part):
        hash_name = 'username_id'
        definition = """
        -> User
        ---
        -> master
        username : varchar(128)
        """
    
    class Exclude(djp.Part):
        hash_name = 'username_id'
        definition = """
        -> master
        """

    class Maker(djp.Part, dj.Computed):
        enable_hashing = True
        hash_name = 'username_id'
        hashed_attrs = 'user_input_id', 'user_input_handler_id'
        definition = """
        -> UserInput
        -> UserInputHandler
        -> master
        ---
        ts_inserted=CURRENT_TIMESTAMP : timestamp
        """

        @property
        def key_source(self):
            return UserInput.Username.proj() * UserInputHandler.Username.get_latest_entries().proj()

        def make(self, key):
            self.master.Log('info', f'Populating {self.class_name}')
            key = {**key, **UserInput.get(key)}
            key = {**key, **UserInputHandler.run(key)}
            key = {**key, **{self.hash_name: self.hash1(key)}}

            if key['username']:
                if key['user_input_type'] == 'slack_username':
                    self.master.Log('info', f'Populating {self.master.Slack.class_name}')
                    self.master.Slack.insert1(key, insert_to_master=True, ignore_extra_fields=True, skip_duplicates=True)
                    self.insert1(key, skip_hashing=True, ignore_extra_fields=True)
            else:
                self.master.Log('info', f'Populating {self.master.Exclude.class_name}')
                self.master.Exclude.insert1(key, insert_to_master=True, ignore_extra_fields=True)
                self.insert1(key, skip_hashing=True, ignore_extra_fields=True)
            

# LOG

@schema
class UserEvent(djp.Lookup):
    hash_name = 'user_log_event'
    definition = """
    user_log_event : varchar(24) # hash of user event
    """

    class DashboardAccess(djp.Part):
        enable_hashing = True
        hash_name = 'user_log_event'
        hashed_attrs = 'user', 'timestamp'
        definition = """
        -> User
        -> master
        ---
        timestamp : timestamp
        """
        @classmethod
        def add_event(cls, user):
            cls.master.Log('info', f'User {user} accessed dashboard.')
            cls.insert1({
                'user': user, 
                'timestamp': current_timestamp('US/Central', fmt="%Y-%m-%d_%H:%M:%S")
            }, insert_to_master=True)
    
    class Input(djp.Part):
        enable_hashing = True
        hash_name = 'user_log_event'
        hashed_attrs = 'user', 'user_input_id', 'user_input_type_id', 'timestamp'
        definition = """
        -> User
        -> UserInputType
        -> UserInput
        -> master
        ---
        timestamp : timestamp
        """
        @classmethod
        def add_event(cls, user, user_input_type_id, user_input_id):
            user_input_type = UserInputType.r1pwh(user_input_type_id).fetch1('user_input_type')
            cls.master.Log('info', f'User {user} input {user_input_type}. user_input_id: {user_input_id}.')
            cls.insert1({
                'user': user,  
                'user_input_type_id': user_input_type_id, 
                'user_input_id': user_input_id, 
                'timestamp': current_timestamp('US/Central', fmt="%Y-%m-%d_%H:%M:%S")
            }, insert_to_master=True)

schema.spawn_missing_classes()
schema.connection.dependencies.load()