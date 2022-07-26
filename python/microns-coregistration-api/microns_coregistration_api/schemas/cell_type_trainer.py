"""
DataJoint tables for cell type training.
"""

import json
from pathlib import Path
import datajoint_plus as djp
import datajoint as dj
from microns_utils.misc_utils import classproperty
from microns_utils.datetime_utils import current_timestamp
from microns_utils.version_utils import check_package_version

from microns_materialization_api.schemas import \
    minnie65_materialization as m65mat

from ..config import cell_type_trainer_config as config

config.register_externals()
config.register_adapters(context=locals())

from . import dashboard as db

schema = djp.schema(config.schema_name, create_schema=True)

@schema
class Tag(djp.Lookup):
    definition = """
    tag : varchar(32) # github tag of repository
    """

    @classproperty
    def contents(cls):
        cls.insert1({
            'tag': check_package_version(package='microns-coregistration-api')
        }, ignore_extra_fields=True, skip_duplicates=True)
        return {}


@schema
class User(djp.Lookup):
    definition = """
    user                 : varchar(450)                 # jupyterhub username
    """

    @classmethod
    def on_access(cls, user):
        if len(cls & {'user': user}) == 0:
            cls.add_user(user)

    @classmethod
    def add_user(cls, user):
        cls.insert1({'user': user})

@schema
class UserInputType(djp.Lookup):
    hash_name = 'user_input_type_id'
    enable_hashing = True
    hashed_attrs = 'user_input_type'
    definition = """
    user_input_type_id : varchar(6)
    ---
    user_input_type : varchar(450)
    """
    contents = [{'user_input_type': 'submission'}]


@schema
class UserInput(djp.Lookup):
    hash_name = 'user_input_id'
    definition = """
    user_input_id : varchar(16)
    """

    class Submission(djp.Part):
        enable_hashing = True
        hash_name = 'user_input_id'
        hashed_attrs = ['user', 'user_input_type_id', 'tag', 'ts_inserted']
        definition = """
        -> master
        ---
        -> User
        -> UserInputType
        -> Tag
        ts_inserted : timestamp
        submission : <cell_type_trainer> # json files
        """

        @classmethod
        def on_input(cls, user, submission):
            to_hash = {
                'user': user, 
                'user_input_type_id': (UserInputType & {'user_input_type': 'submission'}).fetch1('user_input_type_id'),
                'tag': check_package_version('microns-coregistration-api'),
                'ts_inserted': current_timestamp('US/Central', fmt="%Y-%m-%d_%H:%M:%S")
            }
            user_input_id = cls.hash1(to_hash)
            filename = Path(config.externals.get('cell_type_trainer').get('location')).joinpath(user_input_id).with_suffix('.json') 
            with open(filename, "w") as f:
                f.write(json.dumps(submission))
            cls.insert1({**to_hash, **{'submission': filename}}, insert_to_master=True)

@schema
class Submission(djp.Lookup):
    hash_name = 'user_input_id'
    definition = """
    user_input_id : varchar(32) # id of submission
    """

    class View:
        def __new__(cls):
            return Submission.Store * UserInput.Submission

    class Exclude(djp.Part):
        hash_name = 'user_input_id'
        definition = """
        -> master
        """

    class Store(djp.Part):
        hash_name = 'user_input_id'
        definition = """
        -> master
        ---
        -> m65mat.Segment
        segment_cell_type : varchar(256)
        segment_cell_type_prob : float
        user_choice : varchar(256)
        user_note : varchar(1000)
        """

    class Maker(djp.Part, dj.Computed):
        hash_name = 'user_input_id'
        definition = """
        -> master
        -> master.Store
        -> UserInput
        ---
        ts_made=CURRENT_TIMESTAMP : timestamp # 
        """

        @property
        def key_source(self):
            return UserInput.Submission.proj() - Submission.Exclude
        
        def make(self, key):
            submission = UserInput.r1p(key).fetch1('submission')
            try:
                self.master.Store.insert1({**key, **submission}, insert_to_master=True)
            except:
                self.master.Exclude.insert1(key, insert_to_master=True)


schema.spawn_missing_classes()
schema.connection.dependencies.load()