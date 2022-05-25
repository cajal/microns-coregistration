"""
DataJoint tables for coregistration of minnie65 electron microscopy and two-photon data.
"""
import datajoint as dj
import datajoint_plus as djp

from ..config import minnie65_coregistration_config as config
from microns_nda_api.schemas import minnie_nda as nda
from ..utils.coregistration_utils import aibs_coreg_transform
from microns_utils.misc_utils import unwrap

config.register_externals()
config.register_adapters(context=locals())

schema = djp.schema(config.schema_name, create_schema=True)

@schema
class Adjustment(djp.Lookup):
    definition = """
    # types of adjustments
    adjustment     : varchar(48)    # name of adjustment type 
    ---
    description=NULL    : varchar(450)   # details of adjustment (optional)
    """
    
    contents = [
        {'adjustment': 'resize', 'description': 'adjustment to source resolution (e.g. px / cm) but not source dimensionality (# pixels does not change)'},
        {'adjustment': 'resample', 'description': 'adjustment to source dimensionality (e.g. interpolation on a new grid)'},
        {'adjustment': 'crop', 'description': ''}, 
        {'adjustment': 'translation', 'description': 'constant offset added to source grid'},
    ]


@schema
class AdjustmentSet(djp.Lookup):
    enable_hashing = True
    hash_name = 'adjustment_set'
    hashed_attrs = Adjustment.heading.primary_key
    hash_group = True
    definition = """
    # sets of adjustments 
    adjustment_set     : varchar(8)          # adjustment set hash
    ---
    name=NULL          : varchar(48)         # name of adjustment set (optional)
    description=NULL   : varchar(450)        # details of adjustment set (optional)
    ts_inserted=CURRENT_TIMESTAMP : timestamp
    """

    class Member(djp.Part):
        definition = """
        # sets of adjustments
        -> master
        -> Adjustment
        """


@schema
class TPStack(djp.Lookup):
    definition = """
    -> nda.Stack
    """
    contents = [{'animal_id': 17797, 'stack_session': 9, 'stack_idx': 19}]


@schema
class Coregistration(djp.Lookup):
    hash_name = 'coregistration'
    definition = """
    coregistration : varchar(10) # hash of coregistration method
    """

    @classmethod
    def run(cls, coords, key=None, transform_id=None):
        assert (key is not None) ^ (transform_id is not None), 'Provide key or transform_id but not both'
        if transform_id is not None:
            key = {'transform_id': transform_id}
        return cls.r1p(key).run(coords, **key)

    class AIBS(djp.Part):
        enable_hashing = True
        hash_name = 'coregistration'
        hashed_attrs = TPStack.primary_key + ['transform_id', 'version', 'direction', 'transform_type']
        definition = """
        -> master
        ---
        -> TPStack
        transform_id         : int                          # id of the transform
        version              : varchar(256)                 # coordinate framework
        direction            : varchar(16)                  # direction of the transform (EMTP: EM -> 2P, TPEM: 2P -> EM)
        transform_type       : varchar(16)                  # linear (more rigid) or spline (more nonrigid)
        transform_args=null  : longblob                     # parameters of the transform
        transform_solution=null : longblob                     # transform solution
        """

        def run(self, coords, **kwargs):
            params = unwrap(self.fetch('version', 'direction', 'transform_type', 'transform_solution', as_dict=True))
            return aibs_coreg_transform(coords, **params)
            
        
schema.spawn_missing_classes()
schema.connection.dependencies.load()