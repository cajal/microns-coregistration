"""
DataJoint tables for coregistration of minnie65 electron microscopy and two-photon data.
"""
import datajoint as dj
import datajoint_plus as djp

from ..config import minnie65_coregistration_config as config

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

schema.spawn_missing_classes()
schema.connection.dependencies.load()