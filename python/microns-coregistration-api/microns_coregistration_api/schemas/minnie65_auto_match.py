"""
DataJoint tables for automatch.
"""
import datajoint as dj
import datajoint_plus as djp

from ..config import minnie65_auto_match_config as config

config.register_externals()
config.register_adapters(context=locals())

schema = djp.schema(config.schema_name, create_schema=True)

schema.spawn_missing_classes()
