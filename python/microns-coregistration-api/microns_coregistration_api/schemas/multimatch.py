"""
DataJoint tables for manual match.
"""

import datajoint_plus as djp

from ..config import multimatch_config as config

config.register_externals()
config.register_adapters(context=locals())

schema = djp.schema(config.schema_name, create_schema=True)

schema.spawn_missing_classes()
schema.connection.dependencies.load()