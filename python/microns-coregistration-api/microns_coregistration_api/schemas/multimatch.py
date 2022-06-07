"""
DataJoint tables for multimatch.
"""

import datajoint_plus as djp
import datajoint as dj

from ..config import multimatch_config as config

config.register_externals()
config.register_adapters(context=locals())

schema = djp.schema(config.schema_name, create_schema=True)