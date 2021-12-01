import datajoint as dj
import numpy as np
import pandas as pd
import json

from microns_coregistration_api import config
schema_obj = config.SCHEMAS.MINNIE65_AUTO_MATCH

config.register_adapters(schema_obj, context=locals())
config.register_externals(schema_obj)

schema = dj.schema(schema_obj.value)
schema.spawn_missing_classes()
schema.connection.dependencies.load()
