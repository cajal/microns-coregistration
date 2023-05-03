import datajoint as dj
import datajoint_plus as djp

from ..config import minnie65_boundaries_config as config
import microns_utils.datajoint_utils as dju

config.register_externals()
config.register_adapters(context=locals())

schema = djp.schema(config.schema_name, create_schema=True)

schema.spawn_missing_classes()

@schema
class Tag(dju.VersionLookup):
    package = 'microns-coregistration-api'
    attr_name = 'tag'