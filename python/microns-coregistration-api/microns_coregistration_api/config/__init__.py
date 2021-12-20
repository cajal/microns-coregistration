"""
Configuration package/module for microns-coregistration.
"""
import datajoint.datajoint_plus as djp
from microns_utils.config_utils import SchemaConfig
from . import adapters
from . import externals

djp.enable_datajoint_flags()

minnie_em_config = SchemaConfig(
    module_name='minnie_em',
    schema_name='microns_minnie_em',
    externals=externals.minnie_em,
    adapters=adapters.minnie_em
)

minnie65_auto_match_config = SchemaConfig(
    module_name='minnie65_auto_match',
    schema_name='microns_minnie65_auto_match',
    externals=externals.minnie65_auto_match,
    adapters=adapters.minnie65_auto_match
)

minnie65_coregistration_config = SchemaConfig(
    module_name='minnie65_coregistration',
    schema_name='microns_minnie65_coregistration',
    externals=externals.minnie65_coregistration,
    adapters=adapters.minnie65_coregistration
)

minnie65_manual_match_config = SchemaConfig(
    module_name='minnie65_manual_match',
    schema_name='microns_minnie65_manual_match',
    externals=externals.minnie65_manual_match,
    adapters=adapters.minnie65_manual_match
)
