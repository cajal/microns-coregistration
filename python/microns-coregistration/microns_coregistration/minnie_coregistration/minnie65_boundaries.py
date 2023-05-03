"""
DataJoint tables for computing boundaries in minnie65
"""

import datajoint_plus as djp

from microns_coregistration_api.schemas import minnie65_manual_match as m65man

schema = m65man.schema
config = m65man.config

logger = djp.getLogger(__name__)

