"""
DataJoint tables for auto-match
"""

import datajoint_plus as djp

from microns_coregistration_api.schemas import minnie65_auto_match as m65auto

schema = m65auto.schema
config = m65auto.config

logger = djp.getLogger(__name__)