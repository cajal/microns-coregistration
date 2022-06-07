"""
DataJoint tables for multimatch
"""

import datajoint_plus as djp

from microns_coregistration_api.schemas import multimatch as mm

schema = mm.schema
config = mm.config

logger = djp.getLogger(__name__)