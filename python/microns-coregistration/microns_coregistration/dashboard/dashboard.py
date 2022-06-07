"""
DataJoint tables for dashboard
"""

import datajoint_plus as djp

from microns_coregistration_api.schemas import dashboard as db

schema = db.schema
config = db.config

logger = djp.getLogger(__name__)