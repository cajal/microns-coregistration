import datajoint_plus as djp
from . import multimatch, dashboard

djp.reassign_master_attribute(multimatch)
djp.reassign_master_attribute(dashboard)