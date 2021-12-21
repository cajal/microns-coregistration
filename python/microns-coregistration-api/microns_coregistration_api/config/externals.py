"""
Externals for DataJoint tables.
"""

from pathlib import Path
import datajoint.datajoint_plus as djp

base_path = Path() / '/mnt' / 'dj-stor01' / 'microns'
minnie_stack_path = base_path / 'minnie' / 'stacks'

minnie_em = {
    'stacks': djp.make_store_dict(minnie_stack_path)
}

minnie65_auto_match = {
    'stacks': djp.make_store_dict(minnie_stack_path)
}

minnie65_coregistration = {
    'stacks': djp.make_store_dict(minnie_stack_path)
}

minnie65_manual_match = {
    'stacks': djp.make_store_dict(minnie_stack_path)
}