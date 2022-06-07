"""
Externals for DataJoint tables.
"""

from pathlib import Path

import datajoint_plus as djp

base_path = Path() / '/mnt' / 'dj-stor01' / 'microns'
minnie_stack_path = base_path / 'minnie' / 'stacks'
multimatch_stack_path = Path() / '/mnt' / 'scratch09' / 'microns' / 'stacks'

minnie_em = {
    'minnie_stacks': djp.make_store_dict(minnie_stack_path)
}

minnie65_auto_match = {
    'minnie_stacks': djp.make_store_dict(minnie_stack_path)
}

minnie65_coregistration = {
    'minnie_stacks': djp.make_store_dict(minnie_stack_path)
}

minnie65_manual_match = {
    'minnie_stacks': djp.make_store_dict(minnie_stack_path)
}

multimatch = {
    'multimatch_stacks': djp.make_store_dict(multimatch_stack_path)
}

dashboard = {}