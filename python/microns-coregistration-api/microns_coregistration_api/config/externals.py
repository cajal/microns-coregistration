"""
Externals for DataJoint tables.
"""

from pathlib import Path

import datajoint_plus as djp

base_path = Path() / '/mnt' / 'dj-stor01' / 'microns'
base_path2 = Path() / '/mnt' / 'dj-stor02' / 'microns'
minnie_stack_path = base_path / 'minnie' / 'stacks'
multimatch_stack_path = Path() / '/mnt' / 'scratch09' / 'microns' / 'stacks'
dashboard_path = base_path / 'minnie' / 'dashboard'
cell_typer_path = dashboard_path / 'cell_typer_files'
layer_boundary_path = base_path / 'minnie' / 'layer_boundaries'
calcium_sim_base_path = base_path / 'minnie' / 'calcium_sim'
soma_mesh_files = calcium_sim_base_path / 'soma_meshes'
voxelized_soma_files = calcium_sim_base_path / 'voxelized_somas'
calcium_sim_soma_files = calcium_sim_base_path / 'calcium_sim_somas'
field_image_files = base_path2 / 'minnie' / 'field_images'
field_grid_files = base_path2 / 'minnie' / 'field_grids'
match_tables_path = base_path2 / 'minnie' / 'match_tables'

minnie_em = {
    'minnie_stacks': djp.make_store_dict(minnie_stack_path)
}

minnie65_auto_match = {
    'minnie_stacks': djp.make_store_dict(minnie_stack_path),
    'match_tables': djp.make_store_dict(match_tables_path)
}

minnie65_coregistration = {
    'minnie_stacks': djp.make_store_dict(minnie_stack_path)
}

minnie65_manual_match = {
    'minnie_stacks': djp.make_store_dict(minnie_stack_path),
    'match_tables': djp.make_store_dict(match_tables_path)
}

multimatch = {
    'multimatch_stacks': djp.make_store_dict(multimatch_stack_path),
}

cell_typer = {
    'cell_typer_files': djp.make_store_dict(cell_typer_path)
}

minnie65_boundaries = {
    'layer_boundary_pickle_files': djp.make_store_dict(layer_boundary_path),
    'layer_boundary_pandas_pickle_files': djp.make_store_dict(layer_boundary_path),
    'layer_boundary_numpy_files': djp.make_store_dict(layer_boundary_path)
}

minnie_field_images = {
    'field_image_files': djp.make_store_dict(field_image_files),
    'field_grid_files': djp.make_store_dict(field_grid_files),
}

minnie65_calcium_sim = {
    'soma_mesh_files': djp.make_store_dict(soma_mesh_files),
    'voxelized_soma_files': djp.make_store_dict(voxelized_soma_files),
    'calcium_sim_soma_files': djp.make_store_dict(calcium_sim_soma_files),
}