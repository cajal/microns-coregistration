"""
Adapters for DataJoint tables.
"""
from microns_utils.adapter_utils import NumpyAdapter, JsonAdapter, PandasPickleAdapter, PickleFilepathAdapter

minnie_stacks = NumpyAdapter('filepath@minnie_stacks')
multimatch_stacks = NumpyAdapter('filepath@multimatch_stacks')
cell_typer_files = JsonAdapter('filepath@cell_typer_files')
layer_boundary_pickle_files = PickleFilepathAdapter('filepath@layer_boundary_pickle_files')
layer_boundary_pandas_pickle_files = PandasPickleAdapter('filepath@layer_boundary_pandas_pickle_files')
layer_boundary_numpy_files = NumpyAdapter('filepath@layer_boundary_numpy_files')
field_image_files = NumpyAdapter('filepath@field_image_files')
field_grid_files = NumpyAdapter('filepath@field_grid_files')
soma_mesh_files = NumpyAdapter('filepath@soma_mesh_files')
voxelized_soma_files = NumpyAdapter('filepath@voxelized_soma_files')
calcium_sim_soma_files = NumpyAdapter('filepath@calcium_sim_soma_files')
match_tables = PandasPickleAdapter('filepath@match_tables')


minnie_em = {
    'minnie_stacks': minnie_stacks
}
minnie65_auto_match = {
    'match_tables': match_tables
}
minnie65_coregistration = {
    
}
minnie65_manual_match = {
    'match_tables': match_tables
}
multimatch = {
    'multimatch_stacks': multimatch_stacks
}
cell_typer = {
    'cell_typer_files': cell_typer_files
}
minnie65_boundaries = {
    'layer_boundary_pickle_files': layer_boundary_pickle_files,
    'layer_boundary_pandas_pickle_files': layer_boundary_pandas_pickle_files,
    'layer_boundary_numpy_files': layer_boundary_numpy_files,
}
minnie_field_images = {
    'field_image_files': field_image_files,
    'field_grid_files': field_grid_files,
}
minnie65_calcium_sim = {
    'soma_mesh_files': soma_mesh_files,
    'voxelized_soma_files': voxelized_soma_files,
    'calcium_sim_soma_files': calcium_sim_soma_files,
}
