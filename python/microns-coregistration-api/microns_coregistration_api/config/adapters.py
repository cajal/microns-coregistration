"""
Adapters for DataJoint tables.
"""

from microns_utils.adapter_utils import NumpyAdapter, JsonAdapter

minnie_stacks = NumpyAdapter('filepath@minnie_stacks')
multimatch_stacks = NumpyAdapter('filepath@multimatch_stacks')
dashboard = JsonAdapter('filepath@dashboard')
cell_type_trainer = JsonAdapter('filepath@cell_type_trainer')

minnie_em = {'minnie_stacks': minnie_stacks}
minnie65_auto_match = {}
minnie65_coregistration = {}
minnie65_manual_match = {}
multimatch = {'multimatch_stacks': multimatch_stacks}
dashboard = {'dashboard': dashboard}
cell_type_trainer = {'cell_type_trainer': cell_type_trainer}