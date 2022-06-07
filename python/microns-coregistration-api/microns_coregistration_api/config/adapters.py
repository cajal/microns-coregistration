"""
Adapters for DataJoint tables.
"""

from microns_utils.adapter_utils import NumpyAdapter

minnie_stacks = NumpyAdapter('filepath@minnie_stacks')
multimatch_stacks = NumpyAdapter('filepath@multimatch_stacks')

minnie_em = {'minnie_stacks': minnie_stacks}
minnie65_auto_match = {}
minnie65_coregistration = {}
minnie65_manual_match = {}
multimatch = {'multimatch_stacks': multimatch_stacks}
dashboard = {}