"""
DataJoint tables for computing EM -> 2P calcium simulations in minnie65
"""
import json
from pathlib import Path
import numpy as np

import datajoint as dj
import datajoint_plus as djp

from microns_coregistration_api.methods import registration as reg
import microns_utils.transform_utils as tu
from microns_utils.misc_utils import classproperty

from microns_coregistration_api.schemas import minnie65_calcium_sim as m65cs

from microns_morphology_api.schemas import minnie65_morphology as m65mor, minnie65_auto_proofreading as m65auto

schema = m65cs.schema
config = m65cs.config

logger = djp.getLogger(__name__)

class Tag(m65cs.Tag):
    pass

class VoxelizedSomaMethod(m65cs.VoxelizedSomaMethod):

    class Trimesh(m65cs.VoxelizedSomaMethod.Trimesh):
        @classmethod
        def update_method(cls, voxel_res, use_convex_hull, fill, voxel_method, voxel_method_kws=None):
            cls.insert1(
                {
                    'voxel_res': voxel_res,
                    'use_convex_hull': 1 * use_convex_hull,
                    'fill': 1 * fill,
                    'voxel_method': voxel_method,
                    'voxel_method_kws': voxel_method_kws if voxel_method_kws is None else json.dumps(voxel_method_kws),
                    m65cs.Tag.attr_name: m65cs.Tag.version
                }, insert_to_master=True
            )
        
        def run(self, mesh, **kwargs):
            params = (self & kwargs).fetch1()
            voxel_res = params.get('voxel_res')
            use_convex_hull = params.get('use_convex_hull')
            fill = params.get('fill')
            voxel_method = params.get('voxel_method')
            voxel_method_kws = params.get('voxel_method_kws')
            voxel_method_kws = json.loads(params.get('voxel_method_kws')) if voxel_method_kws is not None else {}
            if use_convex_hull:
                mesh = mesh.convex_hull
            vx_mesh = mesh.voxelized(pitch=voxel_res, method=voxel_method, **voxel_method_kws)
            if fill:
                vx_mesh = vx_mesh.fill()
            return vx_mesh
    

class VoxelizedSoma(m65cs.VoxelizedSoma):

    class Store(m65cs.VoxelizedSoma.Store):
        pass
    
    class Maker(m65cs.VoxelizedSoma.Maker):
        @property
        def key_source(self):
            return (m65mor.MeshFragment.SomaObject * (dj.U('nucleus_id') * m65auto.AutoProofreadNeuron) & 'nucleus_id>0' & 'multiplicity=1' & 'cell_type="excitatory"') * VoxelizedSomaMethod
        
        def make(self, key):
            key['vx_soma_id'] = self.hash1(key)
            mesh = (m65mor.MeshFragment.SomaObject & key).fetch1('mesh')
            vx_mesh = VoxelizedSomaMethod.r1p(key).run(mesh=mesh)
            def roundint(x): return np.round(x).astype(int)
            points = roundint(vx_mesh.points)
            bounds = roundint(vx_mesh.bounds)
            volume = roundint(vx_mesh.volume / 10**9)
            fp = self.make_filepath(**key)
            np.savez_compressed(fp, points=points, bounds=bounds, volume=volume)
            key['volume'] = volume
            key['vx_soma'] = fp
            self.master.Store.insert1(key, ignore_extra_fields=True, insert_to_master=True)
            self.insert1(key, skip_hashing=True, ignore_extra_fields=True)