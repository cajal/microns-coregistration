from pathlib import Path
import datajoint as dj
import datajoint_plus as djp
import numpy as np
from ..config import minnie65_calcium_sim_config as config

import microns_utils.datajoint_utils as dju
from microns_utils.misc_utils import classproperty
from microns_morphology_api.schemas import minnie65_morphology as m65mor, minnie65_auto_proofreading as m65auto

config.register_externals()
config.register_adapters(context=locals())

schema = djp.schema(config.schema_name, create_schema=True)

@schema
class Tag(dju.VersionLookup):
    package = 'microns-coregistration-api'
    attr_name = 'tag'


@schema
class VoxelizedSomaMethod(djp.Lookup):
    hash_name = 'vx_soma_method'
    definition = f"""
    {hash_name} : varchar(6)
    """

    class Trimesh(djp.Part):
        enable_hashing = True
        hash_name = 'vx_soma_method'
        hashed_attrs = 'voxel_res', 'use_convex_hull', 'fill', 'voxel_method', 'voxel_method_kws', Tag.attr_name
        definition = f"""
        -> master
        ---
        voxel_res    : float # resolution (nm)
        use_convex_hull : tinyint # 1 if convex_hull is used, 0 otherwise
        fill : tinyint # 1 if voxelization is filled, 0 otherwise
        voxel_method : varchar(64) # method to pass to trimesh.voxel.creation.voxelize
        voxel_method_kws=NULL : varchar(1000) # kwargs to pass to trimesh.voxel.creation.voxelize
        -> Tag
        """


@schema
class VoxelizedSoma(djp.Lookup):
    hash_name = 'vx_soma_id'
    definition = f"""
    {hash_name} : varchar(16)
    """
    
    @classproperty
    def base_path(cls):
        return Path(config.externals['voxelized_soma_files']['location'])
        
    class Store(djp.Part):
        hash_name = 'vx_soma_id'
        definition = """
        -> master
        segment_id           : bigint unsigned              # id of the segment under the nucleus centroid. Equivalent to Allen 'pt_root_id'.
        split_index          : tinyint unsigned             # the index of the neuron object that resulted AFTER THE SPLITTING ALGORITHM 
        soma_index           : tinyint unsigned             #
        nucleus_id           : int unsigned                 # id of nucleus from the flat segmentation  Equivalent to Allen: 'id'.
        ---
        volume               : int                          # mesh volume (um^3)
        vx_soma              : <voxelized_soma_files>        # voxelized soma mesh file
        """
    
    class Maker(djp.Part, dj.Computed):
        enable_hashing = True
        hash_name = 'vx_soma_id'
        hashed_attrs = list(set(m65auto.AutoProofreadNeuron.primary_key).union(m65mor.MeshFragment.SomaObject.primary_key).union(VoxelizedSomaMethod.primary_key))
        definition = """
        -> master
        -> m65auto.AutoProofreadNeuron
        -> m65mor.MeshFragment.SomaObject
        nucleus_id           : int unsigned                 # id of nucleus from the flat segmentation  Equivalent to Allen: 'id'.
        -> VoxelizedSomaMethod
        ---
        ts_inserted=CURRENT_TIMESTAMP : timestamp
        """

        @classmethod
        def make_filepath(cls, vx_soma_id, nucleus_id, segment_id, split_index, soma_index, suffix='.npz', with_basepath=True, **kwargs):
            fp = Path(f'{vx_soma_id}_{nucleus_id}_{segment_id}_{split_index}_{soma_index}').with_suffix(suffix)
            if with_basepath:
                fp = cls.master.base_path.joinpath(fp)
            return fp