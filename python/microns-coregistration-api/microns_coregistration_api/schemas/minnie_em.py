"""
DataJoint tables for processing of minnie EM data.
"""
import datajoint as dj
import datajoint_plus as djp

from ..config import minnie_em_config as config

config.register_externals()
config.register_adapters(context=locals())

schema = djp.schema(config.schema_name, create_schema=True)

from . import minnie65_coregistration as m65crg

@schema
class EM(djp.Lookup):
    definition = """
    # EM dataset Info
    em_name                : varchar(12)      # name of em dataset
    alignment              : tinyint          # id of alignment
    ---
    res_x                  : float            # resolution x, nanometers/ voxel
    res_y                  : float            # resolution y, nanometers/ voxel
    res_z                  : float            # resolution z, nanometers/ voxel
    ctr_pt_x               : float            # center of volume x, voxels
    ctr_pt_y               : float            # center of volume y, voxels
    ctr_pt_z               : float            # center of volume z, voxels
    min_pt_x               : float            # min point of volume bounds x, voxels
    min_pt_y               : float            # min point of volume bounds y, voxels
    min_pt_z               : float            # min point of volume bounds z, voxels
    max_pt_x               : float            # max point of volume bounds x, voxels
    max_pt_y               : float            # max point of volume bounds y, voxels
    max_pt_z               : float            # max point of volume bounds z, voxels
    min_pt_x_anat          : varchar(24)      # min point x, anatomical reference
    min_pt_y_anat          : varchar(24)      # min point y, anatomical reference
    min_pt_z_anat          : varchar(24)      # min point z, anatomical reference
    max_pt_x_anat          : varchar(24)      # max point x, anatomical reference
    max_pt_y_anat          : varchar(24)      # max point y, anatomical reference
    max_pt_z_anat          : varchar(24)      # max point z, anatomical reference
    cv_path                : varchar(450)     # cloudvolume path (from Princeton)  
    description            : varchar(450)     # description of alignment
    """


@schema
class EMAdjusted(djp.Lookup):
    hash_name = 'em_adjusted'
    hash_part_table_names = True
    definition = """
    # Adjusted versions of aligned em volume
    -> EM
    em_adjusted                 :    varchar(8)       # hash of adjusted em volume
    ---
    ts_inserted=CURRENT_TIMESTAMP : timestamp
    """

    class CloudVolume(djp.Part):
        enable_hashing = True
        hash_name = 'em_adjusted'
        hashed_attrs = ['res_x', 'res_y', 'res_z', 'ctr_pt_x', 'ctr_pt_y', 'ctr_pt_z', 'min_pt_x', 'min_pt_y', 'min_pt_z', 'max_pt_x', 'max_pt_y', 'max_pt_z']
        definition = """
        # adjusted EM versions from cloudvolume
        -> master
        ---
        -> m65crg.AdjustmentSet
        mip=NULL               : tinyint          # mip
        res_x                  : float            # resolution x, nanometers/ voxel
        res_y                  : float            # resolution y, nanometers/ voxel
        res_z                  : float            # resolution z, nanometers/ voxel
        ctr_pt_x               : float            # center of volume x, voxels
        ctr_pt_y               : float            # center of volume y, voxels
        ctr_pt_z               : float            # center of volume z, voxels
        min_pt_x               : float            # min point of volume bounds x, voxels
        min_pt_y               : float            # min point of volume bounds y, voxels
        min_pt_z               : float            # min point of volume bounds z, voxels
        max_pt_x               : float            # max point of volume bounds x, voxels
        max_pt_y               : float            # max point of volume bounds y, voxels
        max_pt_z               : float            # max point of volume bounds z, voxels
        voxel_offset_x         : float            # voxel offset x
        voxel_offset_y         : float            # voxel offset y
        voxel_offset_z         : float            # voxel offset z
        """

    class Custom(djp.Part):
        enable_hashing = True
        hash_name = 'em_adjusted'
        hashed_attrs = ['res_x', 'res_y', 'res_z', 'ctr_pt_x', 'ctr_pt_y', 'ctr_pt_z', 'min_pt_x', 'min_pt_y', 'min_pt_z', 'max_pt_x', 'max_pt_y', 'max_pt_z']

        definition = """
        #  custom adjustments to EM volume 
        -> master
        ---
        -> m65crg.AdjustmentSet
        res_x                  : float            # resolution x, nanometers/ voxel
        res_y                  : float            # resolution y, nanometers/ voxel
        res_z                  : float            # resolution z, nanometers/ voxel
        ctr_pt_x               : float            # center of volume x, voxels
        ctr_pt_y               : float            # center of volume y, voxels
        ctr_pt_z               : float            # center of volume z, voxels
        min_pt_x               : float            # min point of volume bounds x, voxels
        min_pt_y               : float            # min point of volume bounds y, voxels
        min_pt_z               : float            # min point of volume bounds z, voxels
        max_pt_x               : float            # max point of volume bounds x, voxels
        max_pt_y               : float            # max point of volume bounds y, voxels
        max_pt_z               : float            # max point of volume bounds z, voxels
        description            : varchar(450)     # adjustment details
        """


@schema
class EMSeg(djp.Lookup):
    hash_part_table_names = True
    hash_name = 'em_seg'
    definition = """
    # Segmentations of aligned EM volumes
    -> EM
    em_seg                     :    varchar(8)      # hash of em segmentation
    ---
    ts_inserted=CURRENT_TIMESTAMP : timestamp
    """

    class CloudVolume(djp.Part):
        enable_hashing = True
        hash_name = 'em_seg'
        hashed_attrs = 'em_seg_name'
        definition = """
        # segmentations hosted on CloudVolume 
        -> master
        ---
        em_seg_name            : varchar(48)      # name of em segmentation
        res_x                  : float            # resolution x, nanometers/ voxel
        res_y                  : float            # resolution y, nanometers/ voxel
        res_z                  : float            # resolution z, nanometers/ voxel
        ctr_pt_x               : float            # center of volume x, voxels
        ctr_pt_y               : float            # center of volume y, voxels
        ctr_pt_z               : float            # center of volume z, voxels
        min_pt_x               : float            # min point of volume bounds x, voxels
        min_pt_y               : float            # min point of volume bounds y, voxels
        min_pt_z               : float            # min point of volume bounds z, voxels
        max_pt_x               : float            # max point of volume bounds x, voxels
        max_pt_y               : float            # max point of volume bounds y, voxels
        max_pt_z               : float            # max point of volume bounds z, voxels
        voxel_offset_x         : float            # voxel offset x
        voxel_offset_y         : float            # voxel offset y
        voxel_offset_z         : float            # voxel offset z
        segment_id=NULL        : bigint unsigned  # id of segment (if applicable)
        cv_path                : varchar(450)     # cloudvolume path
        description            : varchar(450)     # description of alignment
        """


@schema
class EMSegAdjusted(djp.Lookup):
    hash_name = 'em_seg_adjusted'
    hash_part_table_names = True
    definition = """
    # Adjusted versions of EM segmentations
    -> EMSeg
    em_seg_adjusted             :    varchar(8)       # hash of adjusted em segmentation volume
    ---
    ts_inserted=CURRENT_TIMESTAMP : timestamp
    """
    
    class CloudVolume(djp.Part):
        enable_hashing = True
        hash_name = 'em_seg_adjusted'
        hashed_attrs = 'mip'
        definition = """
        # adjusted EM versions from cloudvolume
        -> master
        ---
        -> m65crg.AdjustmentSet
        mip=NULL               : tinyint          # mip
        res_x                  : float            # resolution x, nanometers/ voxel
        res_y                  : float            # resolution y, nanometers/ voxel
        res_z                  : float            # resolution z, nanometers/ voxel
        ctr_pt_x               : float            # center of volume x, voxels
        ctr_pt_y               : float            # center of volume y, voxels
        ctr_pt_z               : float            # center of volume z, voxels
        min_pt_x               : float            # min point of volume bounds x, voxels
        min_pt_y               : float            # min point of volume bounds y, voxels
        min_pt_z               : float            # min point of volume bounds z, voxels
        max_pt_x               : float            # max point of volume bounds x, voxels
        max_pt_y               : float            # max point of volume bounds y, voxels
        max_pt_z               : float            # max point of volume bounds z, voxels
        voxel_offset_x         : float            # voxel offset x
        voxel_offset_y         : float            # voxel offset y
        voxel_offset_z         : float            # voxel offset z
        """

schema.spawn_missing_classes()
schema.connection.dependencies.load()
