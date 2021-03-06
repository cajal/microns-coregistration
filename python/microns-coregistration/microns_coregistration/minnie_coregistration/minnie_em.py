"""
DataJoint tables for processing of minnie EM data.
"""

from datetime import datetime
import datajoint_plus as djp
from datajoint_plus.utils import classproperty
import numpy as np
import traceback
from pathlib import Path

from microns_coregistration_api.schemas import minnie_em as em, minnie65_coregistration as m65crg

import microns_utils.ap_utils as apu
import microns_utils.filepath_utils as fpu

schema = em.schema
config = em.config

logger = djp.getLogger(__name__)

# DATAJOINT TABLES

class EM(em.EM):

    @classmethod
    def fill(cls):
        for em_name in ['minnie65', 'minnie35']:
            if em_name == 'minnie65':
                for alignment in [1, 2]:
                    if alignment == 1:
                        res_x, res_y, res_z = [4, 4, 40]
                        data = {
                            'em_name': em_name,
                            'alignment': alignment,
                            'res_x': 4,
                            'res_y': 4,
                            'res_z': 40,
                            'ctr_pt_x': 1143808.0 / res_x,
                            'ctr_pt_y': 939008.0 / res_y,
                            'ctr_pt_z': 855300.0 / res_z,
                            'min_pt_x': -6144 / res_x,
                            'min_pt_y': -6144 / res_y,
                            'min_pt_z': 595360 / res_z,
                            'max_pt_x': 2293760 / res_x,
                            'max_pt_y': 1884160 / res_y ,
                            'max_pt_z': 1115240 / res_z,
                            'min_pt_x_anat': 'medial',
                            'min_pt_y_anat': 'superficial',
                            'min_pt_z_anat': 'posterior',
                            'max_pt_x_anat': 'lateral',
                            'max_pt_y_anat': 'deep',
                            'max_pt_z_anat': 'anterior',
                            'cv_path' : 'precomputed://https://s3-hpcrc.rc.princeton.edu/minnie65/aligned_image',
                            'description': "This is Seung lab's first alignment of the 1mm x 1mm x 40um IARPA dataset from mouse V1 in MICrONS phase 2."
                        }
                        cls.insert1(data, skip_duplicates=True)

                    if alignment == 2:
                        cv_path = 'precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/em'
                        description = "This is the second alignment of the IARPA 'minnie65' dataset, completed in the spring of 2020 that used the seamless approach."
                        stats = apu.get_stats_from_cv_path(cv_path, mip=0)
                        res = np.array([4, 4, 40])
                        ctr_pt_x, ctr_pt_y, ctr_py_z = (stats['ctr_pt'] * stats['res']) / res
                        min_pt_x, min_pt_y, min_pt_z = (stats['min_pt'] * stats['res']) / res
                        max_pt_x, max_pt_y, max_pt_z = (stats['max_pt'] * stats['res']) / res
                        res_x, res_y, res_z = res
                        data = {
                            'em_name': em_name,
                            'alignment': alignment,
                            'res_x': res_x,
                            'res_y': res_y,
                            'res_z': res_z,
                            'ctr_pt_x': ctr_pt_x,
                            'ctr_pt_y': ctr_pt_y,
                            'ctr_pt_z': ctr_py_z,
                            'min_pt_x': min_pt_x,
                            'min_pt_y': min_pt_y,
                            'min_pt_z': min_pt_z,
                            'max_pt_x': max_pt_x,
                            'max_pt_y': max_pt_y,
                            'max_pt_z': max_pt_z,
                            'min_pt_x_anat': 'medial',
                            'min_pt_y_anat': 'superficial',
                            'min_pt_z_anat': 'posterior',
                            'max_pt_x_anat': 'lateral',
                            'max_pt_y_anat': 'deep',
                            'max_pt_z_anat': 'anterior',
                            'cv_path': cv_path,
                            'description': description
                        }
                        cls.insert1(data, skip_duplicates=True)

            if em_name == 'minnie35':
                cv_path = 'precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie35/em'
                description = 'This is the IARPA "minnie35" dataset, completed in July of 2021.'
                stats = apu.get_stats_from_cv_path(cv_path, mip=0)
                res = np.array([4, 4, 40])
                ctr_pt_x, ctr_pt_y, ctr_py_z = (stats['ctr_pt'] * stats['res']) / res
                min_pt_x, min_pt_y, min_pt_z = (stats['min_pt'] * stats['res']) / res
                max_pt_x, max_pt_y, max_pt_z = (stats['max_pt'] * stats['res']) / res
                res_x, res_y, res_z = res

                data = {
                    'em_name': em_name,
                    'alignment': 1,
                    'res_x': res_x,
                    'res_y': res_y,
                    'res_z': res_z,
                    'ctr_pt_x': ctr_pt_x,
                    'ctr_pt_y': ctr_pt_y,
                    'ctr_pt_z': ctr_py_z,
                    'min_pt_x': min_pt_x,
                    'min_pt_y': min_pt_y,
                    'min_pt_z': min_pt_z,
                    'max_pt_x': max_pt_x,
                    'max_pt_y': max_pt_y,
                    'max_pt_z': max_pt_z,
                    'min_pt_x_anat': 'medial',
                    'min_pt_y_anat': 'superficial',
                    'min_pt_z_anat': 'posterior',
                    'max_pt_x_anat': 'lateral',
                    'max_pt_y_anat': 'deep',
                    'max_pt_z_anat': 'anterior',
                    'cv_path': cv_path,
                    'description': description
                }
                cls.insert1(data, skip_duplicates=True)


class EMAdjusted(em.EMAdjusted):
    class CloudVolume(em.EMAdjusted.CloudVolume):

        @classmethod
        def fill(cls):
            for key in cls.key_source:
                try:
                    for stats in apu.get_stats_from_cv_path(key['cv_path']):
                        axes = ['x', 'y', 'z']
                        names = ['res', 'min_pt', 'max_pt', 'ctr_pt', 'voxel_offset']
                        to_insert = [{name + '_' + c: q for c, q in zip(axes, quantity)} for name, quantity in zip(names, [stats[n] for n in names])]
                        cls.insert1({
                            **key, 
                            **{'adjustment_set': m65crg.AdjustmentSet.hash1([{'adjustment': 'resample'}])},
                            **{'mip': stats['mip']},
                            **{k: v for t in to_insert for k, v in t.items()},
                        }, ignore_extra_fields=True, skip_duplicates=True, insert_to_master=True)
                except:
                    print(f'Could not insert from cloudvolume for key: {key}')
    
    
    class Custom(em.EMAdjusted.Custom):

        @classmethod
        def fill(cls):
            # PHASE 2
            res = np.array([1000, 1000, 1000]) # resolution nanometers
            span = (1616, 1180, 520) * np.array([1000]) # dimensions of cropped EM in nm
            center = (538560.0, 901280.41, 945454.55) # new computed center after cropping in nm
            res_x, res_y, res_z = res
            min_pt_x, min_pt_y, min_pt_z = (center - span / 2) / res
            max_pt_x, max_pt_y, max_pt_z = (center + span / 2) / res
            ctr_pt_x, ctr_pt_y, ctr_py_z = center / res
            description = 'resized to 1000nm/ pixel and cropped to remove excess space. Center adjusted accordingly'
            cls.insert1({
                'em_name': 'minnie65',
                'alignment': 1,
                'adjustment_set': m65crg.AdjustmentSet.hash1([{'adjustment': 'resample'}, {'adjustment': 'crop'}], unique=True),
                'res_x': res_x,
                'res_y': res_y,
                'res_z': res_z,
                'min_pt_x': min_pt_x,
                'min_pt_y': min_pt_y,
                'min_pt_z': min_pt_z,
                'max_pt_x': max_pt_x,
                'max_pt_y': max_pt_y,
                'max_pt_z': max_pt_z,
                'ctr_pt_x': ctr_pt_x,
                'ctr_pt_y': ctr_pt_y,
                'ctr_pt_z': ctr_py_z,
                'description': description
            }, ignore_extra_fields=True, skip_duplicates=True, insert_to_master=True)
            
            # PHASE 3, crop EM to match nuclear segmentation and resize to 1000nm3
            # data source keys
            em_key = {'em_name': 'minnie65', 'alignment': 2, 'mip': 6}
            seg_key = EMSegAdjusted.CloudVolume() & {'mip':3} & (EMSeg.CloudVolume & {'em_seg_name': 'minnie3_v1_nuc_v0'} ).proj()

            # source info
            em_info = EMAdjusted.CloudVolume & em_key
            seg_info = EMSegAdjusted.CloudVolume & seg_key.proj()
            
            em_min_pt = np.stack(em_info.fetch1('min_pt_x', 'min_pt_y', 'min_pt_z'), -1)
            em_max_pt = np.stack(em_info.fetch1('max_pt_x', 'max_pt_y', 'max_pt_z'), -1)
            em_res = np.stack(em_info.fetch1('res_x', 'res_y', 'res_z'), -1)
            em_size = np.stack(em_info.proj(**{f'span_{c}': f'max_pt_{c} - min_pt_{c}' for c in ['x', 'y', 'z']}).fetch1('span_x', 'span_y', 'span_z'), -1)
            em_voxel_offset = np.stack(em_info.fetch1('voxel_offset_x', 'voxel_offset_y', 'voxel_offset_z'), -1)
            seg_size = np.stack(seg_info.proj(**{f'span_{c}': f'max_pt_{c} - min_pt_{c}' for c in ['x', 'y', 'z']}).fetch1('span_x', 'span_y', 'span_z'), -1)
            seg_voxel_offset = np.stack(seg_info.fetch1('voxel_offset_x', 'voxel_offset_y', 'voxel_offset_z'), -1)
            
            # crop EM
            top = seg_voxel_offset - em_voxel_offset # distance to crop from EM off of the top
            bottom = em_size - top - seg_size # distance to crop from EM off of the bottom

            ng_res = np.array([4,4,40])
            em_min_pt = (em_min_pt * em_res / ng_res) # mip 0 min bound
            em_max_pt = (em_max_pt * em_res / ng_res) # mip 0 max bound
            em_center = (((em_max_pt - em_min_pt) / 2) + em_min_pt) * ng_res # em center in nanos
            
            # resize to 1000nm3 and get new ctr_pt, min_pt and max_pt
            new_res = np.array([1000, 1000, 1000])
            new_em_center = (em_center + (top * em_res) / 2 - (bottom * em_res) / 2) / new_res # new em center of cropped em 
            new_em_min_pt = (em_min_pt * ng_res + top * em_res) / new_res
            new_em_max_pt = (em_max_pt * ng_res - bottom * em_res) / new_res
            
            # insert
            data = [{f'{name}_{c}': q for q, c in zip(quantity, ['x', 'y', 'z'])} for quantity, name in zip([new_res, new_em_center, new_em_min_pt, new_em_max_pt], ['res', 'ctr_pt', 'min_pt', 'max_pt'])]
            description = {'description': 'resized to 1000nm/ pixel and cropped to nuclear seg boundaries. Center adjusted accordingly'}
            cls.insert1({
                **{'em_name': 'minnie65', 'alignment': 2}, 
                **{'adjustment_set': m65crg.AdjustmentSet.hash1([{'adjustment': 'resample'}, {'adjustment': 'crop'}], unique=True)}, 
                **{k: v for d in data for k, v in d.items()}, 
                **description
            }, ignore_extra_fields=True, skip_duplicates=True, insert_to_master=True)


class EMSeg(em.EMSeg):

    class CloudVolume(em.EMSeg.CloudVolume):      
        @classmethod
        def cloudvolume_segmentation_keys(cls):
            return [
                # phase 2
                dict(
                    em_name = 'minnie65',
                    alignment = 1, 
                    em_seg_name = 'seg_minnie65_0',
                    cv_path = 'precomputed://https://storage.googleapis.com/microns-seunglab/minnie65/seg_minnie65_0',
                    description = 'This contains the flat segmentation of the phase2 alignment and segmentation of Minnie65.'
                ),

                # phase 3 dynamic seg
                dict(
                    em_name = 'minnie65',
                    alignment = 2,
                    em_seg_name = 'minnie3_v1_seg_v1',
                    cv_path = 'graphene://https://minnie.microns-daf.com/segmentation/table/minnie3_v1',
                    description = 'This is the first version of Minnie that has proofreading enabled. Was first enabled on June 24, 2020.'
                ),

                # phase 3 nuc seg
                dict(
                    em_name = 'minnie65',
                    alignment = 2,
                    em_seg_name = 'minnie3_v1_nuc_v0',
                    cv_path = 'precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/nuclei',
                    description = 'Nuclear segmentation from the Allen Institute v0.'
                ),

                # phase 3 public flat segmentation
                dict(
                    em_name = 'minnie65',
                    alignment = 2,
                    em_seg_name = 'minnie3_v1_public_flat_v117',
                    cv_path = 'graphene://https://minnie.microns-daf.com/segmentation/table/minnie65_public_v117',
                    description = 'minnie65 phase 3 flat segmentation (v117_public)'
                ),

                # phase 3 public flat segmentation
                dict(
                    em_name = 'minnie65',
                    alignment = 2,
                    em_seg_name = 'minnie3_v1_vess_v117',
                    segment_id = 864691136534887842,
                    cv_path = 'graphene://https://minnie.microns-daf.com/segmentation/table/minnie65_public_v117',
                    description = 'vessel segmentation from minnie65 phase 3 flat segmentation (v117_public)'
                ),

                # minnie35 dynamic seg
                dict(
                    em_name = 'minnie35',
                    alignment = 1,
                    em_seg_name = 'minnie35_phase3_v0',
                    cv_path = 'graphene://https://minnie.microns-daf.com/segmentation/table/minnie35_p3_v1',
                    description = 'Downloaded from annotation framework on 10/18/21, no description available.'
                )
            ]

        @classmethod
        def fill(cls):
            insert_kwargs = dict(ignore_extra_fields=True, skip_duplicates=True, insert_to_master=True)
            for key in cls.cloudvolume_segmentation_keys():
                try:
                    for stats in apu.get_stats_from_cv_path(key['cv_path']):
                        res = np.array([4, 4, 40])
                        axes = ['x', 'y', 'z']
                        names = ['min_pt', 'max_pt', 'ctr_pt', 'voxel_offset']
                        to_insert = [{name + '_' + c: q for c, q in zip(axes, quantity)} for name, quantity in zip(names, [(stats[n] * stats['res']) / res for n in names])]
                        key['res_x'], key['res_y'], key['res_z'] = res
                        cls.insert1({
                            **key, 
                            **{k: v for t in to_insert for k, v in t.items()},
                        }, **insert_kwargs)
                except:
                    traceback.print_exc()
                    print(f'Could not insert for key: {key["em_seg_name"]}')
            
            # phase 3 flat seg
            phase_3_flat_seg_key = (cls & {'em_seg_name': 'minnie3_v1_seg_v1'}).fetch(*[a for a in cls.heading.names if a not in cls.hash_name], as_dict=True)[0]
            phase_3_flat_seg_key['em_seg_name'] = 'minnie3_v1_seg_v0',
            phase_3_flat_seg_key['cv_path'] = 'graphene://https://minniev1.microns-daf.com/segmentation/table/minnie3_v0',
            phase_3_flat_seg_key['description'] = 'This is the initial dynamic segmentation of minnie65 phase 3, for which no proofreading is available.',
            cls.insert1(phase_3_flat_seg_key, **insert_kwargs)


class EMSegAdjusted(em.EMSegAdjusted):
    
    class CloudVolume(em.EMSegAdjusted.CloudVolume):

        @classmethod
        def fill(cls):
            for key in cls.key_source:
                try:
                    for stats in apu.get_stats_from_cv_path(key['cv_path']):
                        axes = ['x', 'y', 'z']
                        names = ['res', 'min_pt', 'max_pt', 'ctr_pt', 'voxel_offset']
                        to_insert = [{name + '_' + c: q for c, q in zip(axes, quantity)} for name, quantity in zip(names, [stats[n] for n in names])]
                        cls.insert1({
                            **key, 
                            **{'adjustment_set': m65crg.AdjustmentSet.hash1([{'adjustment': 'resample'}])},
                            **{'mip': stats['mip']},
                            **{k: v for t in to_insert for k, v in t.items()},
                        }, ignore_extra_fields=True, skip_duplicates=True, insert_to_master=True)
                except:
                    print(f'Could not insert from cloudvolume for key: {key}')      



class Stack(em.Stack):
    @classmethod
    def get_stack_from_cloudvolume(cls, source_rel):
        cv_path = source_rel.fetch1('cv_path')
        mip = source_rel.fetch1('mip')
        segment_id= source_rel.fetch1('segment_id') if 'segment_id' in source_rel.heading.attributes else None
        return apu.get_stack_from_cv_path(cv_path=cv_path, mip=mip, seg_ids=segment_id)

    class EMAdjusted(em.Stack.EMAdjusted):
        @classproperty
        def cloudvolume_source(cls):
            return EM.proj('cv_path') * EMAdjusted.CloudVolume

        @classmethod
        def fill_stack_from_cloudvolume(cls, key):
            destination = config.externals['minnie_stacks']['location']
            source_rel = cls.cloudvolume_source & key
            stack = cls.master.get_stack_from_cloudvolume(source_rel)

            # make file path
            key = source_rel.fetch1()
            key['ts_downloaded'] = str(datetime.now())
            filepath = Path(destination).joinpath(cls.hash1(key)).with_suffix('.npy')
            key['filepath'] = filepath
            key['data'] = filepath
            
            # save file
            np.save(filepath, stack)
            cls.insert1(key, insert_to_master=True, ignore_extra_fields=True)

        @classmethod
        def fill_from_path(cls, source_path, destination):
            pass
      
    class EMSeg(em.Stack.EMSeg):
        @classproperty
        def cloudvolume_source(cls):
            return EMSeg.proj('em_seg_name') * EMSeg.CloudVolume.proj('cv_path', 'segment_id') * EMSegAdjusted.CloudVolume
        
        @classmethod
        def fill_stack_from_cloudvolume(cls, key):
            destination = config.externals['minnie_stacks']['location']
            source_rel = cls.cloudvolume_source & key
            stack = cls.master.get_stack_from_cloudvolume(source_rel)

            # make file path
            key = source_rel.fetch1()
            key['ts_downloaded'] = str(datetime.now())
            filepath = Path(destination).joinpath(cls.hash1(key)).with_suffix('.npy')
            key['filepath'] = filepath
            key['data'] = filepath
            
            # save file
            np.save(filepath, stack)
            cls.insert1(key, insert_to_master=True, ignore_extra_fields=True)

        @classmethod
        def fill_stack_from_numpy_path(cls, key, numpy_path):
            pass
            
        # TODO add method for resizing and cropping EM stack using latest torch version
            # em_cropped = em[top[0]:-bottom[0], top[1]:-bottom[1] ,top[2]:-bottom[2]] # em cropped to align with segmentation size
#             um_sizes = em_cropped.shape * cv_em.resolution / 1000
#             em_resized = utils.resize(em_cropped, um_sizes, 1)