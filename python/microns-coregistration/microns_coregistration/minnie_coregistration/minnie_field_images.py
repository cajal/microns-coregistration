"""
DataJoint tables for 2P field images
"""
import json
from pathlib import Path
import numpy as np

import datajoint as dj
import datajoint_plus as djp

from microns_coregistration_api.methods import registration as reg
import microns_utils.transform_utils as tu
from microns_utils.misc_utils import classproperty

from microns_coregistration_api.schemas import minnie_field_images as mfi
from microns_nda_api.schemas import minnie_nda as nda

schema = mfi.schema
config = mfi.config

logger = djp.getLogger(__name__)

class Tag(mfi.Tag):
    pass


class FieldImageMethod(mfi.FieldImageMethod):

    @classmethod
    def run(cls, key):
        return cls.r1p(key).run(**key)
        
    class EnhanceResizeNormalize(mfi.FieldImageMethod.EnhanceResizeNormalize):
        @classmethod
        def update_method(cls, lcn_kws, sharpen_2pimage_kws, resize_kws):
            cls.insert1(
                {
                    'lcn_kws': json.dumps(lcn_kws),
                    'sharpen_2pimage_kws': json.dumps(sharpen_2pimage_kws),
                    'resize_kws': json.dumps(resize_kws),
                    mfi.Tag.attr_name: mfi.Tag.version
                }, insert_to_master=True
            )

        def run(self, original, um_height, um_width, **kwargs):
            params = (self & kwargs).fetch1()

            lcn_kws = json.loads(params.get('lcn_kws'))
            sharpen_2pimage_kws = json.loads(params.get('sharpen_2pimage_kws'))
            resize_kws = json.loads(params.get('resize_kws'))

            im = tu.lcn(original, **lcn_kws)
            im = tu.sharpen_2pimage(im, **sharpen_2pimage_kws)
            im = reg.resize(im, [um_height, um_width], **resize_kws)
            im = tu.normalize_image(im, newrange=[0, 1], astype=float, clip_bounds=[0, im.max()])

            return {
                'image': im
            }
        

class FieldImage(mfi.FieldImage):

    class Store(mfi.FieldImage.Store):
        pass

    class Maker(mfi.FieldImage.Maker):
        @property
        def key_source(self):
            return FieldImageMethod * (nda.SummaryImages() & nda.Field.Meso)

        @classproperty
        def basedir(self):
            return Path(mfi.config.externals['field_image_files']['location'])
        
        @classmethod
        def make_fp(cls, field_im_id, animal_id, scan_session, scan_idx, field, scan_channel, suffix='.npy', **kwargs):
            fn = f'{field_im_id}_{animal_id}_{scan_session}_{scan_idx}_{field}_{scan_channel}'
            return cls.basedir.joinpath(fn).with_suffix(suffix)
        
        def make(self, key):
            um_height, um_width = nda.Field.r1p(key).fetch1('um_height', 'um_width')
            avg_im = (nda.SummaryImages.Average & key).fetch1('average_image')
            corr_im = (nda.SummaryImages.Correlation & key).fetch1('correlation_image')
            image = FieldImageMethod.run({**key, 'original': avg_im * corr_im, 'um_height': um_height, 'um_width': um_width})['image']
            key['field_im_id'] = self.hash1(key)
            key['image'] = self.make_fp(**key)
            np.save(key['image'], image)
            self.master.Store.insert1(key, insert_to_master=True, ignore_extra_fields=True)
            self.insert1(key, ignore_extra_fields=True, skip_hashing=True)
            

class FieldGridMethod(mfi.FieldGridMethod):

    @classmethod
    def run(cls, key):
        return cls.r1p(key).run(**key)
    
    class Affine(mfi.FieldGridMethod.Affine):
        @classmethod
        def update_method(cls, desired_res):
            cls.insert1(
                {
                    'desired_res': desired_res,
                    Tag.attr_name: Tag.version
                }, insert_to_master=True
            )

        def run(self, um_sizes, reg_params_dict, **kwargs):
            params = (self & kwargs).fetch1()
            return {
                'grid': reg.get_affine_grid(um_sizes=um_sizes, reg_params_dict=reg_params_dict, desired_res=params['desired_res'])
            }

class FieldGrid(mfi.FieldGrid):
    
    class Store(mfi.FieldGrid.Store):
        pass

    class Maker(mfi.FieldGrid.Maker):
        @property
        def key_source(self):
            return FieldGridMethod * nda.Registration.Affine

        @classproperty
        def basedir(self):
            return Path(mfi.config.externals['field_grid_files']['location'])
        
        @classmethod
        def make_fp(cls, field_grid_id, field_grid_method, stack_session, stack_idx, volume_id, scan_session, scan_idx, field, registration_method, suffix='.npz', **kwargs):
            fn = f'{field_grid_id}_{field_grid_method}_{stack_session}_{stack_idx}_{volume_id}_{scan_session}_{scan_idx}_{field}_{registration_method}'
            return cls.basedir.joinpath(fn).with_suffix(suffix)
        
        def make(self, key):
            um_height, um_width = nda.Field.r1p(key).fetch1('um_height', 'um_width')
            reg_param_list =  'a11', 'a21', 'a31', 'a12', 'a22', 'a32', 'reg_x', 'reg_y', 'reg_z'
            reg_params = (nda.Registration.Affine & key).fetch(*reg_param_list, as_dict=True)[0]
            motor = FieldGridMethod.run({**key, 'um_sizes': [um_height, um_width], 'reg_params_dict': reg_params})['grid']
            cx, cy, cz, stack_width_um, stack_height_um, stack_depth_um = (nda.Stack.Corrected & key).fetch1('x', 'y', 'z', 'um_width', 'um_height','um_depth')
            center_xyz = np.array([cx, cy, cz])
            bounds_xyz = np.array([stack_width_um, stack_height_um, stack_depth_um])
            centered = motor - center_xyz
            stack = centered + bounds_xyz//2
            key['field_grid_id'] = self.hash1(key)
            key['grid'] = self.make_fp(**key)
            np.savez_compressed(key['grid'], motor=motor, centered=centered, stack=stack)
            self.master.Store.insert1(key, insert_to_master=True, ignore_extra_fields=True)
            self.insert1(key, ignore_extra_fields=True, skip_hashing=True)
