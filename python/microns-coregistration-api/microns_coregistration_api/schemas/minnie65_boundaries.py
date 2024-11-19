from pathlib import Path
import json
import numpy as np
import pickle
import pandas as pd
from itertools import groupby
import datajoint as dj
import datajoint_plus as djp

from microns_materialization_api.schemas import minnie65_materialization as m65mat
from .. import methods as crgm
import microns_utils.datajoint_utils as dju
from microns_utils.misc_utils import classproperty


from ..config import minnie65_boundaries_config as config
config.register_externals()
config.register_adapters(context=locals())

schema = djp.schema(config.schema_name, create_schema=True)

# schema.spawn_missing_classes()

@schema
class Tag(dju.VersionLookup):
    package = 'microns-coregistration-api'
    attr_name = 'tag'

@schema
class AutoProofreadNeuron(djp.Lookup):
    hash_name = 'auto_proofread_neuron'
    definition = f"""
    {hash_name} : varchar(8) # auto proofread neuron data hash
    """
    
    class Data(djp.Part):
        enable_hashing = True
        hash_name = 'auto_proofread_neuron'
        hashed_attrs = 'filename'
        definition = """
        -> master
        ---
        filename : varchar(1000) # name of file
        description : varchar(1000) 
        file : <layer_boundary_pandas_pickle_files>
        ts_inserted=CURRENT_TIMESTAMP : timestamp
        """ 

@schema
class LayerPrediction(djp.Lookup):
    hash_name = 'layer_prediction'
    definition = f"""
    {hash_name} : varchar(8) # layer prediction hash
    """
    
    class Data(djp.Part):
        enable_hashing = True
        hash_name = 'layer_prediction'
        hashed_attrs = 'filename'
        definition = """
        -> master
        ---
        filename : varchar(1000) # name of file
        description : varchar(1000) 
        file : <layer_boundary_pandas_pickle_files>
        ts_inserted=CURRENT_TIMESTAMP : timestamp
        """


@schema
class CoordinateAdjustMethod(djp.Lookup):
    hash_name = 'coordinate_adjust_method'
    definition = f"""
    {hash_name} : varchar(8) # method for adjusting coordinates
    """
    
    class M65Adjust(djp.Part):
        enable_hashing = True
        hash_name = 'coordinate_adjust_method'
        hashed_attrs = Tag.attr_name, 'm65_adjust_kws'
        definition = """
        -> master
        -> Tag
        m65_adjust_kws : varchar(1000)  # load with json.loads
        """
        @classmethod
        def update_method(cls, m65_adjust_kws):
            cls.insert1({
                Tag.attr_name: Tag.version,
                'm65_adjust_kws': json.dumps(m65_adjust_kws)
            }, insert_to_master=True, skip_duplicates=True)
        
        def run(self, coords, **kwargs):
            params = (self & kwargs).fetch1()
            assert params.get(Tag.attr_name) == Tag.version, 'Package version mismatch. Update method.'
            
            m65_adjust_kws = json.loads(params.get('m65_adjust_kws'))

            tform = crgm.M65Adjust(**m65_adjust_kws)

            return tform.adjust(coords)


@schema
class LayerBoundaryModelMethod(djp.Lookup):
    hash_name = 'layer_boundary_model_method'
    definition = f"""
    {hash_name} : varchar(8) # layer boundary model method hash
    """
    
    class FitFromPrediction(djp.Part):
        enable_hashing=True
        hash_name = 'layer_boundary_model_method'
        hashed_attrs = (
            Tag.attr_name, 
            AutoProofreadNeuron.hash_name, 
            CoordinateAdjustMethod.hash_name, 
            LayerPrediction.hash_name, 
            'fit_layer_boundaries_kws'
        )
        definition = """
        -> master
        ---
        -> Tag
        -> AutoProofreadNeuron
        -> CoordinateAdjustMethod
        -> LayerPrediction
        fit_layer_boundaries_kws : varchar(1000)  # load with json.loads
        ts_inserted=CURRENT_TIMESTAMP : timestamp
        """
        
        @classmethod
        def update_method(cls, auto_proofread_neuron, coordinate_adjust_method, layer_prediction, fit_layer_boundaries_kws:dict):
            cls.insert1({
                Tag.attr_name: Tag.version,
                'auto_proofread_neuron': auto_proofread_neuron,
                'coordinate_adjust_method': coordinate_adjust_method,
                'layer_prediction': layer_prediction,
                'fit_layer_boundaries_kws': json.dumps(fit_layer_boundaries_kws)
            }, insert_to_master=True, skip_duplicates=True)
        
        def run(self, **kwargs):
            params = (self & kwargs).fetch1()
            assert params.get(Tag.attr_name) == Tag.version, 'Package version mismatch. Update method.'
            
            fit_layer_boundaries_kws = json.loads(params.get('fit_layer_boundaries_kws'))
            
            apn_df = AutoProofreadNeuron.r1pwh(params.get('auto_proofread_neuron')).fetch1('file')
            apn_df = apn_df.rename(
                columns={
                    'centroid_x_nm': 'x', 
                    'centroid_y_nm': 'y', 
                    'centroid_z_nm': 'z'
                })[['segment_id', 'split_index', 'x', 'y', 'z', 'nucleus_id']]

            pred_df = LayerPrediction.r1pwh(params.get('layer_prediction')).fetch1('file')

            pred_df.loc[:, 'segment_id'] = pred_df.segment_id.astype(np.int64)
            pred_df.loc[:, 'split_index'] = pred_df.split_index.astype(np.int64)

            df = pred_df.merge(apn_df)
            
            ### Rotate coordinates
            soma_xyz = df[['x', 'y' ,'z']].values
            soma_xyz_adj = CoordinateAdjustMethod.r1pwh(params.get('coordinate_adjust_method')).run(soma_xyz)

            df['nx'] = soma_xyz_adj[:, 0]
            df['ny'] = soma_xyz_adj[:, 1]
            df['nz'] = soma_xyz_adj[:, 2]
            
            return crgm.fit_layer_boundaries(df, ('nx', 'ny', 'nz'), **fit_layer_boundaries_kws)


@schema
class LayerBoundaryModel(djp.Lookup):
    hash_name = 'layer_boundary_model'
    definition = f"""
    {hash_name} : varchar(8) # layer boundary model hash
    """
    
    class FitFromPrediction(djp.Part, dj.Computed):
        enable_hashing = True
        hash_name = 'layer_boundary_model'
        hashed_attrs = LayerBoundaryModelMethod.hash_name,  
        definition = """
        -> master
        -> LayerBoundaryModelMethod
        ---
        model_name : varchar(150) # name of model
        model : <layer_boundary_pickle_files>
        grid : <layer_boundary_numpy_files>
        """
        
        @classproperty
        def key_source(cls):
            return LayerBoundaryModelMethod & LayerBoundaryModelMethod.FitFromPrediction()
        
        def make(self, key):
            key_hash = self.hash1(key)
            models, grids = LayerBoundaryModelMethod.r1p(key).run()
            for (model_name, model), (grid_name, grid) in zip(models.items(), grids.items()):
                # make file path
                model_fn = f'LayerBoundaryModel_{key_hash}_model.pkl'
                grid_fn = f'LayerBoundaryModel_{key_hash}_grid.npy'
                
                model_dir_to_save = config.externals.get('layer_boundary_pickle_files').get('location')
                grid_dir_to_save = config.externals.get('layer_boundary_numpy_files').get('location')
                
                model_path_to_save = Path(model_dir_to_save) / model_fn
                grid_path_to_save = Path(grid_dir_to_save) / grid_fn
                
                # save pkl
                with open(model_path_to_save, 'wb') as f:
                    pickle.dump(model, f)
                
                # save npy
                np.save(grid_path_to_save, grid)
                
                # insert
                key['model_name'] = model_name
                key['model'] = model_path_to_save
                key['grid'] = grid_path_to_save
                self.insert1(key, insert_to_master=True)


@schema
class LayerBoundaryModelSet(djp.Lookup):
    hash_name = 'layer_boundary_model_set'
    definition = f"""
    {hash_name} : varchar(8) # hash of layer boundary model set
    """
    
    class Member(djp.Part):
        enable_hashing = True
        hash_name = 'layer_boundary_model_set'
        hashed_attrs = 'idx', LayerBoundaryModel.hash_name
        hash_group = True
        definition = """
        -> master
        idx : int # model order
        -> LayerBoundaryModel
        """


@schema
class LayerAssignmentMethod(djp.Lookup):
    hash_name = 'layer_assignment_method'
    definition = f"""
    {hash_name} : varchar(8)  # hash of layer assignment method
    """
    
    class LayerBoundaryModel(djp.Part):
        enable_hashing = True
        hash_name = 'layer_assignment_method'
        hashed_attrs = (
            Tag.attr_name, 
            LayerBoundaryModelSet.hash_name, 
            'assign_layer_kws'
        )
        definition = """
        -> master
        -> Tag
        -> LayerBoundaryModelSet
        assign_layer_kws : varchar(1000)  # load with json.loads
        """
        
        @classmethod
        def update_method(cls, layer_boundary_model_set, assign_layer_kws:dict):
            cls.insert1({
                Tag.attr_name: Tag.version,
                'layer_boundary_model_set': layer_boundary_model_set,
                'assign_layer_kws': json.dumps(assign_layer_kws)
            }, insert_to_master=True, skip_duplicates=True)

        def run(self, coords, **kwargs):
            """
            Args :
                coords
                    nX3 must be in nanometers
            """
            params = (self & kwargs).fetch1()
            assert params.get(Tag.attr_name) == Tag.version, 'Package version mismatch. Update method.'
            
            model_hashes = LayerBoundaryModelSet.r1pwh(params.get('layer_boundary_model_set')).fetch('layer_boundary_model', order_by='idx')
            
            method_hashes = []
            models = []
            for mh in model_hashes:
                method_hash, model = LayerBoundaryModel.r1pwh(mh).fetch1('layer_boundary_model_method', 'model')
                method_hashes.append(method_hash)
                models.append(model)
            
            coord_adjust_hashes = []
            for mth in method_hashes:
                coord_adjust_hashes.append(LayerBoundaryModelMethod.r1pwh(mth).fetch1('coordinate_adjust_method'))
            
            g = groupby(coord_adjust_hashes)
            assert next(g, True) and not next(g, False), 'all coordinate adjustment hashes must be the same'
            
            coord_adjust_hash = coord_adjust_hashes[0]
            
            adjusted_coords = CoordinateAdjustMethod.r1pwh(coord_adjust_hash).run(coords)
            
            assign_layer_kws = json.loads(params.get('assign_layer_kws'))
            
            return crgm.assign_layer(
                coords=adjusted_coords, 
                models=models,
                **assign_layer_kws
            )


@schema
class LayerAssignmentSource(djp.Lookup):
    hash_name = 'layer_assignment_source'
    definition = f"""
    {hash_name} : varchar(8) # hash of source data for layer assignment
    """
    
    class Nucleus(djp.Part):
        enable_hashing = True
        hash_name = 'layer_assignment_source'
        hashed_attrs = 'ver', 'nucleus_id', 'segment_id'
        hash_group = True
        definition = """
        -> master
        -> m65mat.Nucleus.Info
        ---
        nucleus_x : int # nucleus_x in nanometers
        nucleus_y : int # nucleus_y in nanometers
        nucleus_z : int # nucleus_z in nanometers
        """
        
        @classmethod
        def add_ver(cls, ver):
            nuc_rel = (m65mat.Nucleus.Info & {'ver': ver}).proj(nucleus_x='nucleus_x*4', nucleus_y='nucleus_y*4', nucleus_z='nucleus_z*40')
            cls.insert(nuc_rel, insert_to_master=True)


@schema
class LayerAssignment(djp.Lookup):
    hash_name = 'layer_assignment'
    definition = f"""
    {hash_name} : varchar(8) # 
    ---
    ts_inserted=CURRENT_TIMESTAMP : timestamp
    """

    class Nucleus(djp.Part, dj.Computed):
        enable_hashing = True
        hash_name = 'layer_assignment'
        hashed_attrs = LayerAssignmentSource.hash_name, LayerAssignmentMethod.hash_name
        definition = """
            -> master
            -> LayerAssignmentSource.Nucleus
            -> LayerAssignmentMethod
            ---
            layer : varchar(128) # layer assignment
            """
        
        @classproperty
        def key_source(cls):
            return (LayerAssignmentSource & LayerAssignmentSource.Nucleus) * (LayerAssignmentMethod & LayerAssignmentMethod.LayerBoundaryModel)
        
        def make(self, key):
            source_df = pd.DataFrame(LayerAssignmentSource.r1p(key).fetch())
            source_df['layer_assignment_method'] = key.get('layer_assignment_method')
            source_df['layer'] = LayerAssignmentMethod.r1p(key).run(source_df[['nucleus_x', 'nucleus_y', 'nucleus_z']])
            source_df[self.hash_name] = self.hash1(key)
            self.insert(source_df, insert_to_master=True, skip_hashing=True, ignore_extra_fields=True)
        
        @classmethod
        def get_info(cls, layer_assignment_hash):
            # get layer assignment info
            layer_assignment_rel = (cls & {'layer_assignment': layer_assignment_hash})
            layer_assignment_source_hash = (dj.U('layer_assignment_source') & layer_assignment_rel).fetch1('layer_assignment_source')
            layer_assignment_method_hash = (dj.U('layer_assignment_method') & layer_assignment_rel).fetch1('layer_assignment_method')
            layer_boundary_model_set_hash = LayerAssignmentMethod.r1pwh(layer_assignment_method_hash).fetch1('layer_boundary_model_set')
            layer_boundary_model_set_rel = LayerBoundaryModelSet.r1pwh(layer_boundary_model_set_hash)
            
            # get coordinate adjustment method
            coordinate_adjust_method_hashes = []
            for lbm in layer_boundary_model_set_rel:
                layer_boundary_model_hash = lbm.get('layer_boundary_model')
                layer_boundary_model_method_hash = LayerBoundaryModel.r1pwh(layer_boundary_model_hash).fetch1('layer_boundary_model_method')
                coordinate_adjust_method_hashes.append(LayerBoundaryModelMethod.r1pwh(layer_boundary_model_method_hash).fetch1('coordinate_adjust_method'))
            g = groupby(coordinate_adjust_method_hashes)
            assert next(g, True) and not next(g, False), 'all coordinate adjustment hashes must be the same'
            coordinate_adjust_hash = coordinate_adjust_method_hashes[0]
            
            # make dataframe
            source_df = pd.DataFrame(LayerAssignmentSource.r1pwh(layer_assignment_source_hash).fetch())
            source_df['coordinate_adjust_method'] = coordinate_adjust_hash
            source_df['layer_boundary_model_set'] = layer_boundary_model_set_hash
            source_df['nucleus_x_adj_nm'], source_df['nucleusy_adj_nm'], source_df['nucleus_z_adj_nm'] = CoordinateAdjustMethod.r1pwh(coordinate_adjust_hash).run(source_df[['nucleus_x', 'nucleus_y', 'nucleus_z']]).T
            layer_assignment_df = pd.DataFrame(layer_assignment_rel.fetch())
            return layer_assignment_df.merge(source_df)


schema.spawn_missing_classes()