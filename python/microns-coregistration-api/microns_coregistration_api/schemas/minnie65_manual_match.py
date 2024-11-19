"""
DataJoint tables for manual match.
"""
import json
import time
from pathlib import Path
import numpy as np
from tqdm.notebook import tqdm
import pandas as pd
from scipy.spatial.distance import cdist
from datetime import datetime

import datajoint as dj
import datajoint_plus as djp

from ..methods import match
from ..schemas import minnie65_coregistration as m65cor
from microns_utils.misc_utils import FieldDict, classproperty, wrap, unwrap

from microns_nda_api.schemas import minnie_nda as nda
from microns_materialization_api.schemas import minnie65_materialization as m65mat

from microns_utils.ap_utils import set_CAVEclient
from microns_utils.misc_utils import classproperty
import microns_utils.datajoint_utils as dju

import os
cvt = os.getenv('CLOUDVOLUME_TOKEN')
assert cvt is not None, 'No cloudvolume token found'

# Current module
m65 = djp.create_djp_module(schema_name='microns_minnie65_02')

from ..config import minnie65_manual_match_config as config

config.register_externals()
config.register_adapters(context=locals())

schema = djp.schema(config.schema_name, create_schema=True)

@schema
class Tag(dju.VersionLookup):
    package = 'microns-coregistration-api'
    attr_name = 'tag'

@schema
class ImportMethod(djp.Lookup):
    hash_name = 'import_method_id'
    definition = """
    import_method_id           :  varchar(6)     # id of method 
    """
    @classmethod
    def run(cls, key):
        return cls.r1p(key).run(**key)
    
    class NucleusNeuronSVM(djp.Part):
        enable_hashing = True
        hash_name = 'import_method_id'
        hashed_attrs = 'ver', 'caveclient_kws', Tag.attr_name
        definition = """
        # method for importing caveclient table `nucleus_neuron_svm` with nucleus centroids and segment_ids from nucleus_detection_v0 or nucleus_alternative_points when applicable
        -> master
        ---
        ver                  : decimal(6,2)                 # materialization version
        caveclient_kws=NULL  : varchar(1000)                # use json.loads to recover dict
        -> Tag
        """
        
        @classmethod
        def update_method(cls, ver=None, caveclient_kws=None):
            caveclient_kws = {} if caveclient_kws is None else caveclient_kws
            caveclient_kws.setdefault('auth_token', cvt)
            client = set_CAVEclient(datastack='minnie65_phase3_v1', ver=ver, caveclient_kws=caveclient_kws)
            client = set_CAVEclient(datastack='minnie65_phase3_v1', ver=ver, caveclient_kws=caveclient_kws)
            try:
                caveclient_kws.pop('auth_token')
            except KeyError:
                pass
            entry = {
                'ver': client.materialize.version, 
                'caveclient_kws': json.dumps(caveclient_kws) if caveclient_kws is not None else None,
                Tag.attr_name : Tag.version
            }
            cls.insert(entry, insert_to_master=True)
            
        def run(self, **kwargs):
            params = (self & kwargs).fetch1()
            ver = params.get('ver')
            caveclient_kws = params.get('caveclient_kws')
            caveclient_kws = json.loads(caveclient_kws) if caveclient_kws is not None else None
            caveclient_kws['auth_token'] = cvt
            client = set_CAVEclient(
                datastack='minnie65_phase3_v1', 
                ver=ver, 
                caveclient_kws=caveclient_kws
            )
            nucleus_df = client.materialize.query_table('nucleus_detection_v0', split_positions=True)
            alternative_points_df = client.materialize.query_table('nucleus_alternative_points', split_positions=True)
            matching_inds = nucleus_df.merge(alternative_points_df, left_on='id', right_on='id_ref', how='left').query('pt_root_id_y>0').index
            sub_nucleus_df = nucleus_df.drop(matching_inds)
            alternative_points_mod_df = alternative_points_df[['valid', 'target_id', 'pt_position_x', 'pt_position_y', 'pt_position_z', 'pt_supervoxel_id', 'pt_root_id']].rename(columns={'target_id': 'nucleus_id'})
            nucleus_df = pd.concat([sub_nucleus_df.rename(columns={'id': 'nucleus_id'}), alternative_points_mod_df])
            neuron_df = client.materialize.query_table('nucleus_neuron_svm').rename(columns={'id': 'nucleus_id'})
            df = neuron_df.merge(nucleus_df, on='nucleus_id', suffixes=['_svm', '_det']).rename(columns={
                'pt_root_id_det': 'segment_id',
                'pt_position_x': 'nucleus_x',
                'pt_position_y': 'nucleus_y',
                'pt_position_z': 'nucleus_z'
            })
            df[self.hash_name] = params.get(self.hash_name)
            return df
    
    class MetaModelMtypes(djp.Part):
        enable_hashing = True
        hash_name = 'import_method_id'
        hashed_attrs = 'ver', 'table_name', 'caveclient_kws', Tag.attr_name
        definition = """
        # method for importing caveclient table `nucleus_neuron_svm` with nucleus centroids and segment_ids from nucleus_detection_v0 or nucleus_alternative_points when applicable
        -> master
        ---
        ver                  : decimal(6,2)                 # materialization version
        table_name           : varchar(1000)                # name of cave table
        caveclient_kws=NULL  : varchar(1000)                # use json.loads to recover dict
        -> Tag
        """
        
        @classmethod
        def update_method(cls, ver, table_name, caveclient_kws=None):
            caveclient_kws = {} if caveclient_kws is None else caveclient_kws
            caveclient_kws.setdefault('auth_token', cvt)
            client = set_CAVEclient(datastack='minnie65_phase3_v1', ver=ver, caveclient_kws=caveclient_kws)
            try:
                caveclient_kws.pop('auth_token')
            except KeyError:
                pass
            entry = {
                'ver': client.materialize.version, 
                'table_name': table_name,
                'caveclient_kws': json.dumps(caveclient_kws) if caveclient_kws is not None else None,
                Tag.attr_name : Tag.version
            }

            cls.insert(entry, insert_to_master=True)
            
        def run(self, **kwargs):
            params = (self & kwargs).fetch1()
            ver = params.get('ver')
            table_name = params.get('table_name')
            caveclient_kws = params.get('caveclient_kws')
            caveclient_kws = json.loads(caveclient_kws) if caveclient_kws is not None else None
            caveclient_kws['auth_token'] = cvt
            client = set_CAVEclient(
                datastack='minnie65_phase3_v1', 
                ver=ver, 
                caveclient_kws=caveclient_kws
            )
            nucleus_df = client.materialize.query_table('nucleus_detection_v0', split_positions=True)
            alternative_points_df = client.materialize.query_table('nucleus_alternative_points', split_positions=True)
            matching_inds = nucleus_df.merge(alternative_points_df, left_on='id', right_on='id_ref', how='left').query('pt_root_id_y>0').index
            sub_nucleus_df = nucleus_df.drop(matching_inds)
            alternative_points_mod_df = alternative_points_df[['valid', 'target_id', 'pt_position_x', 'pt_position_y', 'pt_position_z', 'pt_supervoxel_id', 'pt_root_id']].rename(columns={'target_id': 'nucleus_id'})
            nucleus_df = pd.concat([sub_nucleus_df.rename(columns={'id': 'nucleus_id'}), alternative_points_mod_df])
            cell_type_df = client.materialize.query_table(table_name).rename(columns={'target_id': 'nucleus_id'})
            df = cell_type_df.merge(nucleus_df, on=['nucleus_id', 'pt_root_id'], suffixes=['_ct', '_det']).rename(columns={
                'pt_root_id': 'segment_id',
                'pt_position_x': 'nucleus_x',
                'pt_position_y': 'nucleus_y',
                'pt_position_z': 'nucleus_z',
            })[['nucleus_id', 'segment_id', 'nucleus_x', 'nucleus_y', 'nucleus_z', 'classification_system', 'cell_type']].drop_duplicates()
            df[self.hash_name] = params.get(self.hash_name)
            return df

    class UnitSource(djp.Part):
        enable_hashing = True
        hash_name = 'import_method_id'
        hashed_attrs = Tag.attr_name
        definition = """
        # method for importing nda.UnitSource
        -> master
        ---
        -> Tag
        """
        @classmethod
        def update_method(cls):
            to_insert = {
                Tag.attr_name : Tag.version
            }
            cls.insert(to_insert, insert_to_master=True)
        
        def run(self, **kwargs):
            params = (self & kwargs).fetch1()
            df = pd.DataFrame((nda.UnitSource.proj(..., 'mask_type','brain_area', unit_x='stack_x', unit_y='stack_y', unit_z='stack_z') & 'unit_x>750').fetch())
            df[self.hash_name] = params.get(self.hash_name)
            return df


@schema
class ImportedTable(djp.Lookup):
    hash_name = 'imported_table_id'
    definition = """
    # imported tables
    imported_table_id                    : varchar(6)   # identifier for instance of imported table
    ---
    imported_table_ts=CURRENT_TIMESTAMP  : timestamp    # timestamp table imported
    """
    
    @classmethod
    def Info(cls):
        raise NotImplementedError(f"not implemented for {cls.class_name}")
    
    class NucleusNeuronSVM(djp.Part, dj.Computed):
        enable_hashing = True
        hash_name = 'imported_table_id'
        hashed_attrs = ImportMethod.hash_name
        definition = """
        # import of CAVE table nucleus_neuron_svm
        -> master
        -> ImportMethod
        nucleus_id                : bigint unsigned              # nucleus id unique within each Segmentation
        ---
        segment_id                : bigint unsigned              # id of the segment under the nucleus centroid. Equivalent to Allen 'pt_root_id'.
        nucleus_x                 : int unsigned                 # x coordinate of nucleus centroid in EM voxels (x: 4nm, y: 4nm, z: 40nm)
        nucleus_y                 : int unsigned                 # y coordinate of nucleus centroid in EM voxels (x: 4nm, y: 4nm, z: 40nm)
        nucleus_z                 : int unsigned                 # z coordinate of nucleus centroid in EM voxels (x: 4nm, y: 4nm, z: 40nm)
        cell_type                 : varchar(256)                 # cell type classification
        """
        
        @classmethod
        def Info(cls):
            return cls * ImportMethod.NucleusNeuronSVM
        
        @classproperty
        def key_source(cls):
            return ImportMethod & ImportMethod.NucleusNeuronSVM
        
        def make(self, key):
            
            df = ImportMethod.run(key)
            df[self.hash_name] = self.hash1(key)
            
            self.insert(
                df, 
                skip_hashing=True,
                ignore_extra_fields=True, 
                insert_to_master=True, 
                insert_to_master_kws={
                    'skip_duplicates': True, 
                    'ignore_extra_fields': True
                })
    
    class MetaModelMtypes(djp.Part, dj.Computed):
        enable_hashing = True
        hash_name = 'imported_table_id'
        hashed_attrs = ImportMethod.hash_name
        definition = """
        # import of CAVE table nucleus_neuron_svm
        -> master
        -> ImportMethod
        nucleus_id                : bigint unsigned              # nucleus id unique within each Segmentation
        ---
        segment_id                : bigint unsigned              # id of the segment under the nucleus centroid. Equivalent to Allen 'pt_root_id'.
        nucleus_x                 : int unsigned                 # x coordinate of nucleus centroid in EM voxels (x: 4nm, y: 4nm, z: 40nm)
        nucleus_y                 : int unsigned                 # y coordinate of nucleus centroid in EM voxels (x: 4nm, y: 4nm, z: 40nm)
        nucleus_z                 : int unsigned                 # z coordinate of nucleus centroid in EM voxels (x: 4nm, y: 4nm, z: 40nm)
        classification_system     : varchar(256)                 # e/i cell type classification
        cell_type                 : varchar(256)                 # fine cell type classification
        """
        
        @classmethod
        def Info(cls):
            return cls * ImportMethod.MetaModelMtypes
        
        @classproperty
        def key_source(cls):
            return ImportMethod & ImportMethod.MetaModelMtypes
        
        def make(self, key):
            
            df = ImportMethod.run(key)
            df[self.hash_name] = self.hash1(key)
            
            self.insert(
                df, 
                skip_hashing=True,
                ignore_extra_fields=True, 
                insert_to_master=True, 
                insert_to_master_kws={
                    'skip_duplicates': True, 
                    'ignore_extra_fields': True
                })

    class UnitSource(djp.Part, dj.Computed):
        enable_hashing = True
        hash_name = 'imported_table_id'
        hashed_attrs = ImportMethod.hash_name
        definition = """
        # import of nda.UnitSource
        -> master
        -> ImportMethod
        scan_session         : smallint                     # session index for the mouse
        scan_idx             : smallint                     # number of TIFF stack file
        unit_id              : int                          # unique per scan
        ---
        field               : smallint                      # Field Number
        unit_x              : float                         # centroid x stack coordinates (microns)
        unit_y              : float                         # centroid y stack coordinates (microns)
        unit_z              : float                         # centroid z stack coordinates (microns)
        mask_type           : varchar(16)                   # mask classification
        mask_id             : smallint                      # 
        brain_area          : varchar(256)                  # area name
        """
        
        @classmethod
        def Info(cls):
            return cls * ImportMethod.UnitSource
        
        @classproperty
        def key_source(cls):
            return ImportMethod & ImportMethod.UnitSource
        
        def make(self, key):
            df = ImportMethod.run(key)
            df[self.hash_name] = self.hash1(key)

            self.insert(
                df, 
                skip_hashing=True,
                ignore_extra_fields=True, 
                insert_to_master=True, 
                insert_to_master_kws={
                    'skip_duplicates': True, 
                    'ignore_extra_fields': True
                })
            

@schema
class CoregistrationMethod(djp.Lookup):
    hash_name = 'coreg_method_id'
    definition = """
    coreg_method_id   : varchar(8) # id of coregistration method
    """
    @classmethod
    def run(cls, coords, key=None, transform_id=None):
        if transform_id is not None:
            key = {'transform_id': transform_id}
        return cls.r1p(key).run(coords, key=key)
    
    class AIBS(djp.Part, dj.Imported):
        enable_hashing=True
        hash_name = 'coreg_method_id'
        hashed_attrs = m65cor.Coregistration.hash_name
        definition = """
        -> master
        -> m65cor.Coregistration
        ---
        transform_id         : int                          # id of the transform
        direction            : varchar(16)                  # direction of the transform (EMTP: EM -> 2P, TPEM: 2P -> EM)
        """
        @classmethod
        def Info(cls):
            return cls.proj() * m65cor.Coregistration.AIBS
        
        @classproperty
        def key_source(cls):
            return m65cor.Coregistration.AIBS & 'transform_id>=5'
        
        def make(self, key):
            to_insert = (self.key_source & key).fetch('coregistration', 'transform_id', 'direction', as_dict=True)[0]
            self.insert1(to_insert, insert_to_master=True, ignore_extra_fields=True)
        
        def run(self, coords, key=None, transform_id=None):
            if key is not None:
                key = (self & key).fetch1()
            return m65cor.Coregistration.run(coords, key=key, transform_id=transform_id)


@schema
class CoregistrationMethodSet(djp.Lookup):
    hash_name = 'coreg_method_set_id'
    definition = f"""
    {hash_name} : varchar(6) # 
    """
    
    class AIBS(djp.Part, dj.Lookup):
        enable_hashing = True
        hash_name = 'coreg_method_set_id'
        hashed_attrs = 'coreg_method_id'
        hash_group = True
        definition = """
        -> master
        -> CoregistrationMethod
        """
        @classmethod
        def Info(cls):
            return cls * CoregistrationMethod.AIBS


@schema
class GeneralMethod(djp.Lookup):
    hash_name = 'gen_method_id'
    definition = """
    gen_method_id : varchar(8) # id of generic methods
    ---
    gen_method_id_ts=CURRENT_TIMESTAMP : timestamp # timestamp method inserted
    """
        
    @classmethod
    def run(cls, key):
        return cls.r1p(key).run(**key)
    
    class NucleusNucleusDistance(djp.Part):
        enable_hashing = True
        hash_name = 'gen_method_id'
        hashed_attrs = 'dmax', Tag.attr_name
        definition = """
        # method for computing distances between pairs of nearby nuclei
        -> master
        ---
        dmax : smallint # distance (um) from nucleus to bounding box edge of nucleus inclusion in all dimensions (x, y, z)
        -> Tag
        """
        
        @classmethod
        def update_method(cls, dmax):
            cls.insert1({
                'dmax': dmax,
                Tag.attr_name : Tag.version
            }, insert_to_master=True)
            
        def run(self, nucleus_df, **kwargs):
            params = (self & kwargs).fetch1()
            dmax = params.get('dmax')
            for attr in ['nucleus_id', 'nucleus_x', 'nucleus_y', 'nucleus_z']:
                assert attr in nucleus_df.columns, f'nucleus_df must contain column: {attr}'
            coords = nucleus_df[['nucleus_x', 'nucleus_y', 'nucleus_z']].values * np.array([4, 4, 40]) / 1000 # um
            labels = nucleus_df.nucleus_id.values

            dfs = []
            self.Log('info', '-> computing distances')
            for p in tqdm(coords, total=len(coords)):
                xs, ys, zs = np.abs((coords-p)).T
                nearest_mask = (1 * (xs <= dmax) * 1 * (ys <= dmax) * 1*(zs <= dmax)).astype(bool)
                nucs = labels[nearest_mask]
                dxyz = (cdist(p[None, :], coords[nearest_mask]))[0]
                dxy = (cdist(p[None, :][:, [2, 0]], coords[nearest_mask][:, [2, 0]]))[0]
                dxz = (cdist(p[None, :][:, [2, 1]], coords[nearest_mask][:, [2, 1]]))[0]
                dyz = (cdist(p[None, :][:, [0, 1]], coords[nearest_mask][:, [0, 1]]))[0]
                dx = (cdist(p[None, :][:, [2]], coords[nearest_mask][:, [2]]))[0]
                dy = (cdist(p[None, :][:, [0]], coords[nearest_mask][:, [0]]))[0]
                dz = (cdist(p[None, :][:, [1]], coords[nearest_mask][:, [1]]))[0]
                # make dataframe
                inds = np.argsort(dxyz)
                nucs_sorted = nucs[inds]
                nucleus_id = nucs_sorted[0]
                secondary_nucs = nucs_sorted[1:]
                dfs.append(
                    pd.DataFrame({
                        'nucleus_id': np.repeat(nucleus_id, len(secondary_nucs)), 
                        'secondary_nucleus_id': secondary_nucs, 
                        'distance': dxyz[inds][1:],
                        'distance_xy': dxy[inds][1:],
                        'distance_xz': dxz[inds][1:],
                        'distance_yz': dyz[inds][1:],
                        'distance_x': dx[inds][1:],
                        'distance_y': dy[inds][1:],
                        'distance_z': dz[inds][1:],
                    }))
            self.Log('info', '-> making dataframe')
            final_df = pd.concat(dfs).reset_index(drop=True)
            final_df[self.hash_name] = params.get(self.hash_name)
            return final_df
    
    class UnitNucleusDistance(djp.Part):
        enable_hashing = True
        hash_name = 'gen_method_id'
        hashed_attrs = 'dmax', Tag.attr_name, CoregistrationMethod.hash_name
        definition = """
        # method for computing distances between units and nearby nuclei after the coregistration transform
        -> master
        ---
        dmax : smallint # distance (um) from nucleus to nearby nuclei inclusion bounding box edge in all dimensions
        -> CoregistrationMethod
        -> Tag
        """
        
        @classmethod
        def update_method(cls, dmax, coreg_method_id):
            cls.insert1({
                'dmax': dmax,
                'coreg_method_id': coreg_method_id,
                Tag.attr_name : Tag.version
            }, insert_to_master=True)
                   
        def run(self, unit_df, nucleus_df, **kwargs):
            params = (self & kwargs).fetch1()
            dmax = params.get('dmax')
            assert np.all(np.isin(['scan_session', 'scan_idx', 'field', 'unit_id', 'unit_x', 'unit_y', 'unit_z'], unit_df.columns))
            assert np.all(np.isin(['nucleus_id', 'nucleus_x', 'nucleus_y', 'nucleus_z'], nucleus_df.columns))
            
            unit_labels = unit_df[['scan_session', 'scan_idx', 'field', 'unit_id']].values
            unit_coords = unit_df[['unit_x', 'unit_y', 'unit_z']].values
            nuc_coords = nucleus_df[['nucleus_x', 'nucleus_y', 'nucleus_z']].values 
            nuc_labels = nucleus_df.nucleus_id.values
            
            # transform coords
            self.Log('info', '-> transforming coordinates')
            transform_id, direction = CoregistrationMethod.r1p(params).fetch1('transform_id', 'direction')
            if direction == "EM2P":
                nuc_coords = CoregistrationMethod.run(nuc_coords, transform_id=transform_id)
                xdim = 0
                ydim = 1
                zdim = 2
            elif direction == "2PEM":
                unit_coords = CoregistrationMethod.run(unit_coords, transform_id=transform_id) * np.array([4, 4, 40]) / 1000
                nuc_coords = nuc_coords * np.array([4, 4, 40]) / 1000
                xdim = 2
                ydim = 0
                zdim = 1
            else:
                raise AttributeError(f'transform direction {direction} not recognized.')

            dfs = []
            self.Log('info', '-> computing distances')
            for uxyz, (ss, si, f, u) in tqdm(zip(unit_coords, unit_labels), total=len(unit_coords)):
                xs, ys, zs = np.abs(uxyz - nuc_coords).T
                nearest_mask = (1 * (xs <= dmax) * 1 * (ys <= dmax) * 1*(zs <= dmax)).astype(bool)
                nucs = nuc_labels[nearest_mask]
                dxyz = (cdist(uxyz[None, :], nuc_coords[nearest_mask]))[0]
                dxy = (cdist(uxyz[None, :][:, [xdim, ydim]], nuc_coords[nearest_mask][:, [xdim, ydim]]))[0]
                dxz = (cdist(uxyz[None, :][:, [xdim, zdim]], nuc_coords[nearest_mask][:, [xdim, zdim]]))[0]
                dyz = (cdist(uxyz[None, :][:, [ydim, zdim]], nuc_coords[nearest_mask][:, [ydim, zdim]]))[0]
                dx = (cdist(uxyz[None, :][:, [xdim]], nuc_coords[nearest_mask][:, [xdim]]))[0]
                dy = (cdist(uxyz[None, :][:, [ydim]], nuc_coords[nearest_mask][:, [ydim]]))[0]
                dz = (cdist(uxyz[None, :][:, [zdim]], nuc_coords[nearest_mask][:, [zdim]]))[0]
                # make dataframe
                inds = np.argsort(dxyz)
                dfs.append(
                    pd.DataFrame({
                        'scan_session': np.repeat(ss, len(inds)), 
                        'scan_idx': np.repeat(si, len(inds)), 
                        'field': np.repeat(f, len(inds)), 
                        'unit_id': np.repeat(u, len(inds)), 
                        'nucleus_id': nucs[inds], 
                        'distance': dxyz[inds],
                        'distance_xy': dxy[inds],
                        'distance_xz': dxz[inds],
                        'distance_yz': dyz[inds],
                        'distance_x': dx[inds],
                        'distance_y': dy[inds],
                        'distance_z': dz[inds],
                    }))
            self.Log('info', '-> making dataframe')
            final_df = pd.concat(dfs).reset_index(drop=True)
            final_df[self.hash_name] = params.get(self.hash_name)
            return final_df 


@schema
class NucleusNucleusDistance(djp.Lookup):
    hash_name = 'nnd_id'
    definition = """
    nnd_id                       : varchar(6)   # identifier for instance of computed nucleus-nucleus distance
    """
    class Maker(djp.Part, dj.Computed):
        enable_hashing=True
        hash_name = 'nnd_make_id'
        hashed_attrs = ImportedTable.hash_name, GeneralMethod.hash_name
        definition = """
        # maker for NucleusNucleusDistance.Store
        -> master.proj(nnd_make_id='nnd_id')
        -> ImportedTable
        -> GeneralMethod
        ---
        nnd_ts=CURRENT_TIMESTAMP     : timestamp    # timestamp table inserted
        """
        
        @classproperty
        def key_source(cls):
            return (ImportedTable & ImportedTable.NucleusNeuronSVM) * (GeneralMethod & GeneralMethod.NucleusNucleusDistance)
        
        def make(self, key):
            neuron_rel = ImportedTable.r1p(key) & 'cell_type="neuron"'
            key['nucleus_df'] = pd.DataFrame(neuron_rel.fetch('nucleus_id', 'nucleus_x', 'nucleus_y', 'nucleus_z', as_dict=True))
            df = GeneralMethod.run(key)
            df[ImportedTable.NucleusNeuronSVM.hash_name] = key.get(ImportedTable.NucleusNeuronSVM.hash_name)
            
            # generate hash
            key_hash = self.hash1(key)
            df[self.hash_name] = key_hash
            df[self.master.hash_name] = key_hash
            self.Log('info', '-> inserting dataframe')
            # insert
            self.insert(
                df, 
                insert_to_master=True, 
                skip_hashing=True, 
                insert_to_master_kws={
                    'ignore_extra_fields': True, 
                    'skip_duplicates': True
                },
                ignore_extra_fields=True,
                skip_duplicates=True
            )
            self.master.Store.insert(df, ignore_extra_fields=True)

    class Store(djp.Part):
        hash_name = 'nnd_id'
        definition = """
        # nucleus nucleus distance 
        nucleus_id             : bigint unsigned   # id of nucleus
        secondary_nucleus_id   : bigint unsigned   # id of nucleus paired with nucleus_id
        -> master
        --- 
        distance               : float             # euclidean distance (um) from nucleus_id to secondary_nucleus_id in x, y, z (2p ref frame)
        distance_xy            : float             # euclidean distance (um) from nucleus_id to secondary_nucleus_id in x, y (2p ref frame)
        distance_xz            : float             # euclidean distance (um) from nucleus_id to secondary_nucleus_id in x, z (2p ref frame)
        distance_yz            : float             # euclidean distance (um) from nucleus_id to secondary_nucleus_id in y, z (2p ref frame)
        distance_x             : float             # euclidean distance (um) from nucleus_id to secondary_nucleus_id in x (2p ref frame)
        distance_y             : float             # euclidean distance (um) from nucleus_id to secondary_nucleus_id in y (2p ref frame)
        distance_z             : float             # euclidean distance (um) from nucleus_id to secondary_nucleus_id in z (2p ref frame)
        """


@schema
class TableGroup(djp.Lookup):
    hash_name = 'table_group_id'
    definition = """
    table_group_id   : varchar(6) # id of table group
    ---
    table_group_id_ts=CURRENT_TIMESTAMP : timestamp
    """
    @classmethod
    def get(cls, key, **kwargs):
        return cls.r1p(key).get(key, **kwargs)
    
    class UnitNucleusDistance(djp.Part):
        enable_hashing = True
        hash_name = 'table_group_id'
        hashed_attrs = ImportedTable.hash_name, 'table_type'
        hash_group = True
        definition = """
        -> master
        -> ImportedTable
        ---
        table_type : varchar(128) # type of table
        """
        
        @classmethod
        def fill(cls, key):
            cls.insert(key, insert_to_master=True, ignore_extra_fields=True)
            
        def get(self, key):
            unit_rel = ImportedTable.r1p((self & key & 'table_type="unit"'))
            nucleus_rel = ImportedTable.r1p((self & key & 'table_type="nucleus"'))
            return unit_rel, nucleus_rel


@schema
class UnitNucleusDistance(djp.Lookup):
    hash_name = 'und_id'
    definition = """
    und_id                       : varchar(6)   # identifier for instance of computed unit-nucleus distance
    """
    @classmethod
    def Info(cls):
        return cls.Maker * GeneralMethod.UnitNucleusDistance * CoregistrationMethod.AIBS
    
    class Maker(djp.Part, dj.Computed):
        enable_hashing=True
        hash_name = 'und_make_id'
        hashed_attrs = TableGroup.hash_name, GeneralMethod.hash_name
        definition = """
        # maker for UnitNucleusDistance.Store
        -> master.proj(und_make_id='und_id')
        -> TableGroup
        -> GeneralMethod
        ---
        und_ts=CURRENT_TIMESTAMP     : timestamp    # timestamp table inserted
        """
        
        @classproperty
        def key_source(cls):
            return (TableGroup & TableGroup.UnitNucleusDistance) * (GeneralMethod & GeneralMethod.UnitNucleusDistance)
        
        def make(self, key):
            unit_rel, nucleus_rel = TableGroup.get(key)
            key['unit_df'] = pd.DataFrame(unit_rel.fetch('scan_session', 'scan_idx', 'field', 'unit_id', 'unit_x', 'unit_y', 'unit_z', as_dict=True))
            key['nucleus_df'] = pd.DataFrame((nucleus_rel & 'cell_type="neuron"').fetch('nucleus_id', 'nucleus_x', 'nucleus_y', 'nucleus_z', as_dict=True))
            df = GeneralMethod.run(key)
            df[TableGroup.UnitNucleusDistance.hash_name] = key.get(TableGroup.UnitNucleusDistance.hash_name)
            
            # hash key
            key_hash = self.hash1(key)
            df[self.hash_name] = key_hash
            df[self.master.hash_name] = key_hash
            
            # insert
            self.Log('info', '-> inserting dataframe')
            self.insert(
                df, 
                insert_to_master=True, 
                skip_hashing=True, 
                insert_to_master_kws={
                    'ignore_extra_fields': True, 
                    'skip_duplicates': True
                },
                ignore_extra_fields=True,
                skip_duplicates=True
            )
            self.master.Store.insert(df, ignore_extra_fields=True)
    
    class MetaModelMaker(djp.Part, dj.Computed):
        enable_hashing=True
        hash_name = 'und_make_id'
        hashed_attrs = TableGroup.hash_name, GeneralMethod.hash_name
        definition = """
        # maker for UnitNucleusDistance.Store
        -> master.proj(und_make_id='und_id')
        -> TableGroup
        -> GeneralMethod
        ---
        und_ts=CURRENT_TIMESTAMP     : timestamp    # timestamp table inserted
        """
        
        @classproperty
        def key_source(cls):
            return (TableGroup & TableGroup.UnitNucleusDistance) * (GeneralMethod & GeneralMethod.UnitNucleusDistance)
        
        def make(self, key):
            unit_rel, nucleus_rel = TableGroup.get(key)
            key['unit_df'] = pd.DataFrame(unit_rel.fetch('scan_session', 'scan_idx', 'field', 'unit_id', 'unit_x', 'unit_y', 'unit_z', as_dict=True))
            key['nucleus_df'] = pd.DataFrame((nucleus_rel & 'classification_system="excitatory_neuron"').fetch('nucleus_id', 'nucleus_x', 'nucleus_y', 'nucleus_z', as_dict=True))
            df = GeneralMethod.run(key)
            df[TableGroup.UnitNucleusDistance.hash_name] = key.get(TableGroup.UnitNucleusDistance.hash_name)
            
            # hash key
            key_hash = self.hash1(key)
            df[self.hash_name] = key_hash
            df[self.master.hash_name] = key_hash
            
            # insert
            self.Log('info', '-> inserting dataframe')
            self.insert(
                df, 
                insert_to_master=True, 
                skip_hashing=True, 
                insert_to_master_kws={
                    'ignore_extra_fields': True, 
                    'skip_duplicates': True
                },
                ignore_extra_fields=True,
                skip_duplicates=True
            )
            self.master.Store.insert(df, ignore_extra_fields=True)

    class Store(djp.Part):
        hash_name = 'und_id'
        definition = """
        # unit nucleus distance between ImportedTable.UnitSource & ImportedTable.NucleusNeuronSVM
        scan_session          : smallint                     # session index for the mouse
        scan_idx              : smallint                     # number of TIFF stack file
        unit_id               : int                          # unique per scan
        nucleus_id            : bigint unsigned              # id of nucleus paired with unit
        field                  : smallint                     # Field Number
        -> master
        --- 
        distance               : float             # euclidean distance (um) from nucleus_id to secondary_nucleus_id in x, y, z (2p ref frame)
        distance_xy            : float             # euclidean distance (um) from nucleus_id to secondary_nucleus_id in x, y (2p ref frame)
        distance_xz            : float             # euclidean distance (um) from nucleus_id to secondary_nucleus_id in x, z (2p ref frame)
        distance_yz            : float             # euclidean distance (um) from nucleus_id to secondary_nucleus_id in y, z (2p ref frame)
        distance_x             : float             # euclidean distance (um) from nucleus_id to secondary_nucleus_id in x (2p ref frame)
        distance_y             : float             # euclidean distance (um) from nucleus_id to secondary_nucleus_id in y (2p ref frame)
        distance_z             : float             # euclidean distance (um) from nucleus_id to secondary_nucleus_id in z (2p ref frame)
        index(nucleus_id)
        """

        @classmethod
        def get_n_nearest(cls, nearest_type, n=5, order_by='distance', match_id=None, match_key=None):
            assert (match_id is None) != (match_key is None), 'Must provide either match_id or match_key, but not both'

            if match_id is not None:
                match_key = Match.r1pwh(match_id).fetch1()
            
            if match_key is not None:
                match_key = Match.r1p(match_id).fetch1()
            
            make_key_from_attrs = lambda attrs, source: {a: source[a] for a in attrs}

            if nearest_type == 'unit':
                key = {'scan_session': match_key.get('scan_session'), 'scan_idx': match_key.get('scan_idx'), 'unit_id': match_key.get('unit_id')}
            elif nearest_type == 'nucleus':
                key = {'nucleus_id': match_key.get('nucleus_id'), 'field': match_key.get('field')}

            return pd.DataFrame((cls & key).fetch(limit=n, order_by=order_by))


@schema
class M6502Import(djp.Lookup):
    hash_name = 'm6502_id'
    definition = f"""
    {hash_name} : varchar(8) # 
    """
    
    @classmethod
    def fill(cls):
        cls.UnitManualMatchAttempt.fill()
        cls.NucleusManualMatchAttempt.fill()
        cls.UnitNucleusManualMatch.fill()
    
    class UnitManualMatchAttempt(djp.Part, dj.Computed):
        enable_hashing = True
        hash_name = 'm6502_id'
        hashed_attrs = 'unit_submission_id'
        definition = """
        -> master
        unit_submission_id   : varchar(16)                  # unique identifier for submission
        animal_id            : int                          # id number
        scan_session         : smallint                     # scan session index for the mouse
        scan_idx             : smallint                     # id of the scan within the session
        unit_id              : int                          # unique per scan & segmentation method
        ---
        user_id              : varchar(64)                  # User id to login and to match with neurons they have proofread.
        interface=null       : varchar(32)                  # matching interface used by the user
        priority=null        : varchar(32)                  # primary match, secondary match, or NULL (not applicable)
        unit_submission_type : varchar(16)                  # type of submission by the user
        field=null           : smallint                     # field of the scan
        protocol_name=null   : varchar(256)                 # name of protocol that unit was attempted under
        stack_session        : int                          # session index for the mouse
        stack_idx            : int                          # id of the stack
        transform_id         : int                          # id of the transform
        note=null            : varchar(256)                 # user notes
        timestamp=CURRENT_TIMESTAMP : timestamp             # timestamp
        """
    
        @classproperty
        def key_source(cls):
            return m65.UnitManualMatchAttempt
        
        def make(self, key):
            self.insert(self.key_source & key, insert_to_master=True)

        @classmethod
        def fill(cls):
            cls.Log('info', 'Filling UnitManualMatchAttempt')
            cls.insert(cls.key_source - cls, insert_to_master=True, allow_direct_insert=True)
        
    class NucleusManualMatchAttempt(djp.Part, dj.Computed):
        enable_hashing = True
        hash_name = 'm6502_id'
        hashed_attrs = 'nucleus_submission_id'
        definition = """
        -> master
        nucleus_submission_id : varchar(16)                 # unique identifier for submission
        nucleus_id           : bigint unsigned              # nucleus id unique within each Segmentation
        ---
        user_id              : varchar(64)                  # User id to login and to match with neurons they have proofread.
        interface=null       : varchar(32)                  # matching interface used by the user
        priority=null        : varchar(32)                  # primary match, secondary match, or NULL (not applicable)
        nucleus_submission_type : varchar(16)               # type of submission by the user
        animal_id            : int                          # id number
        scan_session=null    : smallint                     # scan session index for the mouse
        scan_idx=null        : smallint                     # id of the scan within the session
        field=null           : smallint                     # field of the scan
        protocol_name=null   : varchar(256)                 # name of protocol that unit was attempted under
        stack_session        : int                          # session index for the mouse
        stack_idx            : int                          # id of the stack
        transform_id         : int                          # id of the transform
        note=null            : varchar(256)                 # user notes
        timestamp=CURRENT_TIMESTAMP : timestamp             # timestamp
        """
        
        @classproperty
        def key_source(cls):
            return m65.NucleusManualMatchAttempt
        
        def make(self, key):
            self.insert(self.key_source & key, insert_to_master=True)
        
        @classmethod
        def fill(cls):
            cls.Log('info', 'Filling NucleusManualMatchAttempt')
            cls.insert(cls.key_source - cls, insert_to_master=True, allow_direct_insert=True)
    
    class UnitNucleusManualMatch(djp.Part, dj.Computed):
        enable_hashing = True
        hash_name = 'm6502_id'
        hashed_attrs = 'unit_submission_id', 'nucleus_submission_id'
        definition = """
        -> master
        unit_submission_id    : varchar(16)                  # unique identifier for submission
        nucleus_submission_id : varchar(16)                  # unique identifier for submission
        ---
        animal_id             : int                          # id number
        scan_session          : smallint                     # scan session index for the mouse
        scan_idx              : smallint                     # id of the scan within the session
        unit_id               : int                          # unique per scan & segmentation method
        nucleus_id            : bigint unsigned              # nucleus id unique within each Segmentation
        """
        
        @classproperty
        def key_source(cls):
            return m65.UnitNucleusManualMatch
        
        def make(self, key):
            self.insert(self.key_source & key, insert_to_master=True)
        
        @classmethod
        def fill(cls):
            cls.Log('info', 'Filling UnitNucleusManualMatch')
            cls.insert(cls.key_source - cls, insert_to_master=True, allow_direct_insert=True)

@schema
class MatchAttempt(djp.Lookup):
    hash_name = 'attempt_id'
    definition = f"""
    {hash_name} : varchar(10) # hash of match attempt
    """
    
    class M6502MakerError(djp.Part):
        hash_name = 'attempt_id'
        definition = """
        # Keys error
        -> master
        -> M6502Import
        ---
        ts_inserted=CURRENT_TIMESTAMP : timestamp
        """
        
    class M6502Maker(djp.Part, dj.Computed):
        enable_hashing = True
        hash_name = 'attempt_id'
        hashed_attrs = 'unit_submission_id', 'nucleus_submission_id'
        definition = """
        # Consolidates UnitManualMatchAttempt, NucleusManualMatchAttempt, and UnitNucleusManualMatch from schema microns_minnie65_02
        -> master
        -> M6502Import
        ---
        ts_inserted=CURRENT_TIMESTAMP : timestamp
        """
        
        @classproperty
        def key_source(cls):
            return M6502Import() - cls.master.M6502MakerError() & [M6502Import.UnitManualMatchAttempt, M6502Import.NucleusManualMatchAttempt] 
        
        def make(self, key):
            try:
                source_row = M6502Import.r1p(key)
                attrs = source_row.heading.names
                if 'unit_submission_id' in attrs:
                    source = 'unit'
                    other = 'nucleus'
                    other_rel = M6502Import.NucleusManualMatchAttempt

                elif 'nucleus_submission_id' in attrs:
                    source = 'nucleus'
                    other = 'unit'
                    other_rel = M6502Import.UnitManualMatchAttempt

                else:
                    raise AttributeError('Row attributes not recognized.')

                source_type_name = f'{source}_submission_type'
                source_id_name = f'{source}_submission_id'
                other_id_name = f'{other}_submission_id'

                source_submission_type = source_row.fetch1(source_type_name)
                if source_submission_type == 'match' or source_submission_type == 'match_uncertain':
                    match_row = M6502Import.UnitNucleusManualMatch & (dj.U(source_id_name) & source_row)
                    other_row = other_rel & (dj.U(other_id_name) & match_row)
                    source_row_df = pd.DataFrame(source_row.fetch())
                    # format match row
                    match_row_df = pd.DataFrame(match_row.fetch())
                    match_row_m6502_id = match_row_df.m6502_id.values[0]
                    match_row_df = match_row_df.drop(columns='m6502_id')
                    # format other row
                    other_row_df = pd.DataFrame(other_row.fetch())
                    other_row_m6502_id = other_row_df.m6502_id.values[0]
                    other_row_df = other_row_df.drop(columns=['m6502_id', 'timestamp', 'protocol_name'])
                    # merge to final row
                    final_row_df = source_row_df.merge(match_row_df).merge(other_row_df)
                    # maker row
                    maker_rows_df = pd.DataFrame([
                    {'m6502_id': source_row_df.m6502_id.values[0]},
                    {'m6502_id': other_row_m6502_id},
                    {'m6502_id': match_row_m6502_id}])

                else:
                    final_row_df = pd.DataFrame(source_row.fetch())
                    final_row_df[other_id_name] = None
                    maker_rows_df = pd.DataFrame([
                        {'m6502_id': final_row_df.m6502_id.values[0]}
                    ])

                if len(final_row_df) != 1:
                    raise ValueError('The number of rows after merging != 1, check for errors.')

                # add missing field information
                if final_row_df.field.values[0] is None:
                    if 'unit_id' in final_row_df:
                        final_row_df['field'] = (
                            nda.UnitSource & {
                                    'scan_session': final_row_df.scan_session.values[0], 
                                    'scan_idx': final_row_df.scan_idx.values[0],
                                    'unit_id': final_row_df.unit_id.values[0]
                                    }
                        ).fetch1('field')

                final_hash = self.hash1(final_row_df)
                final_row_df[self.hash_name] = final_hash
                maker_rows_df[self.hash_name] = final_hash
                self.master.M6502.insert(
                    final_row_df, 
                    insert_to_master=True, 
                    ignore_extra_fields=True,
                    insert_to_master_kws = {
                        'skip_duplicates': True, 
                        'ignore_extra_fields': True
                    })

                self.insert(
                    maker_rows_df, 
                    skip_hashing=True,
                    ignore_extra_fields=True
                )
            except:
                self.Log('error', f'errored on key {key}')
                key.update({'attempt_id': self.hash_len * '0'})
                self.master.M6502MakerError.insert1(
                    key, 
                    insert_to_master=True, 
                    insert_to_master_kws = {
                        'skip_duplicates': True, 
                        'ignore_extra_fields': True
                    }
                )
    
    class M6502(djp.Part):
        hash_name = 'attempt_id'
        definition = """
        # Consolidated match attempts from schema microns_minnie65_02
        -> master
        ---
        unit_submission_id=null        : varchar(16)                  # unique identifier for submission
        nucleus_submission_id=null     : varchar(16)                  # unique identifier for submission
        user_id                        : varchar(64)                  # User id to login and to match with neurons they have proofread.
        animal_id                      : int                          # id number
        scan_session=null              : smallint                     # scan session index for the mouse
        scan_idx=null                  : smallint                     # id of the scan within the session
        field=null                     : smallint                     # field of the scan
        unit_id=null                   : int                          # unique per scan & segmentation method
        nucleus_id=null                : bigint unsigned              # nucleus id unique within each Segmentation
        interface=null                 : varchar(32)                  # matching interface used by the user
        priority=null                  : varchar(32)                  # primary match, secondary match, or NULL (not applicable)
        unit_submission_type=null      : varchar(16)                  # type of unit submission by the user
        nucleus_submission_type=null   : varchar(16)                  # type of nucleus submission by the user
        protocol_name=null             : varchar(256)                 # name of protocol that unit was attempted under
        stack_session                  : int                          # session index for the mouse
        stack_idx                      : int                          # id of the stack
        transform_id                   : int                          # id of the transform
        note=null                      : varchar(256)                 # user notes
        timestamp                      : timestamp                    # timestamp
        """
        
        @classmethod
        def Match(cls):
            return cls & 'unit_submission_type="match"' & 'nucleus_submission_type="match"'
        
        @classmethod
        def MatchUncertain(cls):
            return cls & ['unit_submission_type="match_uncertain"', 'nucleus_submission_type="match_uncertain"']
        
        @classmethod
        def UnitNoMatch(cls):
            return cls & 'unit_submission_type="no_match"'
        
        @classmethod
        def UnitIndiscernable(cls):
            return cls & 'unit_submission_type="indiscernable"'
        
        @classmethod
        def UnitSkip(cls):
            return cls & 'unit_submission_type="skip"'
        
        @classmethod
        def NucleusNoMatch(cls):
            return cls & 'nucleus_submission_type="no_match"'

        @classmethod
        def NucleusInhibitory(cls):
            return cls & 'nucleus_submission_type="inhibitory"'
        
        @classmethod
        def NucleusNonNeuron(cls):
            return cls & 'nucleus_submission_type="non_neuron"'
        
        @classmethod
        def NucleusSkip(cls):
            return cls & 'nucleus_submission_type="skip"'

@schema
class ExclusionMethod(djp.Lookup):
    hash_name = 'exclusion_method_id'
    definition = """
    exclusion_method_id : varchar(8) # id of exclusion method
    """

    class Manual(djp.Part):
        enable_hashing = True
        insert_to_master = True
        hash_name = 'exclusion_method_id'
        hashed_attrs = 'exclusion_method_name', 'exclusion_method_desc'
        definition = """
        -> master
        ---
        exclusion_method_name : varchar(32)
        exclusion_method_desc=NULL : varchar(1000)
        exclusion_method_ts=CURRENT_TIMESTAMP : timestamp
        """

        @classmethod
        def update_method(cls, exclusion_method_name, exclusion_method_desc=None):
            d = {'exclusion_method_name': exclusion_method_name}
            if exclusion_method_desc is not None:
                d['exclusion_method_desc'] = exclusion_method_desc
            cls.insert1(d, insert_to_master=True)


@schema
class InclusionMethod(djp.Lookup):
    hash_name = 'inclusion_method_id'
    definition = """
    inclusion_method_id : varchar(8) # id of inclusion method
    """

    class Manual(djp.Part):
        enable_hashing = True
        insert_to_master = True
        hash_name = 'inclusion_method_id'
        hashed_attrs = 'inclusion_method_name', 'inclusion_method_desc'
        definition = """
        -> master
        ---
        inclusion_method_name : varchar(32)
        inclusion_method_desc=NULL : varchar(1000)
        inclusion_method_ts=CURRENT_TIMESTAMP : timestamp
        """

        @classmethod
        def update_method(cls, inclusion_method_name, inclusion_method_desc=None):
            d = {'inclusion_method_name': inclusion_method_name}
            if inclusion_method_desc is not None:
                d['inclusion_method_desc'] = inclusion_method_desc
            cls.insert1(d, insert_to_master=True)


@schema
class NucleusInhibitory(djp.Lookup):
    hash_name = 'attempt_id'
    definition = f"""
    {hash_name} : varchar(10) # hash of match attempt
    """

    @classmethod
    def Include(cls):
        return dj.U('nucleus_id') & (cls.M6502 - (dj.U('nucleus_id') & cls.ManualExclude))
    
    class M6502(djp.Part, dj.Computed):
        hash_name = 'attempt_id'
        definition = """
        -> master
        -> MatchAttempt.M6502
        nucleus_id            : bigint unsigned              # nucleus id unique within each Segmentation
        """

        @property
        def key_source(self):
            return MatchAttempt.M6502.NucleusInhibitory()

        def make(self, key):
            self.insert(MatchAttempt.M6502 & key, insert_to_master=True, ignore_extra_fields=True)

        @classmethod
        def fill(cls):
            cls.insert(cls.key_source - cls, insert_to_master=True, allow_direct_insert=True, ignore_extra_fields=True)

    class ManualExclude(djp.Part):
        hash_name = 'attempt_id'
        definition = """
        -> master
        -> ExclusionMethod
        nucleus_id            : bigint unsigned              # nucleus id unique within each Segmentation
        ---
        user_id=NULL          : varchar(64)                  # User id performing exclusion if applicable
        ts_inserted=CURRENT_TIMESTAMP : timestamp
        """

        @classmethod
        def exclude(cls, nucleus_id, exclusion_method_id='fe92685e', user_id=None):
            constant_attrs = {'exclusion_method_id': exclusion_method_id}
            if user_id is not None:
                constant_attrs['user_id'] = user_id
            cls.insert(
                cls.master.M6502 & {'nucleus_id': nucleus_id}, 
                constant_attrs=constant_attrs, 
                allow_direct_insert=True, 
                skip_duplicates=True
            )


@schema
class NucleusNonNeuron(djp.Lookup):
    hash_name = 'attempt_id'
    definition = f"""
    {hash_name} : varchar(10) # hash of match attempt
    """

    @classmethod
    def Include(cls):
        return dj.U('nucleus_id') & (cls.M6502 - (dj.U('nucleus_id') & cls.ManualExclude))
    
    class M6502(djp.Part, dj.Computed):
        hash_name = 'attempt_id'
        definition = """
        -> master
        -> MatchAttempt.M6502
        nucleus_id            : bigint unsigned              # nucleus id unique within each Segmentation
        """

        @property
        def key_source(self):
            return MatchAttempt.M6502.NucleusNonNeuron()

        def make(self, key):
            self.insert(MatchAttempt.M6502 & key, insert_to_master=True, ignore_extra_fields=True)

        @classmethod
        def fill(cls):
            cls.insert(cls.key_source - cls, insert_to_master=True, allow_direct_insert=True, ignore_extra_fields=True)

    class ManualExclude(djp.Part):
        hash_name = 'attempt_id'
        definition = """
        -> master
        -> ExclusionMethod
        nucleus_id            : bigint unsigned              # nucleus id unique within each Segmentation
        ---
        user_id=NULL          : varchar(64)                  # User id performing exclusion if applicable
        ts_inserted=CURRENT_TIMESTAMP : timestamp
        """

        @classmethod
        def exclude(cls, nucleus_id, user_id, exclusion_method_id='20e4b46f'):
            constant_attrs = dict(
                exclusion_method_id=exclusion_method_id,
                user_id=user_id
            )
            cls.insert(
                cls.master.M6502 & {'nucleus_id': nucleus_id}, 
                constant_attrs=constant_attrs, 
                allow_direct_insert=True, 
                skip_duplicates=True
            )


@schema
class MatchUserExclude(djp.Lookup):
    definition = f"""
    exclusion_method_id  : varchar(8)                   # id of exclusion method
    user_id              : varchar(10) # user id
    ---
    status=1             : tinyint # if status = 1 exclusion is active, if 0, inactive
    ts_inserted=CURRENT_TIMESTAMP : timestamp
    """
    @classmethod
    def add_user(cls, user_id, exclusion_method_id='86a50b78'):
        cls.insert1({
            'user_id': user_id, 
            'exclusion_method_id': exclusion_method_id
            }, skip_duplicates=True
        )
    
    @classmethod
    def toggle_user(cls, user_id, exclusion_method_id='86a50b78'):
        key = (cls & {'user_id': user_id, 'exclusion_method_id': exclusion_method_id}).fetch1()
        key['status'] = int(not bool(key['status'])) # flip status
        cls.insert1(key, replace=True)


@schema
class MatchManualExclude(djp.Lookup):
    definition = """
    exclusion_method_id  : varchar(8)                   # id of exclusion method
    scan_session         : smallint                     # scan session index for the mouse
    scan_idx             : smallint                     # id of the scan within the session
    field                : smallint                     # field of the scan
    unit_id              : int                          # unique per scan & segmentation method
    nucleus_id           : bigint unsigned              # nucleus id unique within each Segmentation
    ---
    exclude_user_id              : varchar(64)          # User performing exclusion
    ts_inserted=CURRENT_TIMESTAMP : timestamp           #
    """

    @classmethod
    def exclude(cls, key, exclusion_method_id, exclude_user_id):
        constant_attrs = dict(
            exclusion_method_id=exclusion_method_id,
            exclude_user_id=exclude_user_id
        )
        cls.insert1(
            key, 
            constant_attrs=constant_attrs, 
            allow_direct_insert=True, 
            skip_duplicates=True
        )


@schema
class MatchManualInclude(djp.Lookup):
    definition = """
    inclusion_method_id  : varchar(8)                   # id of inclusion method
    scan_session         : smallint                     # scan session index for the mouse
    scan_idx             : smallint                     # id of the scan within the session
    field                : smallint                     # field of the scan
    unit_id              : int                          # unique per scan & segmentation method
    nucleus_id           : bigint unsigned              # nucleus id unique within each Segmentation
    ---
    include_user_id              : varchar(64)          # User performing inclusion
    ts_inserted=CURRENT_TIMESTAMP : timestamp           #
    """

    @classmethod
    def include(cls, key, inclusion_method_id, user_id):
        constant_attrs = dict(
            inclusion_method_id=inclusion_method_id,
            user_id=user_id
        )
        cls.insert1(
            key, 
            constant_attrs=constant_attrs, 
            allow_direct_insert=True, 
            skip_duplicates=True
        )


@schema
class Match(djp.Lookup):
    hash_name = 'attempt_id'
    definition = f"""
    {hash_name} : varchar(10) # hash of match attempt
    """

    @classmethod
    def _include(cls, include_method):
        return MatchIncludeMethod.r1pwh(include_method).run()
    
    @classmethod
    def Include(cls, include_method='a4b7c4'):
        return cls._include(include_method=include_method)
    
    @classmethod
    def Exclude(cls, include_method='a4b7c4'):
        return cls.M6502 - cls._include(include_method=include_method).proj()
    
    class M6502(djp.Part, dj.Computed):
        hash_name = 'attempt_id'
        definition = """
        -> master
        -> MatchAttempt.M6502
        scan_session          : smallint                     # scan session index for the mouse
        scan_idx              : smallint                     # id of the scan within the session
        field                 : smallint                     # field of the scan
        unit_id               : int                          # unique per scan & segmentation method
        nucleus_id            : bigint unsigned              # nucleus id unique within each Segmentation
        index(nucleus_id)
        ---
        user_id              : varchar(64)                  # User id to login and to match with neurons they have proofread
        interface=null       : varchar(32)                  # matching interface used by the user
        priority=null        : varchar(32)                  # primary match, secondary match, or NULL (not applicable)
        protocol_name=null   : varchar(256)                 # name of protocol that unit was attempted under
        """

        @property
        def key_source(self):
            return MatchAttempt.M6502.Match()

        def make(self, key):
            self.insert(MatchAttempt.M6502 & key, insert_to_master=True, ignore_extra_fields=True)

        @classmethod
        def fill(cls):
            cls.insert(cls.key_source - cls.proj(), insert_to_master=True, allow_direct_insert=True, ignore_extra_fields=True)


@schema
class ScanInclude(djp.Lookup):
    hash_name = 'scan_include_set'
    definition = f"""
    {hash_name} : varchar(6) # id of scan include set
    """

    class Member(djp.Part):
        hash_name = 'scan_include_set'
        enable_hashing = True
        hashed_attrs = 'scan_session', 'scan_idx'
        hash_group = True
        definition = """
        -> master
        -> nda.Scan
        ---
        ts_inserted=CURRENT_TIMESTAMP : timestamp
        """

        @classmethod
        def fill(cls, key):
            cls.insert(key, insert_to_master=True)


@schema
class MatchIncludeMethod(djp.Lookup):
    hash_name = 'match_include_method'
    definition = f"""
    {hash_name} : varchar(6)
    """
    @classmethod
    def apply_dj_filters(
        cls,
        exclude_known_inhibitory=None,
        exclude_known_non_neurons=None,
        exclude_manual_exclusions=None,
        exclude_users=None,
        scan_include_set=None,
    ):
        source = Match.M6502
        if exclude_known_inhibitory:
            source -= NucleusInhibitory.Include()
        if exclude_known_non_neurons:
            source -= NucleusNonNeuron.Include()
        if exclude_manual_exclusions:
            source -= MatchManualExclude
        if exclude_users:
            source -= (MatchUserExclude & 'status=1')
        if scan_include_set is not None:
            source &= ScanInclude.r1pwh(scan_include_set)
        return source

    class DataJointView(djp.Part):
        enable_hashing = True
        hash_name = 'match_include_method'
        hashed_attrs = (
            'exclude_known_inhibitory',
            'exclude_known_non_neurons',
            'exclude_manual_exclusions',
            'exclude_users',
            'scan_include_set',
            Tag.attr_name
        )
        definition = """
        -> master
        ---
        exclude_known_inhibitory=NULL  : tinyint # 1 if True, 0 or NULL if False
        exclude_known_non_neurons=NULL : tinyint # 1 if True, 0 or NULL if False
        exclude_manual_exclusions=NULL : tinyint # 1 if True, 0 or NULL if False
        exclude_users=NULL             : tinyint # 1 if True, 0 or NULL if False
        scan_include_set=NULL          : varchar(6) # id of scan include set
        -> Tag
        """

        @classmethod
        def update_method(
            cls, 
            exclude_known_inhibitory=None,
            exclude_known_non_neurons=None,
            exclude_manual_exclusions=None,
            exclude_users=None,
            scan_include_set=None
        ):
            key = {
                'exclude_known_inhibitory': int(exclude_known_inhibitory) if exclude_known_inhibitory is not None else exclude_known_inhibitory,
                'exclude_known_non_neurons': int(exclude_known_non_neurons) if exclude_known_non_neurons is not None else exclude_known_non_neurons,
                'exclude_manual_exclusions': int(exclude_manual_exclusions) if exclude_manual_exclusions is not None else exclude_manual_exclusions,
                'exclude_users': int(exclude_users) if exclude_users is not None else exclude_users,
                'scan_include_set': scan_include_set,
                Tag.attr_name: Tag.version
            }
            cls.insert1(key, insert_to_master=True)
        
        def run(self, **kwargs):
            params = self.fetch1()
            assert params[Tag.attr_name] == Tag.version, 'Tag version mismatch'
            exclude_known_inhibitory = bool(params.get('exclude_known_inhibitory'))
            exclude_known_non_neurons = bool(params.get('exclude_known_non_neurons'))
            exclude_manual_exclusions = bool(params.get('exclude_manual_exclusions'))
            exclude_users = bool(params.get('exclude_users'))
            scan_include_set = params.get('scan_include_set')
            return self.master.apply_dj_filters(
                exclude_known_inhibitory=exclude_known_inhibitory,
                exclude_known_non_neurons=exclude_known_non_neurons,
                exclude_manual_exclusions=exclude_manual_exclusions,
                exclude_users=exclude_users,
                scan_include_set=scan_include_set
            )

    class Comprehensive(djp.Part):
        enable_hashing = True
        hash_name = 'match_include_method'
        hashed_attrs = (
            'exclude_known_inhibitory',
            'exclude_known_non_neurons',
            'exclude_manual_exclusions',
            'exclude_users',
            'scan_include_set',
            'drop_disagreements',
            'drop_attempt_data',
            'add_manual_include',
            'und_id',
            'min_residual',
            'min_nuc_sep',
            'min_unit_sep',
            'max_residual',
            'max_nuc_sep',
            'max_unit_sep',
            Tag.attr_name
        )
        definition = """
        -> master
        ---
        exclude_known_inhibitory=NULL  : tinyint # 1 if True, 0 or NULL if False
        exclude_known_non_neurons=NULL : tinyint # 1 if True, 0 or NULL if False
        exclude_manual_exclusions=NULL : tinyint # 1 if True, 0 or NULL if False
        exclude_users=NULL             : tinyint # 1 if True, 0 or NULL if False
        scan_include_set=NULL   : varchar(6) # id of scan include set
        drop_disagreements=NULL : varchar(32) # drop disagreements on "unit_id", "nucleus_id", or "both"
        drop_attempt_data=NULL  : tinyint    # drop attempt info. 1 if True, 0 or NULL if False
        add_manual_include=NULL     : tinyint    # 1 if True, 0 or NULL if False, only applicable if drop_attempt_data=1
        und_id=NULL             : varchar(6) # id of unit-nucleus distance
        min_residual=NULL       : float      # min residual allowed
        min_nuc_sep=NULL        : float      # min nucleus separation allowed
        min_unit_sep=NULL       : float      # min unit separation allowed
        max_residual=NULL       : float      # max residual allowed
        max_nuc_sep=NULL        : float      # max nucleus separation allowed
        max_unit_sep=NULL       : float      # max unit separation allowed
        -> Tag
        """

        @classmethod
        def update_method(
            cls, 
            exclude_known_inhibitory=None,
            exclude_known_non_neurons=None,
            exclude_manual_exclusions=None,
            exclude_users=None,
            scan_include_set=None,
            drop_disagreements=None,
            drop_attempt_data=None,
            add_manual_include=None,
            und_id=None,
            min_residual=None,
            min_nuc_sep=None,
            min_unit_sep=None,
            max_residual=None,
            max_nuc_sep=None,
            max_unit_sep=None,
        ):
            key = dict(
                exclude_known_inhibitory=int(exclude_known_inhibitory) if exclude_known_inhibitory is not None else exclude_known_inhibitory,
                exclude_known_non_neurons=int(exclude_known_non_neurons) if exclude_known_non_neurons is not None else exclude_known_non_neurons,
                exclude_manual_exclusions=int(exclude_manual_exclusions) if exclude_manual_exclusions is not None else exclude_manual_exclusions,
                exclude_users=int(exclude_users) if exclude_users is not None else exclude_users,
                scan_include_set=scan_include_set if scan_include_set is not None else scan_include_set,
                drop_disagreements=drop_disagreements,
                drop_attempt_data=int(drop_attempt_data) if drop_attempt_data is not None else drop_attempt_data,
                add_manual_include=int(add_manual_include) if add_manual_include is not None else add_manual_include,
                und_id=und_id,
                min_residual=min_residual,
                min_nuc_sep=min_nuc_sep,
                min_unit_sep=min_unit_sep,
                max_residual=max_residual,
                max_nuc_sep=max_nuc_sep,
                max_unit_sep=max_unit_sep,
            )
            key[Tag.attr_name]= Tag.version
            cls.insert1(key, insert_to_master=True)
        
        def run(self, **kwargs):
            params = self.fetch1()
            assert params[Tag.attr_name] == Tag.version, 'Tag version mismatch'
            exclude_known_inhibitory = bool(params.get('exclude_known_inhibitory'))
            exclude_known_non_neurons = bool(params.get('exclude_known_non_neurons'))
            exclude_manual_exclusions = bool(params.get('exclude_manual_exclusions'))
            exclude_users = bool(params.get('exclude_users'))
            scan_include_set = params.get('scan_include_set')
            drop_disagreements = params.get('drop_disagreements')
            drop_attempt_data = bool(params.get('drop_attempt_data'))
            add_manual_include = bool(params.get('add_manual_include'))
            und_id = params.get('und_id')
            min_residual = params.get('min_residual')
            min_nuc_sep = params.get('min_nuc_sep')
            min_unit_sep = params.get('min_unit_sep')
            max_residual = params.get('max_residual')
            max_nuc_sep = params.get('max_nuc_sep')
            max_unit_sep = params.get('max_unit_sep')

            self.Log('info', 'applying datajoint filters to match table')
            match_rel = self.master.apply_dj_filters(
                exclude_known_inhibitory=exclude_known_inhibitory,
                exclude_known_non_neurons=exclude_known_non_neurons,
                exclude_manual_exclusions=exclude_manual_exclusions,
                exclude_users=exclude_users,
                scan_include_set=scan_include_set
            )
            
            self.Log('info', 'fetching match table')
            match_df = pd.DataFrame(match_rel.fetch())
            
            if drop_disagreements is not None:
                self.Log('info', 'dropping disagreements')
                match_df = match.drop_match_disagreements(match_df, on=drop_disagreements)

            if drop_attempt_data:
                self.Log('info', 'dropping attempt data')
                match_df = match_df[['scan_session', 'scan_idx', 'field', 'unit_id', 'nucleus_id']].drop_duplicates()

                if add_manual_include:
                    self.Log('info', 'adding manual include')
                    manual_include_df = pd.DataFrame(MatchManualInclude.fetch())
                    manual_include_df = manual_include_df[['scan_session', 'scan_idx', 'field', 'unit_id', 'nucleus_id']].drop_duplicates()
                    match_df = pd.concat([match_df, manual_include_df], 0)

            if und_id is not None:
                self.Log('info', 'fetching unit nucleus distance table')
                und_df = pd.DataFrame(UnitNucleusDistance.r1pwh(und_id).fetch())
                
                self.Log('info', 'adding metrics to match table')
                interm_df = match.add_residual_to_match_df(match_df.drop_duplicates(['scan_session', 'scan_idx', 'field', 'unit_id', 'nucleus_id']), und_df)
                interm_df = match.add_separation_to_match_df(interm_df.drop_duplicates(['scan_session', 'scan_idx', 'field', 'unit_id', 'nucleus_id']), und_df)
                match_df = match_df.merge(interm_df)
                

                self.Log('info', 'apply metric filters')
                match_df = match_df.query(f'residual>={min_residual}') if min_residual is not None else match_df
                match_df = match_df.query(f'residual<{max_residual}') if max_residual is not None else match_df
                match_df = match_df.query(f'nuc_sep>={min_nuc_sep}') if min_nuc_sep is not None else match_df
                match_df = match_df.query(f'nuc_sep<{max_nuc_sep}') if max_nuc_sep is not None else match_df
                match_df = match_df.query(f'unit_sep>={min_unit_sep}') if min_unit_sep is not None else match_df
                match_df = match_df.query(f'unit_sep<{max_unit_sep}') if max_unit_sep is not None else match_df
            
            match_df = match_df.sort_values(by=['scan_session', 'scan_idx', 'field', 'unit_id', 'nucleus_id'])

            return FieldDict(match_table=match_df)


@schema
class MatchTable(djp.Lookup):
    hash_name = 'match_table_id'
    definition = f"""
    {hash_name} : varchar(8)
    """

    @classmethod
    def get(cls, key):
        results = []
        for store in cls.stores:
            results.extend(store().get(key))
        return results

    @classmethod
    def get1(cls, key):
        return cls.r1p(key, include_parts=cls.stores).get1()
    
    class Maker(djp.Part, dj.Computed):
        enable_hashing = True
        hash_name = 'match_table_id'
        hashed_attrs = 'match_include_method', 'ts_computed'
        definition = """
        -> master
        -> MatchIncludeMethod
        ts_computed=CURRENT_TIMESTAMP : timestamp
        """

        @property
        def key_source(self):
            return MatchIncludeMethod 

        def make(self, key):
            key['ts_computed'] = str(datetime.utcnow())#.strftime("%Y-%m-%d_%H:%M:%S")
            key[self.hash_name] = self.hash1(key)
            key = {**key, **MatchIncludeMethod.r1p(key).run(**key)}
            key = self.master.Store.put(key)
            self.master.Store.insert1(key, ignore_extra_fields=True, insert_to_master=True)
            self.insert1(key, ignore_extra_fields=True, skip_hashing=True, allow_direct_insert=True) 

    class Store(djp.Part):
        hash_name = 'match_table_id'
        definition = """
        -> master
        ---
        match_table : <match_tables> # match table
        ts_inserted=CURRENT_TIMESTAMP : timestamp
        """

        @classproperty
        def store_path(cls):
            return Path(config.externals['match_tables']['location'])
        
        @classmethod
        def get_file_path(cls, match_table_id, ext='.pkl'):
            return cls.store_path / f'{match_table_id}{ext}'
        
        def _get_restrict(self, key=None):
            key_source = self.master.Maker
            return self * (key_source & key) if key is not None else self * key_source
                
        def _get_fetch(self, rel):
            print(f'Fetching {len(rel)} row(s) from {self.class_name}.')
            return [FieldDict(**r) for r in rel.fetch(as_dict=True)]
        
        def get(self, key=None):
            assert not isinstance(self, dict), 'get must be called on an instance, not a class'
            rel = self._get_restrict(key)
            return self._get_fetch(rel)
        
        def get1(self, key=None):
            assert not isinstance(self, dict), 'get1 must be called on an instance, not a class'
            rel = self._get_restrict(key)
            len_rel = len(rel)
            assert len_rel == 1, f'Expected 1 row, got {len_rel}'
            return unwrap(self._get_fetch(rel))
        
        @classmethod
        def put(cls, key, ext='.pkl'):
            fp = cls.get_file_path(key[cls.hash_name], ext=ext)
            key['match_table'].to_pickle(fp)
            key['match_table'] = fp
            return key
    
    @classproperty
    def stores(cls):
        return [cls.Store]


