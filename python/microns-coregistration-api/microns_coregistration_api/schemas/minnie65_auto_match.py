"""
DataJoint tables for automatch.
"""
import datajoint as dj
import datajoint_plus as djp
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
import microns_utils.datajoint_utils as dju
from microns_utils.misc_utils import FieldDict, classproperty, wrap, unwrap
from ..methods import match
from tqdm import tqdm

from ..config import minnie65_auto_match_config as config

from ..schemas import minnie65_manual_match as m65mm

config.register_externals()
config.register_adapters(context=locals())

schema = djp.schema(config.schema_name, create_schema=True)

@schema
class Tag(dju.VersionLookup):
    package = 'microns-coregistration-api'
    attr_name = 'tag'


@schema
class UnitNucleusDistance(djp.Lookup):
    hash_name = 'und_id'
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
    
    class Imported(djp.Part):
        enable_hashing = True
        hash_name = 'und_id'
        hashed_attrs = 'name', 'source', Tag.attr_name
        definition = f"""
        -> master
        ---
        name : varchar(64) # name of data
        source : varchar(128) # source of the imported data
        description : varchar(1024) # description of the imported data
        -> Tag
        ts_inserted=CURRENT_TIMESTAMP : timestamp
        """

        @classmethod
        def fill(cls, und_table, name, source, description):
            key = {
                'name': name,
                'source': source,
                'description': description,
                Tag.attr_name : Tag.version
            }
            key[cls.hash_name] = cls.hash1(key)
            key['und_table'] = und_table
            key = cls.master.Store.put(key)
            with dj.conn().transaction:
                cls.master.Store.insert1(key, ignore_extra_fields=True, insert_to_master=True, skip_duplicates=True)
                cls.insert1(key, ignore_extra_fields=True, skip_hashing=True)


    class Store(djp.Part):
        hash_name = 'und_id'
        definition = """
        -> master
        ---
        und_table : <match_tables> # unit nucleus distance table
        """

        @classproperty
        def store_path(cls):
            return Path(config.externals['match_tables']['location'])
        
        @classmethod
        def get_file_path(cls, und_id, ext='.pkl'):
            return cls.store_path / f'{und_id}{ext}'
        
        def _get_restrict(self, key=None):
            key_source = self.master.Imported
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
            key['und_table'].to_pickle(fp)
            key['und_table'] = fp
            return key

    @classproperty
    def stores(cls):
        return [cls.Store]


@schema
class AutoMatchMethod(djp.Lookup):
    hash_name = 'auto_match_method'
    definition = f"""
    {hash_name} : varchar(8) # hash of the auto-match method
    """

    class Proximity(djp.Part):
        enable_hashing = True
        hash_name = 'auto_match_method'
        hashed_attrs = 'feature', 'threshold', Tag.attr_name
        definition = f"""
        -> master
        ---
        feature : varchar(128) # feature used for proximity
        threshold : float # threshold for proximity
        -> Tag
        ts_inserted=CURRENT_TIMESTAMP : timestamp
        """

        @classmethod
        def update_method(cls, feature, threshold):
            cls.insert1({
                'feature': feature,
                'threshold': threshold,
                Tag.attr_name : Tag.version
            }, insert_to_master=True)

        def run(self, und_df, **kwargs):
            params = self.fetch1()
            feature = params.get('feature')
            dmax = params.get('threshold')
            self.Log('info', 'Loading unit-nucleus distance data')
            self.Log('info', 'Running proximity matching algorithm per field.')
            dfs = []
            for (ss, si, f), group_df in tqdm(und_df.groupby(['scan_session', 'scan_idx', 'field'])):
                # Step 1: Identify unique unit_id and nuclei
                unit_labels = group_df.unit_id.unique()
                nuc_labels = group_df.nucleus_id.unique()
                
                # Step 2: Initialize cost matrix with large value to handle missing distances
                large_value = 1e6
                cost_matrix = np.full((len(unit_labels), len(nuc_labels)), large_value)
                
                # Step 3: Populate matrix with known distances
                unit_to_index = {unit: idx for idx, unit in enumerate(unit_labels)}
                nucleus_to_index = {nucleus_id: idx for idx, nucleus_id in enumerate(nuc_labels)}
                
                for _, row in group_df.iterrows():
                    unit_idx = unit_to_index[row['unit_id']]
                    nucleus_idx = nucleus_to_index[row['nucleus_id']]
                    cost_matrix[unit_idx, nucleus_idx] = row[feature]
                
                # Solve the linear sum assignment problem
                rows, cols = linear_sum_assignment(cost_matrix)
                
                # Filter matches by max_match_dist
                match_mask = cost_matrix[rows, cols] < dmax
                
                # Apply mask to select the matched labels/indices and residuals
                unit_ids = unit_labels[rows][match_mask]
                nucleus_ids = nuc_labels[cols][match_mask]
                residuals = cost_matrix[rows, cols][match_mask]
                
                df = pd.DataFrame([])
                df['scan_session'] = ss * np.ones(len(unit_ids), dtype=int)
                df['scan_idx'] = si * np.ones(len(unit_ids), dtype=int)
                df['field'] = f * np.ones(len(unit_ids), dtype=int)
                df['unit_id'] = unit_ids
                df['nucleus_id'] = nucleus_ids
                df['residual'] = residuals
                dfs.append(df)

            match_df = pd.concat(dfs, 0)
            
            # add separation to match_df
            self.Log('info', 'Adding separation to dataframe.')
            match_df = match.add_separation_to_match_df(match_df, und_df)

            return FieldDict(match_table=match_df)


@schema
class AutoMatchIncludeMethod(djp.Lookup):
    hash_name = 'auto_match_include_method'
    definition = f"""
    {hash_name} : varchar(6)
    """

    class V1(djp.Part):
        enable_hashing = True
        hash_name = 'auto_match_include_method'
        hashed_attrs = (
            'scan_include_set',
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
        scan_include_set=NULL   : varchar(6) # id of scan include set
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
            scan_include_set=None,
            min_residual=None,
            min_nuc_sep=None,
            min_unit_sep=None,
            max_residual=None,
            max_nuc_sep=None,
            max_unit_sep=None,
        ):
            key = dict(
                scan_include_set=scan_include_set if scan_include_set is not None else scan_include_set,
                min_residual=min_residual,
                min_nuc_sep=min_nuc_sep,
                min_unit_sep=min_unit_sep,
                max_residual=max_residual,
                max_nuc_sep=max_nuc_sep,
                max_unit_sep=max_unit_sep,
            )
            key[Tag.attr_name]= Tag.version
            cls.insert1(key, insert_to_master=True)
        
        def run(self, match_table, **kwargs):
            params = self.fetch1()
            assert params[Tag.attr_name] == Tag.version, 'Tag version mismatch'

            scan_include_set = params.get('scan_include_set')
            if scan_include_set is not None:
                scan_include_df = pd.DataFrame(m65mm.ScanInclude.r1pwh(scan_include_set).fetch('scan_session', 'scan_idx', as_dict=True))
            min_residual = params.get('min_residual')
            min_nuc_sep = params.get('min_nuc_sep')
            min_unit_sep = params.get('min_unit_sep')
            max_residual = params.get('max_residual')
            max_nuc_sep = params.get('max_nuc_sep')
            max_unit_sep = params.get('max_unit_sep')

            self.Log('info', 'applying filters')
            if scan_include_set is not None:
                match_table = match_table.merge(scan_include_df)
            match_table = match_table.query(f'residual>={min_residual}') if min_residual is not None else match_table
            match_table = match_table.query(f'residual<{max_residual}') if max_residual is not None else match_table
            match_table = match_table.query(f'nuc_sep>={min_nuc_sep}') if min_nuc_sep is not None else match_table
            match_table = match_table.query(f'nuc_sep<{max_nuc_sep}') if max_nuc_sep is not None else match_table
            match_table = match_table.query(f'unit_sep>={min_unit_sep}') if min_unit_sep is not None else match_table
            match_table = match_table.query(f'unit_sep<{max_unit_sep}') if max_unit_sep is not None else match_table
            
            match_table = match_table.sort_values(by=['scan_session', 'scan_idx', 'field', 'unit_id', 'nucleus_id'])

            return FieldDict(match_table=match_table)


@schema
class AutoMatch(djp.Lookup):
    hash_name = 'auto_match_id'
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
    
    class ProximityMaker(djp.Part, dj.Computed):
        enable_hashing = True
        hash_name = 'auto_match_id'
        hashed_attrs = 'auto_match_method', 'und_id'
        definition = """
        -> master
        -> AutoMatchMethod
        -> m65mm.UnitNucleusDistance
        """

        @property
        def key_source(self):
            return (AutoMatchMethod & (AutoMatchMethod.Proximity & Tag.get_latest_entries())) * m65mm.UnitNucleusDistance

        def make(self, key):
            key[self.hash_name] = self.hash1(key)
            key['und_df'] = pd.DataFrame((m65mm.UnitNucleusDistance.r1pwh(key['und_id'])).fetch())
            key = {**key, **AutoMatchMethod.r1p(key).run(**key)}
            key = self.master.Proximity.put(key)
            self.master.Proximity.insert1(key, ignore_extra_fields=True, insert_to_master=True)
            self.insert1(key, ignore_extra_fields=True, skip_hashing=True)


    class ProximityIncludeMaker(djp.Part, dj.Computed):
        enable_hashing = True
        hash_name = 'auto_match_id'
        hashed_attrs = 'auto_match_method', 'und_id', 'auto_match_include_method'
        definition = """
        -> master
        -> AutoMatchMethod
        -> m65mm.UnitNucleusDistance
        -> AutoMatchIncludeMethod
        """

        @property
        def key_source(self):
            return AutoMatchMethod * m65mm.UnitNucleusDistance * AutoMatchIncludeMethod

        def make(self, key):
            key[self.hash_name] = self.hash1(key)
            key['und_df'] = pd.DataFrame((m65mm.UnitNucleusDistance.r1pwh(key['und_id'])).fetch())
            key = {**key, **AutoMatchMethod.r1p(key).run(**key)}
            key = {**key, **AutoMatchIncludeMethod.r1p(key).run(**key)}
            key = self.master.ProximityInclude.put(key)
            self.master.ProximityInclude.insert1(key, ignore_extra_fields=True, insert_to_master=True)
            self.insert1(key, ignore_extra_fields=True, skip_hashing=True) 


    class ProximityInclude2Maker(djp.Part, dj.Computed):
        enable_hashing = True
        hash_name = 'auto_match_id'
        hashed_attrs = 'auto_match_method', 'und_id', 'auto_match_include_method'
        definition = """
        -> master
        -> AutoMatchMethod
        -> UnitNucleusDistance
        -> AutoMatchIncludeMethod
        """

        @property
        def key_source(self):
            return AutoMatchMethod * UnitNucleusDistance * AutoMatchIncludeMethod

        def make(self, key):
            key[self.hash_name] = self.hash1(key)
            key['und_df'] = UnitNucleusDistance.get1({'und_id': key['und_id']}).und_table
            key = {**key, **AutoMatchMethod.r1p(key).run(**key)}
            key = {**key, **AutoMatchIncludeMethod.r1p(key).run(**key)}
            key = self.master.ProximityInclude2.put(key)
            self.master.ProximityInclude2.insert1(key, ignore_extra_fields=True, insert_to_master=True)
            self.insert1(key, ignore_extra_fields=True, skip_hashing=True) 


    class Proximity(djp.Part):
        hash_name = 'auto_match_id'
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
        def get_file_path(cls, auto_match_id, ext='.pkl'):
            return cls.store_path / f'{auto_match_id}{ext}'
        
        def _get_restrict(self, key=None):
            key_source = self.master.ProximityMaker
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
            fp = cls.get_file_path(key['auto_match_id'], ext=ext)
            key['match_table'].to_pickle(fp)
            key['match_table'] = fp
            return key
    
    class ProximityInclude(djp.Part):
        hash_name = 'auto_match_id'
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
        def get_file_path(cls, auto_match_id, ext='.pkl'):
            return cls.store_path / f'{auto_match_id}{ext}'
        
        def _get_restrict(self, key=None):
            key_source = self.master.ProximityIncludeMaker
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

    class ProximityInclude2(djp.Part):
        hash_name = 'auto_match_id'
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
        def get_file_path(cls, auto_match_id, ext='.pkl'):
            return cls.store_path / f'{auto_match_id}{ext}'
        
        def _get_restrict(self, key=None):
            key_source = self.master.ProximityInclude2Maker
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
        return [cls.Proximity, cls.ProximityInclude, cls.ProximityInclude2]


@schema
class ManAutoMatchMethod(djp.Lookup):
    hash_name = 'man_auto_match_method'
    definition = f"""
    {hash_name} : varchar(6)
    """

    class ManualPriority(djp.Part):
        enable_hashing = True
        hash_name = 'man_auto_match_method'
        hashed_attrs = Tag.attr_name
        definition = """
        -> master
        ---
        -> Tag
        """
            
        @classmethod
        def update_method(cls):
            cls.insert1({Tag.attr_name: Tag.version}, insert_to_master=True)

        def run(self, manual_df, auto_df, **kwargs):
            params = self.fetch1()
            assert params[Tag.attr_name] == Tag.version, 'Tag version mismatch'

            man_auto_df = pd.concat([manual_df, auto_df])
            man_auto_df = man_auto_df.drop_duplicates(['scan_session', 'scan_idx', 'field', 'unit_id'], keep='first')
            man_auto_df = man_auto_df.drop_duplicates(['scan_session', 'scan_idx', 'field', 'nucleus_id'], keep='first')
            man_auto_df = man_auto_df.reset_index(drop=True)

            return FieldDict(match_table=man_auto_df)
    
    class DropDisagreements(djp.Part):
        enable_hashing = True
        hash_name = 'man_auto_match_method'
        hashed_attrs = Tag.attr_name
        definition = """
        -> master
        ---
        -> Tag
        """

        @classmethod
        def update_method(cls):
            cls.insert1({Tag.attr_name: Tag.version}, insert_to_master=True)
        
        def run(self, manual_df, auto_df, **kwargs):
            params = self.fetch1()
            assert params[Tag.attr_name] == Tag.version, 'Tag version mismatch'
            man_auto_df = pd.concat([manual_df, auto_df])
            man_auto_df = match.drop_match_disagreements(man_auto_df, on='unit_id')
            man_auto_df = match.drop_match_disagreements(man_auto_df, on='nucleus_id')
            man_auto_df = man_auto_df.drop_duplicates(['scan_session', 'scan_idx', 'field', 'unit_id', 'nucleus_id'], keep='first')
            man_auto_df = man_auto_df.reset_index(drop=True)

            return FieldDict(match_table=man_auto_df)
        

@schema
class ManAutoMatch(djp.Lookup):
    hash_name = 'man_auto_match_id'
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
        hash_name = 'man_auto_match_id'
        hashed_attrs = 'match_table_id', 'auto_match_id', 'man_auto_match_method', 'ts_computed'
        definition = """
        -> master
        -> m65mm.MatchTable
        -> AutoMatch
        -> ManAutoMatchMethod
        ts_computed=CURRENT_TIMESTAMP : timestamp
        """

        @property
        def key_source(self):
            return m65mm.MatchTable * AutoMatch * ManAutoMatchMethod 

        def make(self, key):
            key['ts_computed'] = str(datetime.utcnow())
            key[self.hash_name] = self.hash1(key)
            key['manual_df'] = m65mm.MatchTable.get1(key).match_table
            key['auto_df'] = AutoMatch.get1(key).match_table
            key = {**key, **ManAutoMatchMethod.r1p(key).run(**key)}
            key = self.master.Store.put(key)
            self.master.Store.insert1(key, ignore_extra_fields=True, insert_to_master=True)
            self.insert1(key, ignore_extra_fields=True, skip_hashing=True) 

    class Store(djp.Part):
        hash_name = 'man_auto_match_id'
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
        def get_file_path(cls, man_auto_match_id, ext='.pkl'):
            return cls.store_path / f'{man_auto_match_id}{ext}'
        
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