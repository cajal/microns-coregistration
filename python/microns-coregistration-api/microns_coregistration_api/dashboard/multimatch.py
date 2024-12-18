"""
MultMatch tool.
"""

import secrets
from collections import namedtuple
from pathlib import Path
from nglui import statebuilder

import datajoint_plus as djp
import numpy as np
import pandas as pd
from ipywidgets import link
import wridgets as wr
import wridgets.app as wra
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.patches import Circle

from microns_nda_api.schemas import \
    minnie_nda as nda, minnie_function as mf
from microns_materialization_api.schemas import \
    minnie65_materialization as m65mat
from microns_morphology_api.schemas import \
    minnie65_auto_proofreading as m65auto
import microns_utils.widget_utils as wu
import microns_dashboard_api as mdapi
from microns_dashboard_api.apps import AppLink

from ..schemas import minnie65_coregistration as m65crg, minnie_em as em
from ..methods import coregistration as crg
from ..schemas import multimatch as database

logger = djp.getLogger(__name__)

m65 = djp.create_djp_module(schema_name='microns_minnie65_02', load_dependencies=False)
stable = m65mat.Materialization & 'ver=343'

class DataType(mdapi.DataType):
    Field = wu.namedtuple_with_defaults(namedtuple('Field', ['scan_session', 'scan_idx', 'field']))
    Unit = wu.namedtuple_with_defaults(namedtuple('Unit', ['scan_session', 'scan_idx', 'field', 'unit_id']))
    Nucleus = wu.namedtuple_with_defaults(namedtuple('Nucleus', ['nucleus_id']))
    Match = wu.namedtuple_with_defaults(namedtuple('Match', ['scan_session', 'scan_idx', 'field', 'unit_id', 'nucleus_id', 'type']))


class Precomputed:
    scan_session_options = (djp.U('scan_session') & nda.Field).fetch('scan_session', order_by='scan_session').tolist()
    scan_idx_options = (djp.U('scan_idx') & nda.Field).fetch('scan_idx', order_by='scan_idx').tolist()
    field_options = (djp.U('field') & nda.Field).fetch('field', order_by='field').tolist()
    nucleus_id_options = m65mat.Nucleus.fetch('nucleus_id', order_by='nucleus_id').tolist()
    em_stack_info = (em.EM & 'em_name="minnie65"' & 'alignment=2').fetch1()
    oracle_rel = (mf.OracleScanSet() & {'oracle_scan_set_hash':'5801cc4145656ea3ca8cbe3f811b0a49'}).oracle()


class FieldUnitIDManager(wra.App):
    store_config = [
        ('unit_source', nda.UnitSource),
        ('scan_session_options', Precomputed.scan_session_options),
        ('scan_idx_options', Precomputed.scan_idx_options),
        ('field_options', Precomputed.field_options),
        ('unit_id_options', [])
    ]
    
    def make(self, on_select=None, on_select_kws=None, **kwargs):
        self.on_select = on_select
        self.on_select_kws = on_select_kws
        self.setdefault('width', 125)
        self.setdefault('height', 150)
        self.defaults.update(kwargs)
        
        shared_kws = dict(
            app1='Field', 
            app2='Select',
            orientation='vertical',
            rev_transform=lambda x: str(x)
        )
        scan_session_fwd_transform = lambda x: int(x) if x.isnumeric() and int(x) in self.scan_session_options else ''
        scan_idx_fwd_transform = lambda x: int(x) if x.isnumeric() and int(x) in self.scan_idx_options else ''
        field_fwd_transform = lambda x: int(x) if x.isnumeric() and int(x) in self.field_options else ''
        unit_id_fwd_transform = lambda x: int(x) if x.isnumeric() and int(x) in self.unit_id_options else ''
        
        for name, options, fwd, dis in zip(
            ['scan_session', 'scan_idx', 'field', 'unit_id'], 
            [self.scan_session_options, self.scan_idx_options, self.field_options, self.unit_id_options],
            [scan_session_fwd_transform, scan_idx_fwd_transform, field_fwd_transform, unit_id_fwd_transform],
            [True, True, True, False]
        ):
            app_kws = self.setdefault('_'.join([name, 'app_kws']), {})
            app_kws.update(shared_kws)
            app_kws.update(dict(
                fwd_transform=fwd,
                app1_kws=dict(
                    layout={'width': f'{self.getdefault("width")}px'}, 
                    disabled=dis
                ),
                app2_kws=dict(
                    options=[''] + options,
                    layout={'width': f'{self.getdefault("width")}px', 'height': f'{self.getdefault("height")}px'},
                    )
                )
            )

            setattr(self, '_' + name + '_app', wra.Label(text=name.title().replace('_', ' ')) - AppLink(**app_kws))
        
        self._load_units_button = wra.Button(description='Load Units', layout={'width': f'{self.getdefault("width")}px'}, on_interact=self.load_units, name='LoadUnitsButton')
        self._select_button = wra.Button(description='Select', layout={'width': f'{self.getdefault("width")}px'}, on_interact=self.on_select, on_interact_kws=self.on_select_kws, name='SelectButton')
        self._reset_button = wra.Button(description='Reset', layout={'width': f'{self.getdefault("width")}px'}, on_interact=self.reset, name='ResetButton')
        self.core = (
            (
                self._scan_session_app
            ) + \
            (
                self._scan_idx_app
            ) + \
            (
                self._field_app
            ) + \
            (
                (self._unit_id_app + (wra.Button(hide=True) - self._select_button - self._reset_button)) - self._load_units_button
            )
        )

    @property
    def scan_session(self):
        value = self._scan_session_app.ch.Field.get1('value')
        return value if value else None
    
    @property
    def scan_idx(self):
        value = self._scan_idx_app.ch.Field.get1('value')
        return value if value else None
    
    @property
    def field(self):
        value = self._field_app.ch.Field.get1('value')
        return value if value else None
    
    @property
    def unit_id(self):
        value = self._unit_id_app.ch.Field.get1('value')
        return value if value else None
    
    def load_units(self):
        unit_rel = self.unit_source & {'animal_id': 17797} & \
            [{'scan_session': self.scan_session} if self.scan_session is not None else {}] & \
            [{'scan_idx': self.scan_idx} if self.scan_idx is not None else {}] & \
            [{'field': self.field} if self.field is not None else {}]
        if len(unit_rel) != 0:
            self.units = list(set(unit_rel.fetch('unit_id')))
            self.update_unit_id_options(options=self.units)
        else:
            self.msg('No units. Check scan_session, scan_idx, field combination.')
    
    def update_unit_id_options(self, options=None):
        options = options if options is not None else []
        self.unit_id_options = options
        self._unit_id_app.ch.Select.set(options=[''] + options)
    
    def reset(self):
        self._scan_session_app.ch.Select.set(value='')
        self._scan_idx_app.ch.Select.set(value='')
        self._field_app.ch.Select.set(value='')
        self._unit_id_app.ch.Select.set(value='', options=[''])
        self.unit_id_options = []


class NucleusIDManager(wra.App):
    store_config = [
        ('nucleus_id_options', Precomputed.nucleus_id_options)
    ]
    def make(self, on_select=None, on_select_kws=None, **kwargs):
        self.on_select = on_select
        self.on_select_kws = on_select_kws
        self.setdefault('width', 125)
        self.setdefault('height', 150)
        self.defaults.update(kwargs)
        
        layout1 = {'width': f'{self.getdefault("width")}px'}
        layout2 = {'width': f'{self.getdefault("width")}px', 'height': f'{self.getdefault("height")}px'}
        
        self._label = wra.Label(text='Nucleus ID', prefix=self.name)
        self._field = wra.Field(on_interact=self.nucleus_search, layout=layout1, prefix=self.name)
        self._select = wra.Select(options=[''], layout=layout2, prefix=self.name)
        self._select_button = wra.Button(description='Select', layout=layout1, on_interact=self.on_select, on_interact_kws=self.on_select_kws, prefix=self.name)
        self._reset_button = wra.Button(description='Reset', layout=layout1, on_interact=self.reset, prefix=self.name)
        
        self.core = (
            (
                self._label - \
                self._field - \
                self._select
            ) + \
            (
                wra.Button(hide=True, name='SpacerButton') - \
                self._select_button - \
                self._reset_button
            )
        )
            
    @property
    def nucleus_id(self):
        value = self._select.get1('value')
        return value if value else None
    
    def nucleus_search(self):
        value = int(self._field.get1('value'))
        if value in self.nucleus_id_options:
            self._select.set(options=[''] + [value], value=value)

        else:
            self._field.set(on_interact_disabled=True, value='')
            self._select.set(options=[''], value='')
            self._field.set(on_interact_disabled=False)
            self.msg('Nucleus ID not in options')
    
    def reset(self):
        self._field.set(on_interact_disabled=True, value='')
        self._select.set(options=[''], value='')
        self._field.set(on_interact_disabled=False)


class UnitNucleusCentroidSearch(wra.App):
    store_config = [
        ('disable_fields', False)
    ]
    def make(self, **kwargs):
        self.defaults = kwargs
        self.ver = self.getdefault('ver', stable)
        self.em_stack_info = self.getdefault('em_stack_info', Precomputed.em_stack_info)
        self.nucleus_id_rel_um = self.getdefault('nucleus_rel_um', (m65mat.Nucleus.Info & self.ver & 'segment_id>0').proj(
                nucleus_x_um='nucleus_x*4/1000', 
                nucleus_y_um='nucleus_y*4/1000', 
                nucleus_z_um='nucleus_z*40/1000'
            )
        )
        self.nucleus_id_rel = m65mat.Nucleus.Info & self.ver & 'segment_id>0'
        self.unit_id_rel_default = nda.UnitSource * Precomputed.oracle_rel
        self.unit_id_rel = self.getdefault('unit_id_rel', self.unit_id_rel_default)
        self.distance_thresh_default = 15
        self.distance_thresh = self.getdefault('disance_thresh', self.distance_thresh_default)
        self.disable_fields = self.getdefault('disable_fields', self.disable_fields)
        
        # LAYOUT
        field_layout = {'width': '100px'}
        button_layout = {'width': '100px'}
        button_layout2 = {'width': '175px'} 
        select_layout = {'width': '375px'}
        header_layout = {'height':'28px', 'width':  '160px'}
        style = {'description_width': 'initial'}

        # OPTIONS
        self._distance_thresh_field = wra.BoundedIntText(
            name='DistanceThreshField',
            min=0, max=50, 
            value=self.distance_thresh, 
            layout=field_layout, 
            style=style, 
            disabled=self.disable_fields
        )
        self._unit_id_sort_dropdown = wra.Dropdown(
            name='UnitIDSortDropdown',
            options=['distance', 'oracle'], 
            value='distance', 
            layout=field_layout, 
            style=style, 
            disabled=self.disable_fields
        )
        self._clear_all_button = wra.Button(
            name='ClearAllButton',
            description='Clear All',
            on_interact=self.on_clear_all, 
            layout={'width': '100px', 'height': '124px'}
        )
        self._nucleus_id_field = wra.Field(
            name='NucleusIDField', 
            wridget_type='BoundedIntText',
            min=0,
            max=np.max(Precomputed.nucleus_id_options),
            on_interact=self.nucleus_id_search, 
            layout={'width': '100px'}
        )
        self._nucleus_id_search_button = wra.Button(
            name='NucleusIDSearchButton',
            description='Search',
            on_interact=self.nucleus_id_search, 
            layout=button_layout
        )
        self._nucleus_id_clear_button = wra.Button(
            name='NucleusIDClearButton',
            on_interact=self.on_nucleus_id_clear, 
            description='Clear', 
            layout=button_layout
        )
        self._em_centroid_x_field = wra.BoundedFloatText(
            name='EMCentroidXField',
            step=0.01, min=0, max=self.em_stack_info.get('max_pt_x'), 
            layout=field_layout, 
            value=self.getdefault('em_x', 0), 
            style=style, 
            disabled=self.disable_fields
        )
        self._em_centroid_y_field = wra.BoundedFloatText(
            name='EMCentroidYField',
            step=0.01, min=0, max=self.em_stack_info.get('max_pt_y'), 
            layout=field_layout, 
            value=self.getdefault('em_y', 0), 
            style=style, 
            disabled=self.disable_fields
        )
        self._em_centroid_z_field = wra.BoundedFloatText(
            name='EMCentroidZField',
            step=0.01, min=0, max=self.em_stack_info.get('max_pt_z'), 
            layout=field_layout, 
            value=self.getdefault('em_z', 0),
            style=style, 
            disabled=self.disable_fields
        )
        self._em_centroid_search_button = wra.Button(
            name='EMCentroidSearchButton',
            description='Search',
            on_interact=self.em_centroid_search, 
            layout=button_layout
        )
        self._em_centroid_clear_button = wra.Button(
            name='EMCentroidClearButton',
            description='Clear',
            on_interact=self.on_em_centroid_clear, 
            layout=button_layout
        )
        self._scan_session_field = wra.BoundedIntText(
            name='ScanSessionField',
            min=0,
            layout=field_layout, 
            style=style, 
            value=self.getdefault('scan_session', 0), 
            disabled=self.disable_fields
        )
        self._scan_idx_field = wra.BoundedIntText(
            name='ScanIdxField',
            min=0,
            layout=field_layout, 
            style=style, 
            value=self.getdefault('scan_idx', 0), 
            disabled=self.disable_fields
        )
        self._unit_id_field = wra.BoundedIntText(
            name='UnitIDField',
            min=0, max=100000,
            layout=field_layout, 
            style=style, 
            value=self.getdefault('unit_id', 0), 
            disabled=self.disable_fields
        )
        self._unit_id_search_button = wra.Button(
            name='UnitIDSearchButton',
            description='Search', 
            on_interact=self.unit_id_search, 
            layout=button_layout
        )
        self._unit_id_clear_button = wra.Button(
            name='UnitIDClearButton',
            on_interact=self.on_unit_id_clear, 
            description='Clear', 
            layout=button_layout
        )
        self._tp_centroid_x_field = wra.BoundedFloatText(
            name='TPCentroidXField',
            step=0.01, min=0, max=1412, 
            layout=field_layout, 
            value=self.getdefault('tp_x', 0), 
            style=style, disabled=self.disable_fields
        )
        self._tp_centroid_y_field = wra.BoundedFloatText(
            name='TPCentroidYField',
            step=0.01, min=0, max=1322, 
            layout=field_layout, value=self.getdefault('tp_y', 0), 
            style=style, 
            disabled=self.disable_fields
        ) 
        self._tp_centroid_z_field = wra.BoundedFloatText(
            name='TPCentroidZField',
            step=0.01, min=0, max=670, 
            layout=field_layout, value=self.getdefault('tp_z', 0), 
            style=style, 
            disabled=self.disable_fields
        )
        self._tp_centroid_search_button = wra.Button(
            name='TPCentroidSearchButton',
            on_interact=self.tp_centroid_search, 
            description='Search', 
            layout=button_layout
        )
        self._tp_centroid_clear_button = wra.Button(
            name='TPCentroidClearButton',
            on_interact=self.on_tp_centroid_clear, 
            description='Clear', 
            layout=button_layout
        )
        # SELECT
        self._field_nearest_select = wra.Select(
            name='FieldNearestSelect',
            layout={'width': '450px', 'height':'100px'}
        )
        self._unit_id_nearest_select = wra.Select(
            name='UnitIDNearestSelect',
            layout={'width': '600px', 'height':'100px'}
        )
        self._nucleus_id_nearest_select = wra.Select(
            name='NucleusIDNearestSelect',
            layout={'width': '400px', 'height':'100px'}
        )
        # CLEAR SELECT
        self._field_nearest_select_clear_button = wra.Button(
            name='FieldClearButton',
            description='Clear Fields',
            on_interact=self.on_field_select_clear, 
            layout=button_layout2
        )
        self._unit_id_select_clear_button = wra.Button(
            name='UnitIDClearButton',
            on_interact=self.on_unit_id_select_clear, 
            description='Clear Units', 
            layout=button_layout2
        )
        self._nucleus_id_select_clear_button = wra.Button(
            name='NucleusIDClearButton',
            on_interact=self.on_nucleus_id_select_clear, 
            description='Clear Nuclei',
            layout=button_layout2
        )
        # ON SELECT BUTTON
        self._field_select_button = wra.Button(
            name='FieldSelectButton',
            description=self.getdefault('field_select_button_description', 'Select'),
            on_interact=self.getdefault('on_field_select', None),
            on_interact_kws=self.getdefault('on_field_select_kws', {}),
            layout=button_layout2
        )
        self._unit_id_select_button = wra.Button(
            name='UnitIDSelectButton',
            description=self.getdefault('unit_id_select_button_description', 'Select'),
            on_interact=self.getdefault('on_unit_id_select', None),
            on_interact_kws=self.getdefault('on_unit_id_select_kws', {}),
            layout=button_layout2
        )
        self._nucleus_id_select_button = wra.Button(
            name='NucleusIDSelectButton',
            description=self.getdefault('nucleus_id_select_button_description', 'Select'),
            on_interact=self.getdefault('on_nucleus_id_select', None),
            on_interact_kws=self.getdefault('on_nucleus_id_select_kws', {}),
            layout=button_layout2
        )
        # LABELS
        self._field_nearest_label_default_value = '<font size="+1"> Nearest Fields </font>'
        self._field_nearest_label = wra.HTML(
            name='FieldNearestLabel',
            value=self._field_nearest_label_default_value
        )
        self._unit_id_nearest_label_default_value = '<font size="+1"> Nearest Units </font>'
        self._unit_id_nearest_label = wra.HTML(
            name='UnitIDNearestLabel',
            value=self._unit_id_nearest_label_default_value
        )
        self._nucleus_id_nearest_label_default_value = '<font size="+1"> Nearest Nuclei </font>'
        self._nucleus_id_nearest_label = wra.HTML(
            name='NucleusIDNearestLabel',
            value=self._nucleus_id_nearest_label_default_value
        )
        self._spacer_field = wra.BoundedFloatText(
            name='SpacerField',
            layout=field_layout, 
            hide=True
        )
        self.core = (
            (
                wra.Label(text="Options", layout=header_layout) + \
                wra.Label(text="max dist (\u03BCm)", fontsize=0.5, layout={'width': '100px'}) + \
                self._distance_thresh_field
            ) - \
            (
                wra.Label(layout=header_layout) + \
                wra.Label(text="sort by", fontsize=0.5, layout={'width': '100px'}) + \
                self._unit_id_sort_dropdown
            ) - \
            (
                wra.Box(layout={'height': '50px'})
            ) - \
            (
                (
                    (
                        wra.Label(text="Nucleus ID", layout=header_layout) + \
                        self._nucleus_id_field + \
                        self._spacer_field + \
                        self._spacer_field + \
                        self._nucleus_id_search_button + \
                        self._nucleus_id_clear_button
                    ) - \
                    (
                        wra.Label(text="EM Centroid (xyz)", layout=header_layout) + \
                        self._em_centroid_x_field + \
                        self._em_centroid_y_field + \
                        self._em_centroid_z_field + \
                        self._em_centroid_search_button + \
                        self._em_centroid_clear_button
                    ) - \
                    (
                        wra.Label(text="Session, Idx, Unit", layout=header_layout) + \
                        self._scan_session_field + \
                        self._scan_idx_field + \
                        self._unit_id_field + \
                        self._unit_id_search_button + \
                        self._unit_id_clear_button
                    ) - \
                    (
                        wra.Label(text="2P Centroid (xyz)", layout=header_layout) + \
                        self._tp_centroid_x_field + \
                        self._tp_centroid_y_field + \
                        self._tp_centroid_z_field + \
                        self._tp_centroid_search_button + \
                        self._tp_centroid_clear_button
                    )
                ) + self._clear_all_button
            ) - \
            (
                wra.Box(layout={'height': '25px'})
            ) - \
            (
                (
                    self._field_nearest_label - \
                    self._field_nearest_select - \
                    (
                        self._field_select_button + \
                        self._field_nearest_select_clear_button
                    )
                ) + \
                (
                    (
                        self._unit_id_nearest_label - \
                        self._unit_id_nearest_select - \
                        (
                            self._unit_id_select_button + \
                            self._unit_id_select_clear_button
                        )
                    )
                ) + \
                (
                    self._nucleus_id_nearest_label - \
                    self._nucleus_id_nearest_select - \
                    (
                        self._nucleus_id_select_button + \
                        self._nucleus_id_select_clear_button
                    )
                )
            )
        )

    def nucleus_id_search(self):
        nucleus_id = self._nucleus_id_field.get1('value')
        
        if nucleus_id==0:
            return
        
        try:
            nuc_x, nuc_y, nuc_z = (self.nucleus_id_rel & {'nucleus_id': nucleus_id}).fetch1('nucleus_x', 'nucleus_y', 'nucleus_z')
            
            self._em_centroid_x_field.set(value=nuc_x)
            self._em_centroid_y_field.set(value=nuc_y)
            self._em_centroid_z_field.set(value=nuc_z)
                
        except:
            self.msg('Nucleus ID not found in database.')
            return 
        
        self.em_centroid_search(nucleus_id=nucleus_id)
    
    def on_nucleus_id_clear(self):
        self._nucleus_id_field.set(value=0)
    
    def unit_id_search(self):
        scan_session = self._scan_session_field.get1('value')
        scan_idx = self._scan_idx_field.get1('value')
        unit_id = self._unit_id_field.get1('value')
        
        if scan_session==0 and scan_idx==0 and unit_id==0:
            return
        try:
            stack_x, stack_y, stack_z = (self.unit_id_rel & {'scan_session': scan_session, 'scan_idx': scan_idx, 'unit_id': unit_id}).fetch1('stack_x', 'stack_y', 'stack_z')
            
            self._tp_centroid_x_field.set(value=stack_x)
            self._tp_centroid_y_field.set(value=stack_y)
            self._tp_centroid_z_field.set(value=stack_z)
        
        except:
            self.msg('Unit not found in database.')
            return 
        
        self.tp_centroid_search()
    
    def on_unit_id_clear(self):
        self._scan_session_field.set(value=0)
        self._scan_idx_field.set(value=0)
        self._unit_id_field.set(value=0)
    
    def em_centroid_search(self, nucleus_id=None, em_x=None, em_y=None, em_z=None):
        em_x = self._em_centroid_x_field.get1('value') if em_x is None else em_x
        em_y = self._em_centroid_y_field.get1('value') if em_y is None else em_y
        em_z = self._em_centroid_z_field.get1('value') if em_z is None else em_z
        
        if em_x==0 and em_y==0 and em_z==0:
            return
        
        if nucleus_id is not None:
            em_x_tp, em_y_tp, em_z_tp = (m65.EMSegStack2PCoordsCentroids & 'resized_idx=3' & {'nucleus_id': nucleus_id}).fetch1('nucleus_x', 'nucleus_y', 'nucleus_z')
        
        else:
            em_x_tp, em_y_tp, em_z_tp = m65crg.Coregistration.run([[em_x, em_y, em_z]], transform_id=8)
        
        self._tp_centroid_x_field.set(value=em_x_tp.round(decimals=2))
        self._tp_centroid_y_field.set(value=em_y_tp.round(decimals=2))
        self._tp_centroid_z_field.set(value=em_z_tp.round(decimals=2))
        
        # Nuc Info        
        nearest_nuc_df = self.find_nearest_nucs(em_x, em_y, em_z)
        
        self._nucleus_id_nearest_label.make(value=f"<font size='+1'>Nearest Nuclei to: </font> <p> <font size='-1'>em_xyz=({em_x:.0f}, {em_y:.0f}, {em_z:.0f}):</font>")
        self._nucleus_id_nearest_select.set(options=[{'Nucleus': DataType.Nucleus(n), 'distance': np.round(d, 2)} for n, d in nearest_nuc_df.values])
        
        # Unit Info
        nearest_unit_df = self.find_nearest_units(em_x_tp, em_y_tp, em_z_tp)
        self._unit_id_nearest_label.make(value=f"<font size='+1'>Nearest Units to: </font> <p> <font size='-1'>2p_xyz=({em_x_tp:.0f}, {em_y_tp:.0f}, {em_z_tp:.0f}):</font>")
        self._unit_id_nearest_select.set(options=[{'Unit': DataType.Unit(s, i, f, u), 'distance': np.round(d, 2), 'oracle': np.round(o, 2)} for s, i, f, u, d, o in nearest_unit_df.values])
        
        # Field Info
        nearest_field_df = self.find_nearest_fields(em_x_tp, em_y_tp, em_z_tp)
        self._field_nearest_label.make(value=f"<font size='+1'>Nearest Fields to: </font> <p> <font size='-1'>2p_xyz=({em_x_tp:.0f}, {em_y_tp:.0f}, {em_z_tp:.0f}):</font>")
        self._field_nearest_select.set(options=[{'Field': DataType.Field(int(s), int(i), int(f)), 'distance': np.round(d, 2)} for s, i, f, _, d, _ in nearest_field_df.values])
        
    def on_em_centroid_clear(self):
        self._em_centroid_x_field.set(value=0)
        self._em_centroid_y_field.set(value=0)
        self._em_centroid_z_field.set(value=0)
    
    def tp_centroid_search(self, tp_x=None, tp_y=None, tp_z=None):
        tp_x = self._tp_centroid_x_field.get1('value') if tp_x is None else tp_x
        tp_y = self._tp_centroid_y_field.get1('value') if tp_y is None else tp_y
        tp_z = self._tp_centroid_z_field.get1('value') if tp_z is None else tp_z
        
        if tp_x==0 and tp_y==0 and tp_z==0:
            return
        
        tp_x_em, tp_y_em, tp_z_em = m65crg.Coregistration.run([[tp_x, tp_y, tp_z]], transform_id=7)
        
        self._em_centroid_x_field.set(value=tp_x_em.round(decimals=2))
        self._em_centroid_y_field.set(value=tp_y_em.round(decimals=2))
        self._em_centroid_z_field.set(value=tp_z_em.round(decimals=2))
        
        # Nuc Info
        nearest_nuc_df = self.find_nearest_nucs(tp_x_em, tp_y_em, tp_z_em)
        self._nucleus_id_nearest_label.make(value=f"<font size='+1'>Nearest Nuclei to: </font> <p> <font size='-1'>em_xyz=({tp_x_em:.0f}, {tp_y_em:.0f}, {tp_z_em:.0f}):</font>")
        self._nucleus_id_nearest_select.set(options=[{'Nucleus': DataType.Nucleus(n), 'distance': np.round(d, 2)} for n, d in nearest_nuc_df.values])
        
        # Unit Info
        nearest_unit_df = self.find_nearest_units(tp_x, tp_y, tp_z)
        nearest_field_df = self.find_nearest_fields(tp_x, tp_y, tp_z)
        
        self._unit_id_nearest_label.make(value= f"<font size='+1'>Nearest Units to: </font> <p> <font size='-1'>2p_xyz=({tp_x:.0f}, {tp_y:.0f}, {tp_z:.0f}):</font>")
        self._field_nearest_label.make(value=f"<font size='+1'>Nearest Fields to: </font> <p> <font size='-1'>2p_xyz=({tp_x:.0f}, {tp_y:.0f}, {tp_z:.0f}):</font>")
        
        self._unit_id_nearest_select.set(options=[{'Unit': DataType.Unit(s, i, f, u), 'distance': np.round(d, 2), 'oracle': np.round(o, 2)} for s, i, f, u, d, o in nearest_unit_df.values])
        self._field_nearest_select.set(options=[{'Field': DataType.Field(int(s), int(i), int(f)), 'distance': np.round(d, 2)} for s, i, f, _, d, _ in nearest_field_df.values])
        
    def find_nearest_units(self, tp_x, tp_y, tp_z):
        nearest_unit_id_rel = self.unit_id_rel.proj('field', 'oracle', distance=f'sqrt(power(stack_x-{tp_x},2) + power(stack_y-{tp_y},2) + power(stack_z-{tp_z},2))') & f'distance<={self._distance_thresh_field.get1("value")}'
        ascending = True if self._unit_id_sort_dropdown.get1('value') == 'distance' else False
        return pd.DataFrame(nearest_unit_id_rel.fetch())[['scan_session', 'scan_idx', 'field', 'unit_id', 'distance', 'oracle']].sort_values(self._unit_id_sort_dropdown.get1('value'), ascending=ascending).reset_index(drop=True)

    def find_nearest_fields(self, tp_x, tp_y, tp_z):
        nearest_unit_df = self.find_nearest_units(tp_x, tp_y, tp_z)
        nearest_unit_df['distance'] = pd.to_numeric(nearest_unit_df.distance)
        ascending = True if self._unit_id_sort_dropdown.get1('value') == 'distance' else False
        return nearest_unit_df.groupby(['scan_session', 'scan_idx', 'field'], as_index=False).min().sort_values(self._unit_id_sort_dropdown.get1('value'), ascending=ascending).reset_index(drop=True)
    
    def find_nearest_nucs(self, em_x, em_y, em_z):
        nearest_nuc_rel = self.nucleus_id_rel_um.proj(distance=f'sqrt(power(nucleus_x_um-{em_x*4/1000},2) + power(nucleus_y_um-{em_y*4/1000},2) + power(nucleus_z_um-{em_z*40/1000},2))') & f'distance<={self._distance_thresh_field.get1("value")}'
        return pd.DataFrame(nearest_nuc_rel.fetch())[['nucleus_id', 'distance']].sort_values('distance', ascending=True).reset_index(drop=True)
    
    def on_tp_centroid_clear(self):
        self._tp_centroid_x_field.set(value=0)
        self._tp_centroid_y_field.set(value=0)
        self._tp_centroid_z_field.set(value=0)
                                                
    def on_field_select_clear(self):
        self._field_nearest_select.set(options = [])
        self._field_nearest_label.make(value=self._field_nearest_label_default_value)
                                                
    def on_unit_id_select_clear(self):
        self._unit_id_nearest_select.set(options = [])
        self._unit_id_nearest_label.make(value=self._unit_id_nearest_label_default_value)
    
    def on_nucleus_id_select_clear(self):
        self._nucleus_id_nearest_select.set(options = [])
        self._nucleus_id_nearest_label.make(value=self._nucleus_id_nearest_label_default_value)

    def update_unit_id_rel(self, unit_id_rel=None, reset_to_default=False):
        if not reset_to_default:
            if unit_id_rel is not None:
                self.unit_id_rel = unit_id_rel
        else:
            self.unit_id_rel = self.unit_id_rel_default 
    
    def update_distance_thresh(self, distance_thresh=None, reset_to_default=False):
        if not reset_to_default:
            if distance_thresh is not None:
                self._distance_thresh = distance_thresh
                self._distance_thresh_field.set(value=self.distance_thresh)
        else:
            self._distance_thresh = self.distance_thresh_default
            self._distance_thresh_field.set(value=self.distance_thresh)

    def on_clear_all(self):
        self.on_nucleus_id_clear()
        self.on_unit_id_clear()
        self.on_em_centroid_clear()
        self.on_tp_centroid_clear()
        self.on_field_select_clear()
        self.on_unit_id_select_clear()
        self.on_nucleus_id_select_clear()


class Queue(wra.App):
    def make(self, actions:dict=None, actions_kws:dict=None, **kwargs):
        self.actions = self.setdefault('actions', {} if actions is None else actions)
        self.actions_kws = self.setdefault('actions_kws', {} if actions_kws is None else actions_kws)
        self.defaults.update(kwargs)
        
        queue_select_label_kws = self.setdefault('queue_select_label_kws', {})
        self._queue_select_label = wra.Label(
            prefix=self.name,
            name=queue_select_label_kws.setdefault('name', 'QueueSelectLabel'),
            text=queue_select_label_kws.setdefault('text', 'Queue'),
        )
        queue_select_kws = self.setdefault('queue_select_kws', {})
        self._queue_select = wra.Select(
            prefix=self.name,
            name=queue_select_kws.setdefault('name', 'QueueSelect'),
            wridget_type='Select' if not self.setdefault('select_multiple', False) else 'SelectMultiple',
            layout=queue_select_kws.setdefault(
                'layout', {
                    'width': queue_select_kws.setdefault('width', '450px'), 
                    'height': queue_select_kws.setdefault('height', '100px'), 
                }
            )            
        )
        action_select_label_kws = self.setdefault('action_select_label_kws', {})
        self._action_select_label = wra.Label(
            prefix=self.name,
            name=action_select_label_kws.setdefault('name', 'ActionSelectLabel'),
            text=action_select_label_kws.setdefault('text', 'Actions'),
            
        )
        action_select_kws = self.setdefault('action_select_kws', {})
        self._action_select = wra.Select(
            prefix=self.name,
            name=action_select_kws.setdefault('name', 'ActionSelect'),
            layout=action_select_kws.setdefault(
                'layout', {
                    'width': action_select_kws.setdefault('width', '450px'), 
                    'height': action_select_kws.setdefault('height', '75px'), 
                }
            ),
            options=[(fxn_name, fxn) for fxn_name, fxn in self.actions.items()]
        )
        action_run_button_kws = self.setdefault('action_run_button_kws', {})
        self._action_run_button = wra.Button(
            prefix=self.name,
            name=action_run_button_kws.setdefault('name', 'ActionRunButton'),
            description=action_run_button_kws.setdefault('description', 'Run'),
            on_interact=self.on_action_run_button_click,
        )
        queue_remove_entry_button_kws = self.setdefault('queue_remove_entry_button_kws', {})
        self._queue_remove_entry_button = wra.Button(
            prefix=self.name,
            name=queue_remove_entry_button_kws.setdefault('name', 'QueueRemoveEntryButton'),
            description=queue_remove_entry_button_kws.setdefault('description', 'Remove Entry'),
            on_interact=self.remove_entry_from_queue
        )
        queue_clear_button_kws = self.setdefault('queue_clear_button_kws', {})
        self._queue_clear_button = wra.Button(
            prefix=self.name,
            name=queue_clear_button_kws.setdefault('name', 'QueueClearButton'),
            description=queue_clear_button_kws.setdefault('description', 'Clear'),
            on_interact=self.clear_queue
        )
        
        self.core = (
            (
                self._queue_select_label - \
                self._queue_select
            ) - \
            (
                self._action_select_label - \
                self._action_select - \
                (
                    self._action_run_button + self._queue_remove_entry_button + self._queue_clear_button
                )
            )
        )
    
    def select_entry_in_queue(self, name, verbose=True):
        for n, v in self._queue_select.get1('options'):
            if name == n:
                self._queue_select.set(label=(name,))
                return True
        if verbose:
            self.msg('Entry not found.')
    
    def add_entry_to_queue(self, name, value=None, verbose=True):
        if not self.select_entry_in_queue(name, verbose=False):
            entry = (name, name) if value is None else (name, value)
            self._queue_select.set(options=list(self._queue_select.get1('options')) + [entry])
            self._queue_select.set(value=(entry[1],))
        else:
            if verbose:
                self.msg('Entry already present.')
    
    def remove_entry_from_queue(self):
        for value in self._queue_select.get1('value'):
            self._queue_select.set(options=[(n, v) for n, v in self._queue_select.get1('options') if v != value])
    
    def clear_queue(self):
        self._queue_select.set(options=())
    
    def on_action_run_button_click(self):
        if self.actions:
            fxn_name = self._action_select.get1('label')
            fxn = self._action_select.get1('value')
            fxn_kws = self.actions_kws.get(fxn_name)
            if fxn_kws is not None:
                fxn(**fxn_kws)
            else:
                fxn()    


class FieldUnitObject:
    def __init__(self, key=None, scan_session=None, scan_idx=None, field=None, unit_id=None):
        if key is not None:
            if isinstance(key, type(DataType.Field())):
                self.field_key = key
                self.unit_key = DataType.Unit(*self.field_key._asdict())
                
            elif isinstance(key, type(DataType.Unit())):
                self.field_key = DataType.Field(key.scan_session, key.scan_idx, key.field)
                self.unit_key = key

            else:
                raise Exception(f'Key must be of type: {type(DataType.Field())} or {type(DataType.Unit())}')
                        
        else:
            assert scan_session is not None and scan_idx is not None and field is not None, 'Provide key or scan_session, scan_idx, field'
            
            self.field_key = DataType.Field(scan_session, scan_idx, field)
            self.unit_key = DataType.Unit(scan_session, scan_idx, field)
            
            if unit_id is not None:
                self.unit_key = DataType.Unit(scan_session, scan_idx, field, unit_id)
        
        # FIELD UNIT INFO
        self.unit_mask_centroids = np.stack((m65.UnitInfo() & self.field_key._asdict()).fetch('mask_px_x', 'mask_px_y'), axis=1)
        self.unit_stack_centroids = np.stack((m65.UnitStackInfo() & self.field_key._asdict()).fetch('unit_x', 'unit_y', 'unit_z', order_by='unit_id'), -1)
        self.unit_labels = np.stack((m65.UnitStackInfo() & self.field_key._asdict()).fetch('unit_id', order_by='unit_id'), -1)
        self.field_stack_mean_depth = self.unit_stack_centroids.mean(axis=0)[-1].round(decimals=2)
        self.field_image = (m65.SummaryImagesEnhancedResizeStack2D() & self.field_key._asdict()).fetch1('average_corr')
        self.scan_vess = (m65.Stack2PResizedFieldStack2D & self.field_key._asdict() & {'resized_name':'vessel_stack'}).fetch1('stack_field')
        self.scan_nuc_seg = (m65.EMSegStack2PCoordsFieldStack2D() & self.field_key._asdict() & 'resized_idx=3').fetch1('stack_field')
        
        # SINGLE UNIT INFO
        if self.unit_key.unit_id is not None:
            self.unit_mask_centroid = np.stack((m65.UnitInfo() & self.unit_key._asdict()).fetch1('mask_px_x', 'mask_px_y'))
            self.unit_stack_centroid = np.stack((m65.UnitStackInfo() & self.unit_key._asdict()).fetch1('unit_x', 'unit_y', 'unit_z'), -1)
            self.unit_label = (m65.UnitStackInfo() & self.unit_key._asdict()).fetch1('unit_id')
            self.unit_stack_depth = self.unit_stack_centroid[-1]
            self.match_rel = m65.UnitEMCentroidMatch & self.unit_key._asdict()
        else:
            self.unit_mask_centroid = None
            self.unit_stack_centroid = None
            self.unit_stack_centroid = None
            self.unit_label = None
            self.unit_stack_depth = None
            self.match_rel = None

    def clear_unit_only(self):
        if self.unit_key.unit_id is not None:
            self.unit_key = DataType.Unit(**self.field_key._asdict())
            self.unit_mask_centroid = None
            self.unit_stack_centroid = None
            self.unit_stack_centroid = None
            self.unit_label = None
            self.unit_stack_depth = None
            self.match_rel = None

    
class NucleusObject:
    def __init__(self, nucleus=None):
        if isinstance(nucleus, type(DataType.Nucleus())):
            self.nucleus_key = nucleus
        
        else:
            self.nucleus_key = DataType.Nucleus(nucleus)

        if self.nucleus_key.nucleus_id is not None:
            # NUCLEUS INFO
            self.em_x, self.em_y, self.em_z = (m65mat.Nucleus.Info & stable & self.nucleus_key._asdict()).fetch1('nucleus_x', 'nucleus_y', 'nucleus_z')

            self.em_x_tp, self.em_y_tp, self.em_z_tp = (m65.EMSegStack2PCoordsCentroids & 'resized_idx=1' & self.nucleus_key._asdict()).fetch1('nucleus_x', 'nucleus_y', 'nucleus_z')
            
            if self.em_x_tp > 0 and self.em_y_tp > 0 and self.em_z_tp > 0:
                pass
            else:
                self.em_x_tp, self.em_y_tp, self.em_z_tp = m65crg.Coregistration.run([self.em_x, self.em_y, self.em_z], transform_id=6)

            self.match_rel = m65.UnitEMCentroidMatch & self.nucleus_key._asdict()


class MatchObject:
    def __init__(self, field_unit_object=None, nucleus_object=None, submission_type=None):
        self.field_unit_object = field_unit_object
        self.nucleus_object = nucleus_object
        self.submission_type = submission_type

        # UNPACK VALUES

        if self.field_unit_object is not None:
            if self.nucleus_object is not None:
                self.key = DataType.Match(**self.field_unit_object.unit_key._asdict(), **self.nucleus_object.nucleus_key._asdict(), type=submission_type)
            else:
                self.key = DataType.Match(**self.field_unit_object.unit_key._asdict(), type=submission_type)
        else:
            if self.nucleus_object is not None:
                self.key = DataType.Match(**self.nucleus_object.nucleus_key._asdict(), type=submission_type)
            else:
                self.key = DataType.Match()


class MultiMatch:
    # TODO
        # fix issue with cache images (checkbox is currently hidden)
        # remove inhibitory neurons based on Brendan's prediction
    def make(self, **kwargs):     
        self.defaults.update(kwargs)
        
        # Initialize 
        plt.close('all')
        
        self._multimatch_params = dict(
            load_source=self.default_values('load_source', 'numpy'),
            load_mode=self.default_values('load_mode', 'view'),
            transform_id=self.default_values('transform_id', 7)
        )

        self._figure = None
        
        self._plot_functions = [self.plot_scan_field_and_vessels, self.plot_nuc_seg, self.plot_em_and_vessels]

        self._fig_kws = dict(
            sharex=self.default_values('sharex', True),
            sharey=self.default_values('sharey', True),
            dpi=self.default_values('dpi', 150),
            figsize=(self.default_values('fig_width', 4*len(self._plot_functions)),
                     self.default_values('fig_height', 4*len(self._plot_functions)/3),
            )
        )
        
        self._plot_kws = dict(
            depth=self.default_values('depth', 150),
            xlim=self.default_values('xlim', (0, 1412)),
            ylim=self.default_values('ylim', (0, 1322))
        )
        
        self._images = dict(
            scan = np.zeros((1322, 1412)),
            scan_vess = np.zeros((1322, 1412)),
            scan_nuc_seg = np.zeros((1322, 1412)),
            vess = np.zeros((1322, 1412)),
            prob = np.zeros((1322, 1412)),
            nuc_seg = np.zeros((1322, 1412)),
            neuron_nuc_seg = np.zeros((1322, 1412)),
            em = np.zeros((1322, 1412)),
            m35em = np.zeros((1322, 1412)),
            custom = np.zeros((1322, 1412))
        )

        self._nucleus_object = None        
        self._field_unit_object = None
        self._selected_nucleus = None
        self._selected_unit = None
        self._current_centroids = None
        self._current_labels = None
        self._current_nuc_image = None
        self._selected_match_protocol = None
        self.selected_match_protocol_header = wr.HTML(value="<font size='+1'>None Selected</font>")

        # Widget layout and style
        self.style = {'description_width': 'initial'}
        self.layout = {'width': '125px'}
        self.select_layout = {'width': '450px', 'height': '50px'}
        self.select_layout2= {'width': '750px', 'height': '50px'}
        self.button_layout = {'width': '175px'}
        self.button_layout2 = {'width': '200', 'height':'75px'}
        self.button_layout2_slim = {'width': '200', 'height':'auto'}

        self.button_layout3 = {'width': '65px'}

        # Match protocol buttons
        self.select_random_candidate_button = wr.Button(description="Load Random", on_interact=self._select_random_match_candidate, layout={'width': 'auto'})
        self.clear_selected_match_protocol_button = wr.Button(description="Clear", on_interact=self.clear_selected_match_protocol, layout={'width': 'auto'})
        
        # Figure fields and modifiers
        self.fig_width_field = wr.IntText(description='fig width: ', value=self.fig_kws['figsize'][0], on_interact=self.update_fig_kws, interact_kws=dict(from_ui=True), style=self.style, layout={'width': '125px'})
        self.fig_height_field = wr.IntText(description='fig height: ', value=self.fig_kws['figsize'][1], on_interact=self.update_fig_kws, interact_kws=dict(from_ui=True),  style=self.style, layout={'width': '125px'})
        
        self.dpi_field = wr.IntText(description='dpi: ', step=10, value=self.fig_kws['dpi'], on_interact=self.update_fig_kws, interact_kws=dict(from_ui=True),  style=self.style, layout={'width': '100px'})
        
        # Plot fields and modifiers
        self.replot_button = wr.Button(description='Replot', on_interact=self.update_plot)
        self.depth_field = wr.BoundedIntText(description='depth: ', value=self.plot_kws['depth'], on_interact=self.update_plot_kws, interact_kws=dict(from_ui=True), min=0, max=669, style=self.style, layout={'width': '100px'})
        self.xlim_slider = wr.IntRangeSlider(description='xlim: ', value=self.plot_kws['xlim'], on_interact=self.update_plot_kws, interact_kws=dict(from_ui=True), min=0, max=1412, step=50, style=self.style, layout={'width': '250px', 'display': 'none'})
        self.ylim_slider = wr.IntRangeSlider(description='ylim: ', value=self.plot_kws['ylim'], on_interact=self.update_plot_kws, interact_kws=dict(from_ui=True), min=0, max=1322, step=50, style=self.style, layout={'width': '250px', 'display': 'none'})
        self.max_lim_button = wr.Button(description='max lim', on_interact=self.update_plot_kws, interact_kws=dict(xlim=(0,1422), ylim=(0,1322)), layout={'width':'100px'})
        self.zoom_to_unit_check = wr.Checkbox(description='zoom to unit', value=True, style=self.style, indent=False, on_interact=self.update_plot)
        self.zoom_to_nucleus_check = wr.Checkbox(description='zoom to nucleus', value=True, style=self.style, indent=False, on_interact=self.update_plot)
        self.circle_unit_check = wr.Checkbox(description='circle unit', value=False, style=self.style, indent=False, on_interact=self.update_plot)
        self.unit_scatter_check = wr.Checkbox(description='unit centroids', value=True, style=self.style, indent=False, on_interact=self.update_plot)
        self.filter_non_neurons_check = wr.Checkbox(description='remove non-neurons', on_interact=self.update_plot, value=True, style=self.style, indent=False)
        self.cache_images_check = wr.Checkbox(description='cache images', value=True if self.multimatch_params['load_mode'] == 'cache' else False, on_interact=self.update_multimatch_params, on_interact_kws=dict(from_ui=True), layout={'display':'none'})
        self.vessels_check = wr.Checkbox(description="vessels", value=True, on_true=self.update_plot)

        self.user_id_field = wr.Text(description='User ID', layout={'width': '260px'}, value=self.default_values('user_id', ''), disabled=True)

        #FIELD/UNIT
        self.field_key_select = wr.Select(layout={'width': '315px', 'height': '50px'}, style=self.style, disabled=True)
        self.primary_unit_field = WidgetTemplates.scan_unit_field(layout=self.layout, style=self.style,disabled=True)
        self.selected_unit_field = WidgetTemplates.scan_unit_field(on_interact=self.update_selected_unit, interact_kws=dict(from_ui=True),layout=self.layout, style=self.style, disabled=False) 
        self.clear_field_unit_object_button = wr.Button(description='Clear Field/ Unit', layout={'width': 'auto'}, on_interact=self.clear_field_unit_object)
        self.clear_primary_unit_button = wr.Button(description='Clear', layout={'width': 'auto'}, on_interact=self.clear_primary_unit)
        self.clear_selected_unit_button = wr.Button(description='Clear', layout={'width': 'auto'}, on_interact=self.clear_selected_unit)
        self.primary_unit_center_button = wr.Button(description='Center', layout={'width': 'auto'}, on_interact=self.center_plot, interact_kws=dict(primary_unit=True))
        self.selected_unit_center_button = wr.Button(description='Center', layout={'width': 'auto'}, on_interact=self.center_plot, interact_kws=dict(selected_unit=True)) 

        # NUCLEUS
        self.primary_nucleus_field = WidgetTemplates.nucleus_id_field(layout=self.layout, style=self.style, disabled=True)
        self.selected_nucleus_field = WidgetTemplates.nucleus_id_field(on_interact=self.update_selected_nucleus, interact_kws=dict(from_ui=True), layout=self.layout, style=self.style, disabled=False)
        self.clear_nucleus_object_button = wr.Button(description='Clear', layout={'width': 'auto'}, on_interact=self.clear_nucleus_object)
        self.clear_selected_nucleus_button = wr.Button(description='Clear',layout={'width': 'auto'}, on_interact=self.clear_selected_nucleus)
        self.primary_nucleus_center_button = wr.Button(description='Center', layout={'width': 'auto'}, on_interact=self.center_plot, interact_kws=dict(primary_nucleus=True))
        self.selected_nucleus_center_button = wr.Button(description='Center', layout={'width': 'auto'}, on_interact=self.center_plot, interact_kws=dict(selected_nucleus=True))

        # MATCHES
        self.match_selected_button = wr.Button(description ='Match Selected', layout=self.button_layout2, on_interact=self.update_match, interact_kws=dict(submission_type='match'))
        self.match_uncertain_button = wr.Button(description ='Match Uncertain', layout=self.button_layout2_slim, on_interact=self.update_match, interact_kws=dict(submission_type='match_uncertain'))
        self.unit_no_match_selected_button = wr.Button(description='Unit No Match', layout=self.button_layout2, on_interact=self.update_match, interact_kws=dict(submission_type='unit_no_match'))
        self.unit_indiscernable_button = wr.Button(description='Unit Indiscernable', layout=self.button_layout2_slim, on_interact=self.update_match, interact_kws=dict(submission_type='unit_indiscernable'))
        self.unit_skip_button = wr.Button(description='Unit Skip', layout=self.button_layout2_slim, on_interact=self.update_match, interact_kws=dict(submission_type='unit_skip'))
        self.nucleus_no_match_selected_button = wr.Button(description='Nucleus No Match', layout=self.button_layout2, on_interact=self.update_match, interact_kws=dict(submission_type='nucleus_no_match'))
        self.nucleus_inhibitory_button = wr.Button(description='Nucleus Inhibitory', layout=self.button_layout2_slim, on_interact=self.update_match, interact_kws=dict(submission_type='nucleus_inhibitory'))
        # self.nucleus_non_neuron_button = wr.Button(description='Nucleus Non Neuron', layout=self.button_layout2_slim, on_interact=self.update_match, interact_kws=dict(submission_type=True))
        self.nucleus_skip_button = wr.Button(description='Nucleus Skip', layout=self.button_layout2_slim, on_interact=self.update_match, interact_kws=dict(submission_type='nucleus_skip'))
        self.check_neuroglancer_button = wr.Button(on_interact=self.generate_ng_link, description='Check Neuroglancer', layout=self.button_layout2_slim)

        self.primary_match_select = wr.Select(options=[], layout=self.select_layout2)
        self.primary_match_button = wr.Button(description='Submit', layout={'width': 'auto'}, on_interact=self.submit_match, interact_kws=dict(primary=True))
        self.primary_match_center_unit_button = wr.Button(description='Center Unit', layout={'width': 'auto'}, on_interact=self.center_plot, interact_kws=dict(primary_match_unit=True))
        self.primary_match_center_nucleus_button = wr.Button(description='Center Nucleus', layout={'width': 'auto'}, on_interact=self.center_plot, interact_kws=dict(primary_match_nucleus=True))
        self.clear_primary_match_select_button = wr.Button(description='Clear', layout={'width': 'auto'}, on_interact=self.clear_match_select, interact_kws=dict(primary=True))
        
        self.secondary_match_select = wr.Select(options=[], layout={'width': self.select_layout2['width'], 'height':'125px'})
        self.secondary_match_button = wr.Button(description='Submit', layout={'width': 'auto'}, on_interact=self.submit_match, interact_kws=dict(secondary=True))
        # self.secondary_match_all_button = wr.Button(description='Submit All Secondary', layout={'width': 'auto'}, on_interact=self.submit_match, interact_kws=dict(secondary=True, submit_all=True))
        self.secondary_center_unit_button = wr.Button(description='Center Unit', layout={'width': 'auto'}, on_interact=self.center_plot, interact_kws=dict(secondary_match_unit=True))
        self.secondary_center_nucleus_button = wr.Button(description='Center Nucleus', layout={'width': 'auto'}, on_interact=self.center_plot, interact_kws=dict(secondary_match_nucleus=True))
        self.clear_secondary_match_select_button = wr.Button(description='Clear', layout={'width': 'auto'}, on_interact=self.clear_match_select, interact_kws=dict(secondary=True))
        # self.clear_all_secondary_match_select_button = wr.Button(description='Clear All Secondary', layout={'width': 'auto'}, on_interact=self.clear_match_select, interact_kws=dict(secondary=True, secondary_clear_all=True))
        
        self.submit_all_primary_secondary_button = wr.Button(description='Submit All', layout={'width': 'auto', 'height': '121px'}, on_interact=self.submit_match, interact_kws=dict(submit_all=True))
        self.clear_all_primary_secondary_button = wr.Button(description='Clear All', layout={'width': 'auto', 'height': '121px'}, on_interact=self.clear_match_select, interact_kws=dict(clear_all=True))

        self.unit_manual_entry_field = WidgetTemplates.scan_unit_field(layout=self.layout, style=self.style,disabled=True)
        self.nucleus_manual_entry_field = WidgetTemplates.scan_unit_field(layout=self.layout, style=self.style,disabled=True)

        self.match_note_field = wr.Textarea(layout={'width': '500px', 'height':'50px'})
        self.clear_match_note_button = wr.Button(description='Clear Note', layout={'width': 'auto'}, on_interact=self.clear_match_note)
        
        self.reset_multimatch_button = wr.Button(description="Reset MultiMatch", layout={'width': 'auto'}, on_interact=self.reset_multimatch)

        # ng links
        self.ng_link_html = wr.HTML(on_interact=self.update_ng_link)
        self.clear_ng_html_button = wr.Button(description='Clear NG Link', on_interact=self.update_ng_link, layout={'width': 'auto', 'display':'none'}, interact_kws=dict(clear=True))

        # Instantiate modules
        self.field_unit_selector_module = FieldUnitSelector(display=False, function_mapping={'Add to Unit Queue': self.add_to_unit_queue})
        
        self.nucleus_selector_module = NucleusSelector(display=False, function_mapping={'Add to Nucleus Queue': self.add_to_nucleus_queue})
        
        self.unit_match_candidate_module = ProtocolManager(source=m65.UnitMatchProtocol, header='Unit Protocols', on_select=self.on_match_protocol_select, on_select_kws={'protocol_type': 'Unit'})

        self.nucleus_match_candidate_module = ProtocolManager(source=m65.NucleusMatchProtocol, header='Nucleus Protocols', on_select=self.on_match_protocol_select, on_select_kws={'protocol_type': 'Nucleus'})

        self.unit_queue_module = Queue(display=False, header='Unit Queue:', multiselect=False, function_select_height='115px', function_mapping={
            'Search Nearby': self.search_nearby_unit, 
            'Send to MultiMatch': self.send_field_unit_to_multimatch
        })
        
        self.unit_queue_module.action_button.interact_kws = dict(from_queue_module=True)
        
        self.nucleus_queue_module = Queue(display=False, header='Nucleus Queue:', multiselect=False, function_select_height='115px', function_mapping={
                'Search Nearby': self.search_nearby_nucleus, 
                'Send to MultiMatch': self.send_nucleus_to_multimatch,
                'Check Morphology': self.check_morphology,
                'Submit Inhibitory': self.custom_msg_before_action,
                'Submit Nucleus No Match': self.custom_msg_before_action,
                'Submit Nucleus Non Neuron': self.custom_msg_before_action
            },
            function_mapping_kwargs={
                'Submit Inhibitory': {'msg': 'Are you sure you would like to submit?', 'output':'self', 'action': self.submit_nucleus_inhibitory, 'interact_kws': dict(from_queue_module=True), 'action_button_description': 'Submit'},
                'Submit Nucleus No Match': {'msg': 'Are you sure you would like to submit?', 'output':'self', 'action': self.submit_nucleus_no_match, 'interact_kws': dict(from_queue_module=True), 'action_button_description': 'Submit'},
                'Submit Nucleus Non Neuron': {'msg': 'Are you sure you would like to submit?', 'output':'self', 'action': self.submit_nucleus_non_neuron, 'interact_kws': dict(from_queue_module=True), 'action_button_description': 'Submit'},
            }
        )
        
        self.nucleus_queue_module.action_button.interact_kws = dict(from_queue_module=True)
        
        self.unit_nucleus_search_module = UnitNucleusCentroidSearch(
            display=False, 
            field_action_button_description='Send to MultiMatch',
            field_action_function=self.send_field_unit_to_multimatch,
            unit_action_button_description='Send to MultiMatch', 
            unit_action_function=self.send_field_unit_to_multimatch,
            nucleus_action_button_description='Send to MultiMatch', 
            nucleus_action_function=self.send_nucleus_to_multimatch,
            disable_fields=True
        )

        self.unit_nucleus_search_module.field_action_button.interact_kws = dict(from_search_module=True, field_only=True)
        self.unit_nucleus_search_module.unit_action_button.interact_kws = dict(from_search_module=True)
        self.unit_nucleus_search_module.nucleus_action_button.interact_kws = dict(from_search_module=True)
        

        # Outputs
        self.feedback_out = wr.Output()
        self.clear_feedback_button = wr.Button(on_interact=self.feedback_out.clear_output, description='Clear', button_style='info', layout={'width': '70px'})

        self.match_protocol_feedback_out = wr.Output()
        self.clear_match_protocol_feedback_button = wr.Button(on_interact=self.match_protocol_feedback_out.clear_output, description='Clear', button_style='info', layout={'width': '70px'})

        self.plot_output = wr.Output()
        self.slack_output = wu.SlackForWidget(default_channel='#platinum_tracer_notifications')

        self.submit_lag_report_button = wr.Button(description='Report Lag', on_interact=self.submit_report, interact_kws=dict(lag=True), layout=dict(width='auto'))

        if self.user_id_field.widget.value != '':
            try:
                self.slack_username = (db.Username.Slack & {'user': self.user_id_field.widget.value}).fetch1('slack_username')
            except:
                self.slack_username = None
        else:
            self.slack_username = None

        self.initialize_stack_loaders()

        self.generate_module()

        if not display:
            self.module.layout.display = 'none'
            wr.display(self.module)
        else:
            wr.display(self.module)
            # self.update_figure()
        
    def generate_module(self):
        # INFO MODULE
        
        self.info_module = wr.VBox([
            wr.HBox([
                wr.HTML(value=f"<font size='+1'>Field: </font>", layout={'height': '50px', 'width': '70px'}).widget, 
                self.field_key_select.widget,
                self.clear_field_unit_object_button.widget
            ]),
            
            wr.HBox([            
                wr.VBox([
                    wr.HTML(value=f"<font size='+1'>Unit: </font>").widget,
                    wr.HTML(value=f"Reference: ").widget, 
                    wr.Label(),
                    wr.HTML(value=f"Selected: ").widget, 
                    wr.Label()
                ]),
                wr.VBox([
                    wr.Label(),
                    self.primary_unit_field.widget,
                    wr.HBox([self.primary_unit_center_button.widget, self.clear_primary_unit_button.widget]),
                    self.selected_unit_field.widget, 
                    wr.HBox([self.selected_unit_center_button.widget, self.clear_selected_unit_button.widget])
                ]),

                wr.VBox([
                    wr.HTML(value=f"<font size='+1'>Nucleus: </font>").widget, 
                    wr.HTML(value=f"Reference: ").widget, 
                    wr.Label(),
                    wr.HTML(value=f"Selected: ").widget, 
                    wr.Label()
                ]), 

                wr.VBox([
                    wr.Label(),
                    self.primary_nucleus_field.widget,
                    wr.HBox([self.primary_nucleus_center_button.widget, self.clear_nucleus_object_button.widget]),
                    self.selected_nucleus_field.widget, 
                    wr.HBox([self.selected_nucleus_center_button.widget, self.clear_selected_nucleus_button.widget])
                ]), 

                

                
            ]),
            wr.HBox([
                    wr.VBox([self.match_selected_button.widget, self.match_uncertain_button.widget, self.check_neuroglancer_button.widget, self.ng_link_html.widget]),
                    wr.VBox([self.unit_no_match_selected_button.widget, self.unit_indiscernable_button.widget, self.unit_skip_button.widget, self.clear_ng_html_button.widget]),
                    wr.VBox([self.nucleus_no_match_selected_button.widget, self.nucleus_inhibitory_button.widget, self.nucleus_skip_button.widget]),
                    ]),
        ])
        
        self.match_module = wr.HBox([
            
            wr.VBox([
                wr.HTML(value=f"<font size='+1'>Primary match: </font>").widget,
                self.primary_match_select.widget,
                wr.HBox([self.primary_match_button.widget, self.primary_match_center_unit_button.widget, self.primary_match_center_nucleus_button.widget, self.clear_primary_match_select_button.widget]),
                wr.HTML(value=f"<font size='+1'>Secondary match: </font>").widget,
                self.secondary_match_select.widget,
                wr.HBox([self.secondary_match_button.widget, self.secondary_center_unit_button.widget, self.secondary_center_nucleus_button.widget, self.clear_secondary_match_select_button.widget]),
                wr.HTML(value=f"<font size='+0'>Match Note: </font>").widget,
                wr.HBox([self.match_note_field.widget, self.clear_match_note_button.widget])
            ]),
            wr.VBox([
                wr.Label(),
                self.submit_all_primary_secondary_button.widget,
                self.clear_all_primary_secondary_button.widget
            ])
        ])

        
        self.fig_module = wr.HBox([
            wr.VBox([
                wr.HBox([
                    wr.HTML(value=f"<font size='+1'>Fig Options:</font>").widget,
                    self.fig_width_field.widget, 
                    self.fig_height_field.widget, 
                    self.dpi_field.widget,
                    self.replot_button.widget
                    ]),

                wr.HBox([
                    wr.HTML(value=f"<font size='+1'>Axis Options:</font>").widget,
                    self.depth_field.widget,
                    self.xlim_slider.widget,
                    self.ylim_slider.widget,
                    self.max_lim_button.widget
                    ]),

                wr.HBox([
                    self.zoom_to_unit_check.widget,
                    self.circle_unit_check.widget,
                    self.zoom_to_nucleus_check.widget,
                    self.unit_scatter_check.widget,
                    self.filter_non_neurons_check.widget,
                    self.cache_images_check.widget,
                    self.vessels_check.widget
                    # self.clear_feedback_button.widget
                ])
            ])
        ])
        
        self.module = wr.VBox([

            wr.HBox([self.user_id_field.widget, self.submit_lag_report_button.widget]),

            wr.HBox(layout={'height':'25px'}),

            wr.HTML(value=f"<font size='+2'>Select:</font>").widget,

            wr.HBox([
                self.field_unit_selector_module.module, 
                self.nucleus_selector_module.module, 
                wr.HBox(layout={'height':'200px'})
            ]),

            wr.HBox([wr.HTML(value=f"<font size='+2'>Load:</font>").widget, self.match_protocol_feedback_out,]),
            
            wr.HBox([self.unit_match_candidate_module.module, wr.HBox(layout={'width':'50px'}), self.nucleus_match_candidate_module.module]),

            wr.HBox(layout={'height':'25px'}),

            

            wr.HBox(layout={'height':'25px'}),

            wr.HBox([
                wr.HTML(value=f"<font size='+2'>Selected Match Protocol:</font>").widget, self.selected_match_protocol_header.widget, self.select_random_candidate_button.widget, self.clear_selected_match_protocol_button.widget
                ]),

            wr.HBox(layout={'height':'25px'}),

            wr.HTML(value=f"<font size='+2'>Queue:</font>").widget,
            
            wr.HBox([
                wr.HBox([self.unit_queue_module.module, self.nucleus_queue_module.module])
            ]),
            
            wr.HBox(layout={'height':'50px'}),

            wr.HTML(value=f"<font size='+2'>Search:</font>").widget,
            
            self.unit_nucleus_search_module.module,
            
            wr.HBox(layout={'height':'50px'}),
            
            

            wr.HBox([wr.HTML(value=f"<font size='+2'>Match:</font>").widget, self.reset_multimatch_button.widget]),
            
            wr.HBox([self.info_module, wr.HBox(layout={'width':'50px'}), self.match_module]),
            
            self.fig_module,
                        
            self.feedback_out,
            self.plot_output
        ])

    
    def display(self):
        
        # if not self.datajoint_connect_module.is_connected:
        #     # with self.module_out:
        #     #     wr.clear_output()
        #     display(self.datajoint_connect_module.module)
        # else:
        #     # self.module_out.clear_output()
        wr.display(self.module)
            
    
    
    def custom_msg(self, msg:str):
        with self.feedback_out:
            wr.clear_output()
            wr.display(wr.HBox([wr.Label(msg), self.clear_feedback_button.widget]))

    def custom_msg_before_action(self, msg:str, output:wr.Output, action, interact_kws={}, action_button_description:str=None):
        with output:
            wr.clear_output()
            wr.display(wr.HBox([
                wr.Label(msg), 
                wr.Button(on_interact=action, interact_kws=interact_kws, description='Action' if action_button_description is None else action_button_description, layout={'width': 'auto'}, button_style='success').widget,
                wr.Button(on_interact=output.clear_output, description='Clear', layout={'width': 'auto'}, button_style='info').widget
            ]))
    
    def match_protocol_custom_msg(self, msg:str):
        with self.match_protocol_feedback_out:
            wr.clear_output()
            wr.display(wr.HBox([wr.Label(msg), self.clear_match_protocol_feedback_button.widget]))
    
    
    def default_values(self, name, value):
        return value if name not in self.defaults else self.defaults[name]
    
        # PROPERTY METHODS
    @property
    def multimatch_params(self):
        return self._multimatch_params

    @property
    def figure(self):
        return self._figure
    
    @property
    def plot_functions(self):
        return self._plot_functions
    
    @property
    def fig_kws(self):
        return self._fig_kws
    
    @property
    def plot_kws(self):
        return self._plot_kws

    @property
    def images(self):
        return self._images
    
    @property
    def field_unit_object(self):
        return self._field_unit_object
    
    @property
    def nucleus_object(self):
        return self._nucleus_object
    
    @property
    def selected_nucleus(self):
        return self._selected_nucleus
    
    @property
    def selected_unit(self):
        return self._selected_unit

    @property
    def current_centroids(self):
        return self._current_centroids

    @property
    def current_labels(self):
        return self._current_labels

    @property
    def selected_match_protocol(self):
        return self._selected_match_protocol
    
    def initialize_stack_loaders(self):      
        self.stack_info = {
            'vess': {
                'parent_table': m65.Stack2PResized, 
                'depth_table': m65.Stack2PResizedByDepth, 
                'stack_key': {'resized_idx': 2, 'resized_name': 'vessel_stack'},
                'stack_npy_path': Path('/mnt/scratch10/platinum-scratch/stacks_multimatch/sharpened_vessels.npy'),
                'load_source': self.multimatch_params['load_source']
            },
            'prob': {
                'parent_table': m65.Stack2PResized, 
                'depth_table': m65.Stack2PResizedByDepth, 
                'stack_key': {'resized_idx': 5, 'resized_name': 'prob_stack_thresh_0p2_surface'},
                'stack_npy_path': Path('/mnt/scratch10/platinum-scratch/stacks_multimatch/prob_stack_thresh_0p2_surface.npy'),
                'load_source': self.multimatch_params['load_source']
            },
            'nuc_seg': {
                'parent_table': m65.EMSegStack2PCoords, 
                'depth_table': m65.EMSegStack2PCoordsByDepth, 
                'stack_key': {'em_stack_idx': 2, 'em_seg_stack_idx': 4, 'resized_idx': 1, 'transform_id': 7},
                'stack_npy_path': Path('/mnt/scratch10/platinum-scratch/stacks_multimatch/nuc_seg_2p_coords_1_um_per_pixel_phase3.npy'),
                'load_source': self.multimatch_params['load_source']
            },
            'neuron_nuc_seg': {
                'parent_table': m65.EMSegStack2PCoords, 
                'depth_table': m65.EMSegStack2PCoordsByDepth, 
                'stack_key': {'em_stack_idx': 2, 'em_seg_stack_idx': 4, 'resized_idx': 3, 'transform_id': 7},
                'stack_npy_path': Path('/mnt/scratch10/platinum-scratch/stacks_multimatch/nuc_seg_2p_coords_1_um_per_pixel_phase3_filtered_to_neurons.npy'),
                'load_source': self.multimatch_params['load_source']
            },

            'em': {'parent_table': m65.EMStack2PCoords, 
                   'depth_table': m65.EMStack2PCoordsByDepth, 
                   'stack_key': {'em_stack_idx': 2, 'resized_idx': 1, 'transform_id': self.multimatch_params['transform_id']},
                   'stack_npy_path': Path(f'/mnt/scratch10/platinum-scratch/stacks_multimatch/EM_at_1_um_per_px_phase3_2p_coords_t{self.multimatch_params["transform_id"]}.npy'),
                    'load_source': self.multimatch_params['load_source']
                  },

            # 'm35em': {'parent_table': m65.EMStack2PCoords, 
            #        'depth_table': m65.EMStack2PCoordsByDepth, 
            #        'stack_key': {'em_stack_idx': 3, 'resized_idx': 1, 'transform_id': 7},
            #        'stack_npy_path': Path('/mnt/scratch10/platinum-scratch/stacks_multimatch/EM_m35_at_1_um_per_px_phase3_2p_coords.npy'),
            #         'load_source': self.multimatch_params['load_source']
            #       }
        }
        
        self.stack_loaders = {k: wu.StackByDepthLoader(**v) for k, v in self.stack_info.items()}    
    
    # UPDATE METHODS
    def update_multimatch_params(self, load_mode=None, from_ui=False):
        if load_mode is not None:
            if load_mode == 'cache':
                self._multimatch_params['load_mode'] = 'cache'
                self.cache_images_check.update_value(True, action=False)
            
            elif load_mode == 'view':
                self._multimatch_params['load_mode'] = 'view'
                self.cache_images_check.update_value(False, action=False)
            
            else:
                raise Exception('"load_mode" not recognized. Choose "cache" or "view".')
        
        if from_ui:
            if self.cache_images_check.widget.value:
                self._multimatch_params['load_mode'] = 'cache'
            else:
                self._multimatch_params['load_mode'] = 'view'


    def close_figure(self):
        if self.figure is not None:
            plt.close(self.figure.fig)
            self.plot_output.clear_output()
            self._figure = None
            self._current_nuc_image = None
    
    
    def update_figure(self):
        self.close_figure()
        self._figure = wu.Fig(plot_functions=self.plot_functions, output=self.plot_output, fig_kws=self.fig_kws)        
        self.figure.add_scroll_event(scroll_up_action=self.scroll_up_action, scroll_down_action=self.scroll_down_action)
        self.figure.add_button_press_event(button_press_action=self.button_press_action)
        self.figure.add_draw_event(draw_action=self.draw_action)
        self.figure.add_pick_event(pick_action=self.pick_action)
        self.figure.fig.tight_layout(pad=2)
        self.figure.fig.canvas.capture_scroll = True
    
    
    def update_fig_kws(self, fig_width=None, fig_height=None, dpi=None, from_ui=False, from_figure=False, update_figure=False):        
        if fig_width is not None:
            old_width, old_height = self.fig_kws['figsize']
            self._fig_kws['figsize'] = (fig_width, old_height)
            
            self.fig_width_field.update_value(fig_width, action=False)
            update_figure = True
        
        if fig_height is not None:
            old_width, old_height = self.fig_kws['figsize']
            self._fig_kws['figsize'] = (old_width, fig_height)
            
            self.fig_height_field.update_value(fig_height, action=False)
            update_figure = True
        
        if dpi is not None:
            self._fig_kws['dpi'] = dpi
            self.dpi_field.update_value(dpi, action=False)
            update_figure = True

        if from_ui:
            self._fig_kws['figsize'] = (self.fig_width_field.widget.value, self.fig_height_field.widget.value)
            self._fig_kws['dpi'] = self.dpi_field.widget.value
            update_figure = True
        
        if from_figure:
            self._fig_kws['figsize'] = (self.figure.fig.get_figwidth(), self.figure.fig.get_figheight())
            self.fig_width_field.update_value(self.fig_kws['figsize'][0], action=False)
            self.fig_height_field.update_value(self.fig_kws['figsize'][1], action=False)
        
        if update_figure:
            self.update_figure()
            
    
    def update_plot_functions(self, plot_functions=None, from_ui=False, update_figure=False):
        if plot_functions is not None:
            self._plot_functions = plot_functions
            update_figure=True
            
        if from_ui:
            pass
        
        if update_figure:
            self.update_figure()
    
    
    def update_plot(self):
        if self.figure is not None:
            self.figure.update_plot()
        else:
            self.update_figure()
            self.figure.update_plot()
    
    
    def update_plot_kws(self, depth=None, xlim=None, ylim=None, from_ui=False, from_figure=False, update_plot=True):
        if depth is not None:
            self._plot_kws['depth'] = depth
            self.depth_field.update_value(depth, action=False)
        
        if xlim is not None:
            self._plot_kws['xlim'] = xlim
            self.xlim_slider.update_value(xlim, action=False)
            
        if ylim is not None:
            self._plot_kws['ylim'] = ylim
            self.ylim_slider.update_value(ylim, action=False)
        
        if from_ui:
            self._plot_kws['depth'] = self.depth_field.widget.value
            self._plot_kws['xlim'] = self.xlim_slider.widget.value
            self._plot_kws['ylim'] = self.ylim_slider.widget.value

        if from_figure:
            self._plot_kws['xlim'] = [ax.get_xlim() for ax in self.figure.fig.axes][0]
            self._plot_kws['ylim'] = [ax.get_ylim()[::-1] for ax in self.figure.fig.axes][0]
            self.xlim_slider.update_value(self.plot_kws['xlim'], action=False)
            self.ylim_slider.update_value(self.plot_kws['ylim'], action=False)
            update_plot=False
            
        if update_plot:
            self.update_plot()
    

    def load_images(self, key):
        with self.feedback_out:
            wr.clear_output() 
            
            if key=='vess':                    
                if not self.stack_loaders['vess'].check_if_loaded(depth=self.plot_kws["depth"]):
                    print('loading 2P vessel segmentation...')
                    self.stack_loaders['vess'].get_stack_images(depth=self.plot_kws["depth"], padding=25, load_mode=self.multimatch_params['load_mode'])
                    wr.clear_output()

            elif key=='prob':
                if not self.stack_loaders['prob'].check_if_loaded(depth=self.plot_kws["depth"]):
                    print('loading 2P soma probability...')
                    self.stack_loaders['prob'].get_stack_images(depth=self.plot_kws["depth"], padding=25)
                    wr.clear_output()

            elif key=='nuc_seg':
                if not self.stack_loaders['nuc_seg'].check_if_loaded(depth=self.plot_kws['depth']):
                    print('loading EM nucleus segmentation...')
                    self.stack_loaders['nuc_seg'].get_stack_images(depth=self.plot_kws['depth'], padding=25)
                    wr.clear_output()

            elif key=='neuron_nuc_seg':
                if not self.stack_loaders['neuron_nuc_seg'].check_if_loaded(depth=self.plot_kws['depth']):
                    print('loading EM neuron nucleus segmentation...')
                    self.stack_loaders['neuron_nuc_seg'].get_stack_images(depth=self.plot_kws['depth'], padding=25)
                    wr.clear_output()

            elif key=='em':
                for d in [self.plot_kws["depth"]-10, self.plot_kws["depth"]+10]:
                    if not self.stack_loaders['em'].check_if_loaded(d):
                        print('loading EM ...')
                        self.stack_loaders['em'].get_stack_images(depth=d, padding=25)
                        wr.clear_output()

            elif key=='m35em':
                for d in [self.plot_kws["depth"]-10, self.plot_kws["depth"]+10]:
                    if not self.stack_loaders['m35em'].check_if_loaded(d):
                        print('loading m35 EM ...')
                        self.stack_loaders['m35em'].get_stack_images(depth=d, padding=25)
                        wr.clear_output()


    def update_images(self, update_scan=False, update_vess=False, update_prob=False, update_nuc_seg=False, update_neuron_nuc_seg=False, update_em=False, update_m35em=False, view_only=False):
        if update_scan:
            if self.field_unit_object is not None:
                field_image = self.field_unit_object.field_image
                scan_vess = self.field_unit_object.scan_vess
                scan_nug_seg = self.field_unit_object.scan_nuc_seg

                self._images['scan'] = crg.normalize(field_image, newrange=[0,1], astype=np.float, clip_bounds=[0, field_image.max()])
                self._images['scan_vess'] = crg.normalize(scan_vess, newrange=[0,1], astype=np.float, clip_bounds=[0, scan_vess.max()])  
                self._images['scan_nuc_seg'] = scan_nug_seg
        
        if update_vess:
            if self.multimatch_params['load_mode']=='cache':
                self.load_images('vess')
                im = self.stack_loaders['vess'].loaded_stack[self.plot_kws["depth"]]
            else:
                im = self.stack_loaders['vess'].get_stack_images(depth=self.plot_kws['depth'], load_mode='view')
            
            self._images['vess'] = crg.normalize(im, newrange=[0,1], astype=np.float, clip_bounds=[0, im.max()])
                
        if update_prob:
            if self.multimatch_params['load_mode']=='cache':
                self.load_images('prob')
                im = self.stack_loaders['prob'].loaded_stack[self.plot_kws["depth"]]
            else:
                im = self.stack_loaders['prob'].get_stack_images(depth=self.plot_kws['depth'], load_mode='view')

            self._images['prob'] = im

        if update_nuc_seg:
            if self.multimatch_params['load_mode']=='cache':
                self.load_images('nuc_seg')
                im = self.stack_loaders['nuc_seg'].loaded_stack[self.plot_kws["depth"]]
            else:
                im = self.stack_loaders['nuc_seg'].get_stack_images(depth=self.plot_kws['depth'], load_mode='view')

            self._images['nuc_seg'] = im
            
        if update_neuron_nuc_seg:
            if self.multimatch_params['load_mode']=='cache':
                self.load_images('neuron_nuc_seg')
                im = self.stack_loaders['neuron_nuc_seg'].loaded_stack[self.plot_kws["depth"]]
            else:
                im = self.stack_loaders['neuron_nuc_seg'].get_stack_images(depth=self.plot_kws['depth'], load_mode='view')

            self._images['neuron_nuc_seg'] = im

        if update_em:
            mean_padding = 5
            if self.multimatch_params['load_mode']=='cache':
                self.load_images('em')
                ims = self.stack_loaders['em'].loaded_stack[self.plot_kws["depth"]-mean_padding:self.plot_kws["depth"]+mean_padding]
        
            else:
                ims = self.stack_loaders['em'].get_stack_images(depth_range=(self.plot_kws["depth"]-mean_padding, self.plot_kws["depth"]+mean_padding), load_mode='view')

            mean = np.mean(ims, axis=0)
            im = crg.normalize(mean, clip_bounds=[100,150], newrange=[0,1], astype=np.float)
            self._images['em'] = im

        if update_m35em:
            mean_padding = 5
            if self.multimatch_params['load_mode']=='cache':
                self.load_images('m35em')
                ims = self.stack_loaders['m35em'].loaded_stack[self.plot_kws["depth"]-mean_padding:self.plot_kws["depth"]+mean_padding]
        
            else:
                ims = self.stack_loaders['m35em'].get_stack_images(depth_range=(self.plot_kws["depth"]-mean_padding, self.plot_kws["depth"]+mean_padding), load_mode='view')

            mean = np.mean(ims, axis=0)
            im = crg.normalize(mean, clip_bounds=[100,150], newrange=[0,1], astype=np.float)
            self._images['m35em'] = im


    def clear_images(self):
        for k in self._images.keys():
            self._images[k] = np.zeros((1322, 1412))
        
        self.update_plot()

    def clear_nucleus_object(self):
        if self.nucleus_object is not None:
            self._nucleus_object = None
            self.primary_nucleus_field.update_value(0, action=False)
            self.update_plot()
    
    def update_nucleus_object(self, nucleus_object=None):
        self.clear_nucleus_object()
        
        if nucleus_object is not None:
            self._nucleus_object = NucleusObject(nucleus_object)
            self.primary_nucleus_field.update_value(self.nucleus_object.nucleus_key.nucleus_id, action=False)
            
            if self.zoom_to_nucleus_check.widget.value:
                self.center_plot(primary_nucleus=True)

            else:
                self.update_plot_kws(
                        depth=self.nucleus_object.em_z_tp.round().astype(np.int)
                )
    

    def clear_selected_nucleus(self):
        if self.selected_nucleus is not None:
            self._selected_nucleus = None
            self.selected_nucleus_field.update_value(0, action=False)
            self.update_plot()

    def update_selected_nucleus(self, selected_nucleus=None, from_ui=False):
        self.clear_selected_nucleus()
        
        if selected_nucleus is not None:
            self._selected_nucleus = selected_nucleus
            self.selected_nucleus_field.update_value(selected_nucleus, action=False)
            self.update_plot()
        
        if from_ui:
            selected_nucleus = self.selected_nucleus_field.widget.value
            
            if selected_nucleus in WidgetTemplates.nuclei:
                self._selected_nucleus = selected_nucleus
                self.update_plot()
            else:
                self.custom_msg('Nucleus not in database.')
                self.selected_nucleus_field.update_value(0, action=False)

    def clear_field_unit_object(self):
        if self.field_unit_object is not None:
            self._field_unit_object = None
            self.field_key_select.widget.options = []
            self.field_key_select.update_value(None, action=False)    
            self.primary_unit_field.update_value(0, action=False)
            self._current_centroids = None
            self._current_labels = None
            self.update_ng_link(clear=True)
            self.clear_selected_unit(update_plot=False)
            self.clear_match_select(clear_all=True)
            self.update_plot()
    
    def clear_primary_unit(self):
        if self.field_unit_object is not None:
            self.field_unit_object.clear_unit_only()
            self.primary_unit_field.update_value(0, action=False)
            self.update_plot()
    
    def update_field_unit_object(self, field_unit_object):
        self.clear_field_unit_object()
        
        if field_unit_object is not None:
            self._field_unit_object = FieldUnitObject(field_unit_object)
            self.field_key_select.widget.options = list(self.field_key_select.widget.options) + [self.field_unit_object.field_key]
            self.field_key_select.update_value(self.field_unit_object.field_key, action=None)

            if self.field_unit_object.unit_key.unit_id is not None:
                self.primary_unit_field.update_value(self.field_unit_object.unit_key.unit_id, action=None)

                if self.zoom_to_unit_check.widget.value:
                    self.center_plot(primary_unit=True)
                
                else:
                    self.update_plot_kws(depth=self.field_unit_object.unit_stack_depth.round().astype(np.int))

            else:
                self.update_plot_kws(
                    depth=self.field_unit_object.field_stack_mean_depth.round().astype(np.int),
                    xlim=(0,1412),
                    ylim=(0,1322)
                )

    
    def update_selected_unit(self, unit=None, from_ui=False):        
        if unit is not None:
            self._selected_unit = unit
            self.selected_unit_field.update_value(unit, action=False)
            self.update_plot()
        
        if from_ui:
            selected_unit = self.selected_unit_field.widget.value
            if self.field_unit_object is not None:
                if len(m65.UnitStackInfo() & self.field_unit_object.field_key._asdict() & {'unit_id': selected_unit}) > 0:
                        self._selected_unit = selected_unit
                        self.update_plot()
                else:
                    self.custom_msg('Entered unit not in field.')
                    self.selected_unit_field.update_value(0, action=False)
            else:
                self.custom_msg('Field must be loaded.')
                self.selected_unit_field.update_value(0, action=False)
                
    
    def clear_selected_unit(self, update_plot=True):
        if self.selected_unit is not None:
            self._selected_unit = None
            self.selected_unit_field.update_value(0, action=False)
            if update_plot:
                self.update_plot()       

    def update_selected_match_protocol(self, selected_match_protocol=None, protocol_type=None):
        self.clear_selected_match_protocol()

        if selected_match_protocol is not None:
            self._selected_match_protocol = selected_match_protocol

            if protocol_type is not None:
                self.selected_match_protocol['protocol_type'] = protocol_type
            
            else:
                self.custom_msg('Protocol type should not be "None"')
                return
            
            if self.selected_match_protocol['protocol_type'] == "Unit":
                field_rel = m65.UnitMatchProtocol.FieldInclude & self.selected_match_protocol
                if len(field_rel)>0:
                    self.unit_nucleus_search_module.update_unit_id_rel(unit_id_rel=m65.UnitAnatomyFunction & field_rel)
            
            elif self.selected_match_protocol['protocol_type'] == "Nucleus":
                field_rel = m65.NucleusMatchProtocol.FieldInclude & self.selected_match_protocol
                distance_thresh = (m65.NucleusMatchProtocol & self.selected_match_protocol).fetch1('nearest_unit_max')
                unit_id_rel = m65.NucleusMatchProtocolUnitInclude & self.selected_match_protocol

                if len(field_rel)>0:
                    self.unit_nucleus_search_module.update_unit_id_rel(unit_id_rel=m65.UnitAnatomyFunction & field_rel)

                if len(unit_id_rel)>0:
                    self.unit_nucleus_search_module.update_unit_id_rel(unit_id_rel=m65.UnitAnatomyFunction & unit_id_rel)
                
                if distance_thresh > 0:
                    self.unit_nucleus_search_module.update_distance_thresh(distance_thresh=distance_thresh)
            
            else:
                self.match_protocol_custom_msg('Protocol type not recognized.')
                return

            self.selected_match_protocol_header.widget.value = f"<font size='+1'>{self.selected_match_protocol['protocol_type']} Protocol, {self.selected_match_protocol['protocol_name']}</font>"
                
    def clear_selected_match_protocol(self):
        self._selected_match_protocol = None
        self.selected_match_protocol_header.widget.value = "<font size='+1'>None Selected</font>"
        self.unit_nucleus_search_module.update_unit_id_rel(reset_to_default=True)
        self.unit_nucleus_search_module.update_distance_thresh(reset_to_default=True)
        
    def center_plot(self, primary_unit=False, selected_unit=False, primary_nucleus=False, selected_nucleus=False, primary_match_unit=False, primary_match_nucleus=False, secondary_match_unit=False, secondary_match_nucleus=False):
        if primary_unit:
            if self.field_unit_object is not None:
                    self.update_plot_kws(
                        depth=self.field_unit_object.unit_stack_depth.round().astype(np.int),
                        xlim=(self.field_unit_object.unit_stack_centroid[0] - self.default_values('zoom_width', 100), self.field_unit_object.unit_stack_centroid[0] + self.default_values('zoom_width', 100)),
                        ylim=(self.field_unit_object.unit_stack_centroid[1] - self.default_values('zoom_height', 100), self.field_unit_object.unit_stack_centroid[1] + self.default_values('zoom_height', 100))
                        )
        
        if selected_unit:
            if self.selected_unit is not None:
                selected_unit_stack_centroid = np.stack((m65.UnitStackInfo() & self.field_unit_object.field_key._asdict() & {'unit_id': self.selected_unit}).fetch1('unit_x', 'unit_y', 'unit_z'), -1)
                self.update_plot_kws(
                        depth=selected_unit_stack_centroid[2].round().astype(np.int),
                        xlim=(selected_unit_stack_centroid[0] - self.default_values('zoom_width', 100), selected_unit_stack_centroid[0] + self.default_values('zoom_width', 100)),
                        ylim=(selected_unit_stack_centroid[1] - self.default_values('zoom_height', 100), selected_unit_stack_centroid[1] + self.default_values('zoom_height', 100))
                        )
        
        if primary_nucleus:
            if self.nucleus_object is not None:
                self.update_plot_kws(
                        depth=self.nucleus_object.em_z_tp.round().astype(np.int),
                        xlim=(self.nucleus_object.em_x_tp - self.default_values('zoom_width', 100), self.nucleus_object.em_x_tp + self.default_values('zoom_width', 100)),
                        ylim=(self.nucleus_object.em_y_tp - self.default_values('zoom_height', 100), self.nucleus_object.em_y_tp + self.default_values('zoom_height', 100))
                    )
        
        if selected_nucleus:
            if self.selected_nucleus is not None:
                selected_nucleus_object = NucleusObject(nucleus=DataType.Nucleus(self.selected_nucleus))
                self.update_plot_kws(
                        depth=selected_nucleus_object.em_z_tp.round().astype(np.int),
                        xlim=(selected_nucleus_object.em_x_tp - self.default_values('zoom_width', 100), selected_nucleus_object.em_x_tp + self.default_values('zoom_width', 100)),
                        ylim=(selected_nucleus_object.em_y_tp - self.default_values('zoom_height', 100), selected_nucleus_object.em_y_tp + self.default_values('zoom_height', 100))
                    )
        
        # CENTER ON MATCHES
        match_unit=False
        match_nucleus=False

        if primary_match_unit:
            key = self.primary_match_select.widget.value
            match_unit=True

        if secondary_match_unit:
            key = self.secondary_match_select.widget.value
            match_unit=True

        if primary_match_nucleus:
            key = self.primary_match_select.widget.value
            match_nucleus=True

        if secondary_match_nucleus:
            key = self.secondary_match_select.widget.value
            match_nucleus=True


        
        if match_unit:
            if key.unit_id is not None:
                field_unit_object = FieldUnitObject(scan_session=key.scan_session, scan_idx=key.scan_idx, field=key.field, unit_id=key.unit_id)
                self.update_plot_kws(
                            depth=field_unit_object.unit_stack_depth.round().astype(np.int),
                            xlim=(field_unit_object.unit_stack_centroid[0] - self.default_values('zoom_width', 100), field_unit_object.unit_stack_centroid[0] + self.default_values('zoom_width', 100)),
                            ylim=(field_unit_object.unit_stack_centroid[1] - self.default_values('zoom_height', 100), field_unit_object.unit_stack_centroid[1] + self.default_values('zoom_height', 100))
                            )
            else:
                self.custom_msg('Cannot center to unit if unit is missing.')
                return
        
        if match_nucleus:
            if key.nucleus_id is not None:
                nucleus_object = NucleusObject(key.nucleus_id)
                self.update_plot_kws(
                        depth=nucleus_object.em_z_tp.round().astype(np.int),
                        xlim=(nucleus_object.em_x_tp - self.default_values('zoom_width', 100), nucleus_object.em_x_tp + self.default_values('zoom_width', 100)),
                        ylim=(nucleus_object.em_y_tp - self.default_values('zoom_height', 100), nucleus_object.em_y_tp + self.default_values('zoom_height', 100))
                    )
            else:
                self.custom_msg('Cannot center to nucleus if nucleus is missing.')



        




    # FIGURE ACTIONS
    def draw_action(self, event):
        self.figure.fig.canvas.blit(self.figure.fig.bbox)
        self.figure.fig.canvas.flush_events()
        self.update_plot_kws(from_figure=True)        
    
    def scroll_up_action(self, event):
        self.update_plot_kws(depth=self.plot_kws['depth']+1)

    
    def scroll_down_action(self, event):
        self.update_plot_kws(depth=self.plot_kws['depth']-1)
        
    
    def button_press_action(self, event):     
        for ax in self.figure.fig.axes[:1]:
            for im in ax.images[:1]:
                
                    
                if event.button==3:
                    clicked_location = np.int(event.ydata), np.int(event.xdata)

                    if self.filter_non_neurons_check.widget.value:
                        clicked_nucleus_id = np.int(self.images['neuron_nuc_seg'][clicked_location])
                    else:
                        clicked_nucleus_id = np.int(self.images['nuc_seg'][clicked_location])
                    
                    if self.selected_nucleus == clicked_nucleus_id or clicked_nucleus_id == 0:
                        self.clear_selected_nucleus()

                    else:
                        self.update_selected_nucleus(clicked_nucleus_id)
                            
    
    def pick_action(self, event):
        with self.feedback_out:
            wr.clear_output()
            
            for ax in self.figure.fig.axes[:1]:
                for sc in ax.collections:
                    if sc.get_picker():
                        self.pick_event = event
                        
                        if event.mouseevent.button == 1:
                            clicked_unit = self._current_labels[event.ind][0]

                            if clicked_unit != self.selected_unit:
                                self.update_selected_unit(clicked_unit)
                            else:
                                self.clear_selected_unit()
    
    ## PLOTS
    def plot_scan_field(self, ax, *args, **kwargs):
        self.update_images(update_scan=True)
        
        ax.cla()
        ax.imshow(self.images['scan'])

        self.add_nuc_mask(ax)

        self.ax_format(ax)


        self.add_unit_scatter(ax)

    def plot_scan_field_and_vessels(self, ax, *args, **kwargs):
        self.update_images(update_scan=True)
        
        ax.cla()
        ax.imshow(np.stack([self.images['scan_vess']/2 if self.vessels_check.widget.value else np.zeros_like(self.images['scan']), self.images['scan'], np.zeros_like(self.images['scan'])], -1))


        self.add_nuc_mask(ax)

        self.ax_format(ax)


        self.add_unit_scatter(ax)
    
    def plot_vess_field(self, ax, *args, **kwargs):
        self.update_images(update_vess=True)
        
        ax.cla()
        ax.imshow(self.images['vess'], cmap='Reds')

        self.add_nuc_mask(ax)

        self.ax_format(ax)


        self.add_unit_scatter(ax)

    def plot_prob_field(self, ax, *args, **kwargs):
        self.update_images(update_prob=True)
        
        ax.cla()
        ax.imshow(self.images['prob'])

        self.ax_format(ax)


        self.add_unit_scatter(ax)

    def plot_nuc_seg(self, ax, *args, **kwargs):        
        if self.filter_non_neurons_check.widget.value:
            self.update_images(update_neuron_nuc_seg=True)
            image = self.images['neuron_nuc_seg']
        else:
            self.update_images(update_nuc_seg=True)
            image = self.images['nuc_seg']
        
        ax.cla()
        ax.imshow(image>1, cmap='Greys_r')

        self.add_nuc_mask(ax)
        
        self.ax_format(ax)


        self.add_unit_scatter(ax)


    def plot_em_field(self, ax, *args, **kwargs):
        self.update_images(update_em=True)
        
        ax.cla()
        ax.imshow(self.images['em'], cmap='Greys_r')

        self.ax_format(ax)


        self.add_unit_scatter(ax)

    
    def plot_m35em_field(self, ax, *args, **kwargs):
        self.update_images(update_em=True)
        
        ax.cla()
        ax.imshow(self.images['m35em'], cmap='Greys_r')

        self.ax_format(ax)


        self.add_unit_scatter(ax)


    def plot_em_and_vessels(self, ax, *args, **kwargs):
        self.update_images(update_vess=True, update_em=True)

        ax.cla()
        ax.imshow(self.images['em'], cmap='Greys_r')

        if self.vessels_check.widget.value:
            ax.imshow(self.images['vess'], cmap="Reds", alpha=0.2)

        self.add_nuc_mask(ax)

        self.ax_format(ax)


        self.add_unit_scatter(ax)


    def plot_em_nucs_and_vessels(self, ax, *args, **kwargs):
        self.update_images(update_vess=True, update_nuc_seg=True, update_em=True)

        ax.cla()
        ax.imshow(self.images['em'], cmap='Greys_r')

        if self.vessels_check.widget.value:
            ax.imshow(self.images['vess'], cmap="Reds", alpha=0.2)

        ax.imshow(self.images['nuc_seg']<1, cmap='Greys', alpha=self.images['nuc_seg'])

        self.ax_format(ax)


        self.add_unit_scatter(ax)


    def add_nuc_mask(self, ax):
        # set appropriate stack
        if self.filter_non_neurons_check.widget.value:
            image = self.images['neuron_nuc_seg']
        else:
            image = self.images['nuc_seg']

        # nucleus object
        if self.nucleus_object is not None:
            nucleus_id = self.nucleus_object.nucleus_key.nucleus_id
            if np.where(image == nucleus_id)[0].size > 0:
                ax.imshow(np.ma.masked_where(image != nucleus_id, image), cmap=cm.jet, alpha=0.7)
        
        # selected nucleus
        if self.selected_nucleus is not None:
            if self.selected_nucleus>0:
                if np.where(image == self.selected_nucleus)[0].size > 0:
                    ax.imshow(np.ma.masked_where(image != self.selected_nucleus, image), cmap=cm.PuOr, alpha=0.6)

    def compute_centroids_labels_in_view(self, xy_only=False):
        self._x_range = (self.plot_kws['xlim'][1] - self.plot_kws['xlim'][0])/ 2
        self._x_center = self.plot_kws['xlim'][0] + self._x_range
        self._y_range = (self.plot_kws['ylim'][1] - self.plot_kws['ylim'][0])/ 2
        self._y_center = self.plot_kws['ylim'][0] + self._y_range
        self._z_center = self.plot_kws["depth"]
        self._z_range = 3
        
        if not xy_only:
            centroids, labels = self.units_in_bbox(centroids=self.field_unit_object.unit_stack_centroids, x=self._x_center, x_range=self._x_range, y=self._y_center, y_range=self._y_range, z=self._z_center, z_range=self._z_range, labels=self.field_unit_object.unit_labels)
        else:
            centroids, labels = self.units_in_bbox(centroids=self.field_unit_object.unit_stack_centroids, x=self._x_center, x_range=self._x_range, y=self._y_center, y_range=self._y_range, labels=self.field_unit_object.unit_labels)

        return centroids, labels

    def update_current_centroids_labels(self):
        self._current_centroids, self._current_labels = self.compute_centroids_labels_in_view()

    def add_unit_scatter(self, ax):
        with self.feedback_out:
            if self.field_unit_object is not None:
                if self.unit_scatter_check.widget.value: 

                    # calculate current centroids in view
                    
                    
                    self.update_current_centroids_labels()

                    ax.scatter(*self._current_centroids.T[:2], s=10, color='red', picker=True)
                
                    if self.selected_unit in self._current_labels:
                        self._selected_unit_centroid = self._current_centroids[np.where(self.selected_unit==self._current_labels)[0]].squeeze()
                        ax.scatter(*self._selected_unit_centroid[:2], s=50, color='red')


                    if self.field_unit_object.unit_key.unit_id is not None:
                        if self.field_unit_object.unit_label in self._current_labels:
                            if self.field_unit_object.unit_label == self.selected_unit:
                                ax.scatter(*self.field_unit_object.unit_stack_centroid[:2], color='purple', s=50)
                            else:
                                ax.scatter(*self.field_unit_object.unit_stack_centroid[:2], color='purple', s=10)

                        if self.circle_unit_check.widget.value:
                            ax.add_patch(Circle(self.field_unit_object.unit_stack_centroid[:2], 20, fill=False, edgecolor='purple', alpha=1))
    

    def ax_format(self, ax, title=True):
        ax.set_aspect('equal')
        ax.set_xlim(self.plot_kws["xlim"])
        ax.set_ylim(self.plot_kws["ylim"][::-1])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        if title:
            ax.set_title(f'depth: {self.plot_kws["depth"]} um')

            
            
    def units_in_bbox(self, centroids, x=None, x_range=None, y=None, y_range=None, z=None, z_range=None, labels=None):
        """
        Returns centroids at within bounding box specified by center +- range

        :param centroids: 2D array with columns x, y, z
        :param x: (optional) the x value to center the restriction
        :param y: (optional) the y value to center the restriction
        :param z: (optional) the z value to center the restriction
        :param x_range: the range +/- x
        :param x_range: the range +/- y
        :param x_range: the range +/- z 
        :param labels: (optional) list of labels to index
        """

        x_inds = np.where(np.logical_and(centroids[:,0] > x - x_range, centroids[:,0] < x + x_range))[0] if x is not None else None
        y_inds = np.where(np.logical_and(centroids[:,1] > y - y_range, centroids[:,1] < y + y_range))[0] if y is not None else None
        z_inds = np.where(np.logical_and(centroids[:,2] > z - z_range, centroids[:,2] < z + z_range))[0] if z is not None else None
        
        non_empty = [x for x in [x_inds, y_inds, z_inds] if x is not None]
        if non_empty:
            inds = list(set.intersection(*(set(s) for s in non_empty)) )
        else:
            inds = list(set())

        if labels is not None:
            return centroids[inds], labels[inds]

        else:
            return centroids[inds]
    
    
    def add_to_nucleus_queue(self):
        nucleus_id = self.nucleus_selector_module.nucleus_select.widget.value
        
        if nucleus_id is not None:
            
            if nucleus_id in self.nucleus_selector_module.nuclei_options:
                new_entry = DataType.Nucleus(nucleus_id)
                result = self.nucleus_queue_module.add_entry(new_entry)

                if result == 'added':
                    if self.slack_username is not None:
                        if self.selected_match_protocol is not None:
                            text = f'```{self.slack_username} ({self.user_id_field.widget.value}) loaded nucleus_id={new_entry.nucleus_id} in {self.selected_match_protocol["protocol_type"]} protocol "{self.selected_match_protocol["protocol_name"]}".```'
                        else:
                            text = f'```{self.slack_username} ({self.user_id_field.widget.value}) loaded nucleus_id={new_entry.nucleus_id}.```'
                        self.slack_output.post_to_slack_and_user(text, slack_username=self.slack_username)
    
    
    def add_to_unit_queue(self):
        session = self.field_unit_selector_module.session_select.widget.value if self.field_unit_selector_module.session_select.widget.value != 0 else None
        index = self.field_unit_selector_module.index_select.widget.value if self.field_unit_selector_module.index_select.widget.value != 0 else None
        field  = self.field_unit_selector_module.field_select.widget.value if self.field_unit_selector_module.field_select.widget.value != 0 else None
        unit = self.field_unit_selector_module.unit_select.widget.value if self.field_unit_selector_module.unit_select.widget.value != 0 else None

        if session is not None and index is not None:
            if field is None:
                if unit is None:
                    return
                else:
                    field = self.field_unit_selector_module.restrict_UnitStackInfo(session, index, unit=unit).fetch1('field')
                    new_entry = DataType.Unit(session, index, field, unit)
                    result = self.unit_queue_module.add_entry(new_entry)

                    if result == 'added':
                        if self.slack_username is not None:
                            if self.selected_match_protocol is not None:
                                text = f'```{self.slack_username} ({self.user_id_field.widget.value}) loaded scan_session={new_entry.scan_session}, scan_idx={new_entry.scan_idx}, field={new_entry.field}, unit_id={new_entry.unit_id} in {self.selected_match_protocol["protocol_type"]} protocol "{self.selected_match_protocol["protocol_name"]}".```'
                            else:
                                text = f'```{self.slack_username} ({self.user_id_field.widget.value}) loaded scan_session={new_entry.scan_session}, scan_idx={new_entry.scan_idx}, field={new_entry.field}, unit_id={new_entry.unit_id}.```'
                            self.slack_output.post_to_slack_and_user(text, slack_username=self.slack_username)
            else:
                if unit is None:
                    new_entry = DataType.Field(session, index, field)
                    result = self.unit_queue_module.add_entry(new_entry)

                    if result == 'added':
                        if self.slack_username is not None:
                            if self.selected_match_protocol is not None:
                                text = f'```{self.slack_username} ({self.user_id_field.widget.value}) loaded scan_session={new_entry.scan_session}, scan_idx={new_entry.scan_idx}, field={new_entry.field} in {self.selected_match_protocol["protocol_type"]} protocol "{self.selected_match_protocol["protocol_name"]}".```'
                            else:
                                text = f'```{self.slack_username} ({self.user_id_field.widget.value}) loaded scan_session={new_entry.scan_session}, scan_idx={new_entry.scan_idx}, field={new_entry.field}.```'
                            self.slack_output.post_to_slack_and_user(text, slack_username=self.slack_username)
                else:
                    new_entry = DataType.Unit(session, index, field, unit)
                    result = self.unit_queue_module.add_entry(new_entry)

                    if result == 'added':
                        if self.slack_username is not None:
                            if self.selected_match_protocol is not None:
                                text = f'```{self.slack_username} ({self.user_id_field.widget.value}) loaded scan_session={new_entry.scan_session}, scan_idx={new_entry.scan_idx}, field={new_entry.field}, unit_id={new_entry.unit_id} in {self.selected_match_protocol["protocol_type"]} protocol "{self.selected_match_protocol["protocol_name"]}".```'
                            else:
                                text = f'```{self.slack_username} ({self.user_id_field.widget.value}) loaded scan_session={new_entry.scan_session}, scan_idx={new_entry.scan_idx}, field={new_entry.field}, unit_id={new_entry.unit_id}.```'
                            self.slack_output.post_to_slack_and_user(text, slack_username=self.slack_username)
    
    
    def search_nearby_unit(self, **kwargs):
        unit_key = self.unit_queue_module.queue.widget.value
        if not unit_key:
            self.unit_queue_module.custom_msg('Unit must be selected.')
        
        elif 'unit_id' not in unit_key._fields:
            self.unit_queue_module.custom_msg('Key must contain a unit_id.')
        
        else:
            self.unit_nucleus_search_module.session_field.widget.value = unit_key.scan_session
            self.unit_nucleus_search_module.index_field.widget.value = unit_key.scan_idx
            self.unit_nucleus_search_module.unit_field.widget.value = unit_key.unit_id
            self.unit_nucleus_search_module.unit_search()
            
            
    def search_nearby_nucleus(self, **kwargs):
        nucleus_key = self.nucleus_queue_module.queue.widget.value
        
        if not nucleus_key:
            self.nucleus_queue_module.custom_msg('Nucleus must be selected.')
        
        elif 'nucleus_id' not in nucleus_key._fields:
            self.nucleus_queue_module.custom_msg('Key must contain a nucleus_id.')
        
        else:
            self.unit_nucleus_search_module.nucleus_id_field.widget.value = nucleus_key.nucleus_id
            self.unit_nucleus_search_module.nucleus_search()
    
    def send_field_unit_to_multimatch(self, from_queue_module=False, from_search_module=False, field_only=False, **kwargs):
        if from_queue_module:
            selected = self.unit_queue_module.queue.widget.value
            
            if selected is not None:
                self.update_field_unit_object(selected)
            else:
                self.unit_queue_module.custom_msg('Unit must be selected.')
        
        if from_search_module:
            if field_only:
                selected = self.unit_nucleus_search_module.nearest_field_select.widget.value

                if selected is not None:
                    self.update_field_unit_object(selected['Field'])
            else:
                selected = self.unit_nucleus_search_module.nearest_unit_select.widget.value
            
                if selected is not None:
                    self.update_field_unit_object(selected['Unit'])
        
    
    def send_nucleus_to_multimatch(self, from_queue_module=False, from_search_module=False, **kwargs):
        if from_queue_module:
            selected = self.nucleus_queue_module.queue.widget.value
            if selected is not None:
                self.update_nucleus_object(selected)
            else:
                self.nucleus_queue_module.custom_msg('Nucleus must be selected.')
        
        if from_search_module:
            selected = self.unit_nucleus_search_module.nearest_nucleus_select.widget.value
            if selected is not None:
                self.update_nucleus_object(selected['Nucleus'])
    
    def update_match(self, submission_type):
        # VALIDATION
        if submission_type=='match' or submission_type=='match_uncertain':
            if self.field_unit_object is not None:
                if self.selected_unit is None or self.selected_nucleus is None:
                    self.custom_msg('"Selected Unit" must NOT be 0 and "Selected Nucleus" must NOT be 0')
                    return
            else:
                self.custom_msg('Field must be loaded.')
                return
        
        elif submission_type=='unit_no_match' or submission_type=='unit_indiscernable' or submission_type=='unit_skip':
            if self.field_unit_object is not None:
                if self.selected_unit is None or self.selected_nucleus is not None:
                    self.custom_msg('"Selected Unit" must NOT be 0 and "Selected Nucleus" MUST be 0')
                    return
            else:
                self.custom_msg('Field must be loaded.')
                return
        
        elif submission_type=='nucleus_no_match':
            if self.field_unit_object is not None:
                if self.selected_nucleus is None or self.selected_unit is not None:
                    self.custom_msg('"Selected Nucleus" must NOT be 0 and "Selected Unit" MUST be 0')
                    return 
            else:
                self.custom_msg('Field must be loaded.')
                return

            
        elif submission_type=='nucleus_inhibitory' or submission_type=='nucleus_skip':
            if self.selected_nucleus is None or self.selected_unit is not None:
                self.custom_msg('"Selected Nucleus" must NOT be 0 and "Selected Unit" MUST be 0')
                return 
        
        else:
            raise Exception('"submission_type" not recognized.')

        # GENERATE SELECTED OBJECTS
        selected_field_object = FieldUnitObject(unit_id=self.selected_unit, **self.field_unit_object.field_key._asdict()) if self.field_unit_object is not None else None
        selected_nucleus_object = NucleusObject(nucleus=self.selected_nucleus) if self.selected_nucleus is not None else None

        # GENERATE MATCH OBJECT
        if selected_field_object is None and selected_nucleus_object is None:
            raise Exception('MatchObject can not be blank.')

        else:
            self._update_match_select(MatchObject(field_unit_object=selected_field_object, nucleus_object=selected_nucleus_object, submission_type=submission_type))
            


    def _update_match_select(self, match_object=None, primary=False):
        if self.field_unit_object is not None:
            if self.field_unit_object.unit_key.unit_id is not None:
                if match_object.key.unit_id == self.field_unit_object.unit_key.unit_id:
                    primary = True
        
        if self.nucleus_object is not None:
            if match_object.key.nucleus_id == self.nucleus_object.nucleus_key.nucleus_id:
                primary=True

        all_options = self.primary_match_select.widget.options + self.secondary_match_select.widget.options
        match_object_nucleus_id = match_object.key.nucleus_id
        match_object_unit_key = [match_object.key.scan_session, match_object.key.scan_idx, match_object.key.field, match_object.key.unit_id]


        if match_object_nucleus_id in [m.nucleus_id for m in all_options]:
            if match_object_nucleus_id != None:
                self.custom_msg('"Selected Nucleus" already in primary or secondary match queue.')
                return
        
        if match_object_unit_key in [[m.scan_session, m.scan_idx, m.field, m.unit_id] for m in all_options]:
            if match_object.key.unit_id != None:
                self.custom_msg('"Selected Unit" already in primary or secondary match queue.')
                return
        
        if primary:
            if len(self.primary_match_select.widget.options) < 1:
                self.primary_match_select.widget.options = list(self.primary_match_select.widget.options) + [match_object.key]
                self.primary_match_select.widget.value = match_object.key
            else:
                self.custom_msg('There can only be one primary match in the match queue at a time.')
                return
        
        else:
            self.secondary_match_select.widget.options = list(self.secondary_match_select.widget.options) + [match_object.key]
            self.secondary_match_select.widget.value = match_object.key
        
        self.clear_selected_unit()
        self.clear_selected_nucleus()


    def clear_match_select(self, primary=False, secondary=False, clear_all=False):
        if primary:
            self.primary_match_select.widget.options = []

        if secondary:
            self.secondary_match_select.widget.options = [m for m in list(self.secondary_match_select.widget.options) if m is not self.secondary_match_select.widget.value]

        if clear_all:
            self.primary_match_select.widget.options = []
            self.secondary_match_select.widget.options = []

    def clear_match_note(self):
        self.match_note_field.widget.value = ''

    # def custom_input(self):
    #     with self.feedback_out:
    #         input_field = wr.Text(description='Are you sure you would like to submit all? (Y/N) Note will be applied to all matches.')
    #         submit_input = wr.Button(description='Submit', layout={'width': 'auto'}, on_interact='')
    #         wr.display(wr.HBox([input_field.widget, submit_input.widget]))
    
    # def submit_custom_input(self):


    def submit_match(self, primary=False, secondary=False, submit_all=False):
        if self.user_id_field.widget.value == '':
            self.custom_msg('Please enter your user ID to make submissions.')
            return

        if primary:
            key = self.primary_match_select.widget.value
            self.custom_msg_before_action(msg='Are you sure you would like to submit?', output=self.feedback_out, action = self._submit_match, interact_kws=dict(key=key, priority='primary'), action_button_description='Submit')

        if secondary:
            key = self.secondary_match_select.widget.value
            self.custom_msg_before_action(msg='Are you sure you would like to submit?', output=self.feedback_out, action = self._submit_match, interact_kws=dict(key=key, priority='secondary'), action_button_description='Submit')

        if submit_all:
            primary_value = self.primary_match_select.widget.value
            secondary_options = self.secondary_match_select.widget.options

            keys = []
            priority_list = []
            if primary_value is not None:
                keys.append(primary_value)
                priority_list.append('primary')
            
            if secondary_options != ():
                for key in secondary_options:
                    keys.append(key)
                    priority_list.append('secondary')
            
            if len(keys)>0:
                self.custom_msg_before_action(msg='Are you sure you would like to submit all of these matches?', output=self.feedback_out, action = self._submit_multiple_match, interact_kws=dict(keys=keys, priority_list=priority_list), action_button_description='Submit All')
    
    def _submit_match(self, key, priority, submit_multiple=False):
        if key == '':
            return 

        insert_key = dict(
                animal_id = 17797,
                scan_session = key.scan_session,
                scan_idx = key.scan_idx,
                field = key.field,
                unit_id = key.unit_id if key.unit_id is not None else 0,
                nucleus_id = key.nucleus_id if key.nucleus_id is not None else 0,
                unit_submission_id = secrets.token_urlsafe(12),
                nucleus_submission_id = secrets.token_urlsafe(12),
                unit_submission_type=key.type,
                nucleus_submission_type=key.type,
                user_id = self.user_id_field.widget.value,
                interface = 'multimatch',
                priority = priority,
                stack_session = 9,
                stack_idx = 19,
                transform_id = 7,
                protocol_name= self.selected_match_protocol['protocol_name'] if self.selected_match_protocol is not None else None,
                note = self.match_note_field.widget.value
            )  
        
        if key.type == 'match' or key.type == 'match_uncertain':
            m65.UnitManualMatchAttempt.insert1(insert_key, ignore_extra_fields=True)
            m65.NucleusManualMatchAttempt.insert1(insert_key, ignore_extra_fields=True)
            m65.UnitNucleusManualMatch.insert1(insert_key, ignore_extra_fields=True)
            
            
        elif key.type == 'unit_no_match' or key.type == 'unit_indiscernable' or key.type == 'unit_skip':
            insert_key['unit_submission_type'] = key.type[5:]
            m65.UnitManualMatchAttempt.insert1(insert_key, ignore_extra_fields=True)
            
        
        elif key.type == 'nucleus_no_match' or key.type == 'nucleus_inhibitory' or key.type == 'nucleus_non_neuron' or key.type == 'nucleus_skip':
            insert_key['nucleus_submission_type'] = key.type[8:]
            m65.NucleusManualMatchAttempt.insert1(insert_key, ignore_extra_fields=True)
        
        # Update website time stamp for check-in/ check-out
        # ingest.UserMostRecent.insert1({'user_id': self.user_id_field.widget.value}, replace=True)
        
        # Post to slack
        if self.slack_username is not None:
            if self.selected_match_protocol is not None:
                text = f'```{self.slack_username} ({self.user_id_field.widget.value}) submitted scan_session={insert_key["scan_session"]}, scan_idx={insert_key["scan_idx"]}, field={key.field}, unit_id={insert_key["unit_id"]}, nucleus_id={insert_key["nucleus_id"]} as "{key.type}" in {self.selected_match_protocol["protocol_type"]} protocol {insert_key["protocol_name"]}```'
            else:
                text = f'```{self.slack_username} ({self.user_id_field.widget.value}) submitted scan_session={insert_key["scan_session"]}, scan_idx={insert_key["scan_idx"]}, field={key.field}, unit_id={insert_key["unit_id"]}, nucleus_id={insert_key["nucleus_id"]} as "{key.type}"```'
            self.slack_output.post_to_slack_and_user(text, slack_username=self.slack_username)

        # Feedback
        if not submit_multiple:
            self.custom_msg('Match successfully submitted.')
         
        # Clear submitted match
        if priority == 'primary':
            self.clear_primary_match_select_button.widget.click()
        if priority == 'secondary':
            self.clear_secondary_match_select_button.widget.click()
    
    def _submit_multiple_match(self, keys, priority_list):
        self.custom_msg('Submitting ..... please wait')
        
        for key, priority in zip(keys, priority_list):
            self._submit_match(key, priority, submit_multiple=True)
        
        # Feedback
        self.custom_msg('Matches successfully submitted.')
    
    def submit_nucleus_inhibitory(self, from_queue_module=False):
        if from_queue_module:
            nucleus_id = self.nucleus_queue_module.queue.widget.value.nucleus_id   
            self._submit_nucleus(nucleus_id=nucleus_id, submission_type='inhibitory')
            self.nucleus_queue_module.custom_msg('Nucleus submitted as "Inhibitory"')

    def submit_nucleus_no_match(self, from_queue_module=False):
        if from_queue_module:
            nucleus_id = self.nucleus_queue_module.queue.widget.value.nucleus_id   
            self._submit_nucleus(nucleus_id=nucleus_id, submission_type='no_match')
            self.nucleus_queue_module.custom_msg('Nucleus submitted as "No Match"')

    def submit_nucleus_non_neuron(self, from_queue_module=False):
        if from_queue_module:
            nucleus_id = self.nucleus_queue_module.queue.widget.value.nucleus_id
            self._submit_nucleus(nucleus_id=nucleus_id, submission_type='non_neuron')
            self.nucleus_queue_module.custom_msg('Nucleus submitted as "Non Neuron"')

    def _submit_nucleus(self, nucleus_id, submission_type):
        insert_key = dict(
            animal_id = 17797,
            scan_session = None,
            scan_idx = None,
            field=None,
            unit_id=None,
            nucleus_id = nucleus_id,
            nucleus_submission_id = secrets.token_urlsafe(12),
            nucleus_submission_type=submission_type,
            user_id = self.user_id_field.widget.value,
            protocol_name = self.selected_match_protocol['protocol_name'] if self.selected_match_protocol is not None else None,
            interface = 'multimatch',
            priority = 'primary',
            stack_session = 9,
            stack_idx = 19,
            transform_id = 7,
            note = None
        )  

        m65.NucleusManualMatchAttempt.insert1(insert_key, ignore_extra_fields=True)

        # Update website time stamp for check-in/ check-out
        # ingest.UserMostRecent.insert1({'user_id': self.user_id_field.widget.value}, replace=True)

        # Post to slack
        if self.slack_username is not None:
            if self.selected_match_protocol is not None:
                text = f'```{self.slack_username} ({self.user_id_field.widget.value}) submitted scan_session={insert_key["scan_session"]}, scan_idx={insert_key["scan_idx"]}, field={insert_key["field"]}, unit_id={insert_key["unit_id"]}, nucleus_id={insert_key["nucleus_id"]} as "{insert_key["nucleus_submission_type"]}" in {self.selected_match_protocol["protocol_type"]} protocol {insert_key["protocol_name"]}```'
            else:
                text = f'```{self.slack_username} ({self.user_id_field.widget.value}) submitted scan_session={insert_key["scan_session"]}, scan_idx={insert_key["scan_idx"]}, field={insert_key["field"]}, unit_id={insert_key["unit_id"]}, nucleus_id={insert_key["nucleus_id"]} as "{insert_key["nucleus_submission_type"]}"```'
            self.slack_output.post_to_slack_and_user(text, slack_username=self.slack_username)


    def gen_em_centroid_df(self, centroid):
        if np.stack(centroid).ndim == 2:
            df = pd.DataFrame(centroid)
        elif np.stack(centroid).ndim == 1:
            df = pd.DataFrame([centroid])
        else:
            raise Exception('ndim must be 1 or 2')

        df = df.rename(columns={0:'x', 1: 'y', 2:'z'})
        df = df.apply(lambda c: np.array([c.x, c.y, c.z], dtype=np.int32), axis=1)

        return df

    def generate_ng_link(self):
        if self.field_unit_object is not None:

            current_centroids, current_labels = self.compute_centroids_labels_in_view(xy_only=True)
            if current_centroids is not None and current_labels is not None:
                current_centroids_em = m65crg.Coregistration.run(current_centroids, transform_id=7)
                current_centroids_df = self.gen_em_centroid_df(current_centroids_em)
                current_labels_df = pd.DataFrame(current_labels).rename(columns={0:'unit_id'})
                current_centroids_labels_df = pd.concat([current_centroids_df,current_labels_df], axis=1).rename(columns={0:'centroids'})
                current_centroids_anno_layer = statebuilder.AnnotationLayerConfig(name='field-unit-centroids', mapping_rules=statebuilder.PointMapper('centroids', description_column='unit_id'), color='red')
            else:
                current_centroids_labels_df = None
                current_centroids_anno_layer= None
            
            if self.field_unit_object.unit_key.unit_id is not None:
                reference_centroid_em = m65crg.Coregistration.run(self.field_unit_object.unit_stack_centroid, transform_id=7)
                reference_centroid_df = self.gen_em_centroid_df(reference_centroid_em)
                reference_label_df = pd.DataFrame([self.field_unit_object.unit_label]).rename(columns={0:'unit_id'})
                reference_centroid_labels_df = pd.concat([reference_centroid_df, reference_label_df], axis=1).rename(columns={0:'centroids'})
                reference_unit_anno_layer = statebuilder.AnnotationLayerConfig(name='reference-unit-centroid', mapping_rules=statebuilder.PointMapper('centroids', description_column='unit_id'), color='purple')
            else:
                reference_centroid_labels_df = None
                reference_unit_anno_layer = None
            
            if self.selected_unit is not None:
                selected_field_object = FieldUnitObject(unit_id=self.selected_unit, **self.field_unit_object.field_key._asdict())
                selected_centroid_em = m65crg.Coregistration.run(selected_field_object.unit_stack_centroid, transform_id=7)
                selected_centroid_df = self.gen_em_centroid_df(selected_centroid_em)
                selected_label_df = pd.DataFrame([selected_field_object.unit_label]).rename(columns={0:'unit_id'})
                selected_centroid_labels_df = pd.concat([selected_centroid_df, selected_label_df], axis=1).rename(columns={0:'centroids'})
                selected_unit_anno_layer = statebuilder.AnnotationLayerConfig(name='selected-unit-centroid', mapping_rules=statebuilder.PointMapper('centroids', description_column='unit_id'), color='orange')
            else:
                selected_centroid_labels_df = None
                selected_unit_anno_layer = None
        else:
            current_centroids_labels_df = None
            current_centroids_anno_layer= None
            reference_centroid_labels_df = None
            reference_unit_anno_layer = None
            selected_centroid_labels_df = None
            selected_unit_anno_layer = None
        
        if self.nucleus_object is not None:
            reference_nuc_layer = statebuilder.SegmentationLayerConfig(crg.NgLinks.nuc_src, name='reference-nucleus', fixed_ids=[self.nucleus_object.nucleus_key.nucleus_id])
            ref_nuc_visible=True
        else:
            reference_nuc_layer = statebuilder.SegmentationLayerConfig(crg.NgLinks.nuc_src, name='reference-nucleus')
            ref_nuc_visible=False

        if self.selected_nucleus is not None:
            selected_nuc_layer = statebuilder.SegmentationLayerConfig(crg.NgLinks.nuc_src, name='selected-nucleus', fixed_ids=[self.selected_nucleus])
            sel_nuc_visible=True
        else:
            selected_nuc_layer = statebuilder.SegmentationLayerConfig(crg.NgLinks.nuc_src, name='selected-nucleus')
            sel_nuc_visible = False

        

        base_sb = statebuilder.StateBuilder([crg.NgLinks.em_layer, crg.NgLinks.seg_layer, crg.NgLinks.nuc_layer, reference_nuc_layer, selected_nuc_layer])
        field_units_sb = statebuilder.StateBuilder([current_centroids_anno_layer]) if current_centroids_anno_layer is not None else None
        selected_unit_sb = statebuilder.StateBuilder([selected_unit_anno_layer]) if selected_unit_anno_layer is not None else None
        reference_unit_sb = statebuilder.StateBuilder([reference_unit_anno_layer]) if reference_unit_anno_layer is not None else None


        chained_sb = statebuilder.ChainedStateBuilder([c for c in [base_sb, field_units_sb, reference_unit_sb, selected_unit_sb] if c is not None])
        json_state = chained_sb.render_state([None] + [s for s in [current_centroids_labels_df, reference_centroid_labels_df, selected_centroid_labels_df] if s is not None], return_as='dict')

        json_state['layers'][1].update({'visible': False})
        json_state['layers'][2].update({'visible': False})
        json_state['layers'][3].update({'visible': ref_nuc_visible})
        json_state['layers'][4].update({'visible': sel_nuc_visible})
        json_state['navigation'].update({"pose": {
            "position": {
                "voxelSize": [4, 4, 40],
                "voxelCoordinates": [219805.125, 122834, 20421.5]
                },
                "orientation": [0, -0.7071067690849304, 0, 0.7071067690849304]
                },
                "zoomFactor": 3499.999999999999
            }
        )
        json_state['perspectiveZoom'] = 2000
        self._json_state = json_state

        final_sb = statebuilder.StateBuilder([], base_state=json_state, view_kws={'layout': "4panel", 'zoom_image': 2048, 'zoom_3d': 2048})
        url = final_sb.render_state('url')

        # wu.window_open(url)
        self.ng_link_html.update_value(f"<a href={url}> Neuroglancer Link </a>")
    
    def check_morphology(self, from_queue_module=False):
        
        if from_queue_module:
            nucleus_id = self.nucleus_queue_module.queue.widget.value.nucleus_id

        nuc_layer = statebuilder.SegmentationLayerConfig(crg.NgLinks.nuc_src, name='nuclear-seg', fixed_ids=[nucleus_id])
        base_sb = statebuilder.StateBuilder([crg.NgLinks.em_layer, crg.NgLinks.seg_layer, nuc_layer])

        json_state = base_sb.render_state(return_as='dict')

        json_state['layers'][0].update({'visible': False})
        json_state['layers'][1].update({'visible': True})
        json_state['layers'][2].update({'visible': False})
        json_state['navigation'].update({"pose": {
            "position": {
                "voxelSize": [4, 4, 40],
                "voxelCoordinates": list((m65mat.Nucleus.Info & stable & {'nucleus_id': nucleus_id}).fetch1('nucleus_x', 'nucleus_y', 'nucleus_z'))
                },
                "orientation": [0, -0.7071067690849304, 0, 0.7071067690849304]
                },
                "zoomFactor": 3499.999999999999
            }
        )
        json_state['perspectiveZoom'] = 2000

        final_sb = statebuilder.StateBuilder([], base_state=json_state)
        
        url = final_sb.render_state('url')

        if from_queue_module:
            with self.nucleus_queue_module.feedback_out:
                wr.clear_output()
                wr.display(wr.HBox([
                    wr.HTML(value=f"<a href={url}> Neuroglancer Link </a>").widget,
                    self.nucleus_queue_module.clear_feedback_button.widget
                ]))
        else:
            return f"<a href={url}> Neuroglancer Link </a>"

    def update_ng_link(self, clear=False):
        if clear:
            self.ng_link_html.widget.value=''
            self.clear_ng_html_button.widget.layout.display='none'
        else:
            self.clear_ng_html_button.widget.layout.display=None

    def on_match_protocol_select(self, protocol_type):
        if protocol_type == "Unit":
            selected_match_protocol = (m65.UnitMatchProtocol & {'protocol_name': self.unit_match_candidate_module.protocol_buttons.widget.value}).fetch1('KEY')

        elif protocol_type == "Nucleus":
            selected_match_protocol = (m65.NucleusMatchProtocol & {'protocol_name': self.nucleus_match_candidate_module.protocol_buttons.widget.value}).fetch1('KEY')

        else:
            self.match_protocol_custom_msg("Protocol type not recognized.")
            return

        self.update_selected_match_protocol(selected_match_protocol=selected_match_protocol, protocol_type=protocol_type)


    def _select_random_match_candidate(self):
        if self.selected_match_protocol is not None:
            if self.selected_match_protocol['protocol_type'] == "Unit":
                restrs = m65.UnitMatchCandidate.protocol_restrs(protocol_name=self.selected_match_protocol['protocol_name'], user_id=self.user_id_field.widget.value)
                random_entry_df = m65.UnitMatchCandidate.get_random_candidate(protocol_name=self.selected_match_protocol['protocol_name'], restrs=restrs)
                

                if len(random_entry_df) > 0:
                    field = (m65.UnitAnatomyFunction & random_entry_df.to_records()).fetch1('field')

                    new_entry = DataType.Unit(scan_session=random_entry_df.scan_session.values[0], scan_idx=random_entry_df.scan_idx.values[0], field=field, unit_id=random_entry_df.unit_id.values[0])
                    result = self.unit_queue_module.add_entry(new_entry)

                    # Post to slack
                    if result == 'added':
                        if self.slack_username is not None:
                            text = f'```{self.slack_username} ({self.user_id_field.widget.value}) loaded scan_session={new_entry.scan_session}, scan_idx={new_entry.scan_idx}, field={new_entry.field}, unit_id={new_entry.unit_id} in {self.selected_match_protocol["protocol_type"]} protocol "{self.selected_match_protocol["protocol_name"]}".```'
                            self.slack_output.post_to_slack_and_user(text, slack_username=self.slack_username)

                else:
                    self.match_protocol_custom_msg('Candidate pool depleted.')

            elif self.selected_match_protocol['protocol_type'] == "Nucleus":
                restrs = m65.NucleusMatchCandidate.protocol_restrs(protocol_name=self.selected_match_protocol['protocol_name'], user_id=self.user_id_field.widget.value)
                random_entry_df = m65.NucleusMatchCandidate.get_random_candidate(protocol_name=self.selected_match_protocol['protocol_name'], restrs=restrs)
                
                if len(random_entry_df) > 0:

                    new_entry = DataType.Nucleus(nucleus_id=random_entry_df.nucleus_id.values[0])
                    result = self.nucleus_queue_module.add_entry(new_entry)

                    # Post to slack
                    if result == 'added':
                        if self.slack_username is not None:
                            text = f'```{self.slack_username} ({self.user_id_field.widget.value}) loaded nucleus_id={new_entry.nucleus_id} in {self.selected_match_protocol["protocol_type"]} protocol "{self.selected_match_protocol["protocol_name"]}".```'
                            self.slack_output.post_to_slack_and_user(text, slack_username=self.slack_username)
                
                else:
                    self.match_protocol_custom_msg('Candidate pool depleted.')

            else:
                self.match_protocol_custom_msg('Protocol type not recognized.')

        else:
            self.match_protocol_custom_msg('Please select a protocol.')
            return
    
    def submit_report(self, text=None, lag=False):
        if lag is not None:
            text = f'```{self.slack_username} ({self.user_id_field.widget.value}) submitted a report that the mult-match tool is lagging.```'
        
        if text is not None:
            self.slack_output.send_direct_message(text, slack_username="spapadop")
            self.slack_output.send_direct_message(text, slack_username="cpapadop")

    def reset_multimatch(self):
        self.clear_field_unit_object()
        self.clear_primary_unit()
        self.clear_nucleus_object()
        self.clear_selected_nucleus()
        self.update_ng_link(clear=True)
        self.clear_match_select(clear_all=True)
        self.clear_match_note()
        self.feedback_out.clear_output()
        self.close_figure()
