"""
Utility for training humans on cell type classification.
"""

from pathlib import Path
from nglui import statebuilder

import datajoint_plus as djp
import numpy as np
import pandas as pd
import wridgets as wr
import wridgets.app as wra

from microns_materialization_api.schemas import \
    minnie65_materialization as m65mat
from microns_materialization_api.dashboard.applets import \
    SetMaterializationApp
from microns_morphology_api.schemas import \
    minnie65_auto_proofreading as m65auto
import microns_utils.ng_utils as ngu

# from ..schemas import dashboard as db
from ..schemas import cell_type_trainer as ctnr
# from ..transform import coregistration as crg

schema = ctnr.schema
config = ctnr.config

logger = djp.getLogger(__name__)

class CellTypeTrainer(wra.App):
    store_config = [
        'ver',
        ('autoproof_source', m65auto.AutoProofreadNeuron & 'baylor_cell_type_after_proof=external_cell_type' & 'baylor_cell_type_exc_probability_after_proof>0.95'), 
        ('segment_source', m65mat.Segment.Nucleus),
        'segment_options',
        'segment',
        'segment_cell_type',
        'segment_cell_type_prob',
    ]
    
    def make(self, **kwargs):
        self._user = kwargs.get('user')
        self._user_app = kwargs.get('user_app')
        
        # core apps
        header_app = wra.App(
            wra.Label(text='Cell Type Trainer', fontsize=2, name='HeaderLabel') - \
            wra.Link(link_kws=dict(src='https://docs.google.com/presentation/d/1Gruve0SmsBkzVOpJA_IWXGIj88LkmFDzjBQSIsjuLb8/edit?usp=sharing', text='Training Link'), name='HeaderTrainingLink'),
            name='HeaderApp'
        )

        mat_app = SetMaterializationApp(on_select=self.select_mat, name='MatApp')
                                             
        seg_app = wra.App(
                wra.Label(text='Segment ID', name='SegLabel') + \
                wra.Field(name='SegField', continuous_update=True, on_interact=self.validate_segment) + \
                wra.Button(name='RandomSegButton', description='Random', on_interact=self.get_random_segment) + \
                wra.ToggleButton(name='SegSelectButton', description='Select', button_style='info', on_interact=self.select_segment), 
            name='SegApp', 
            propagate=True
        )
        seg_app.set(disabled=True)
        
        ng_app = wra.App(
                wra.Label(text='Neuroglancer', name='NgLabel') + \
                wra.Link(name='NgLink'), 
            name='NgApp', 
            propagate=True
        )
        res_app = wra.App(
                (wra.Label(text='Submit', name='ResLabel') - \
                wra.SelectButtons(
                    wridget_type='RadioButtons',
                    options=[
                        ('Excitatory', 'excitatory'),
                        ('Inhibitory', 'inhibitory'),
                        ('Non-neuron', 'non_neuron'),
                        ('Uncertain', 'uncertain'),
                        ('Flag', 'flagged')
                    ], name='ResButtons')) + \
                (wra.Label(text='Notes') - wra.Field(wridget_type='Textarea', layout={'width':'initial', 'height': 'initial'}, name='ResNote') - \
                wra.Button(description='Submit', on_interact=self.submit, name='ResSubmit', button_style='info', disabled=True)),
            name='ResApp',
            propagate=True
        )

        # build app
        self.core = header_app - mat_app - seg_app - ng_app - res_app
        
        if kwargs.get('display'):
            self.display()
    
    @property
    def user(self):
        return self._user_app.user if self._user is None else self._user
    
    def set_ver(self, ver=None):
        self.ver = ver
        
    def set_segment(self, segment=None):
        self.segment = segment
        if segment is not None:
            result_rel = self.autoproof_source & {'segment_id': segment}
            self.segment_cell_type = result_rel.fetch1('baylor_cell_type_after_proof')
            self.segment_cell_type_prob = np.round(result_rel.fetch1('baylor_cell_type_exc_probability_after_proof'), decimals=2)
        else:
            self.segment_cell_type = None
            self.segment_cell_type_prob = None
    
    def get_segment_options(self, ver):
        segment_id_rel = djp.U('segment_id') & (self.segment_source & {'ver': self.ver} & self.autoproof_source)
        return segment_id_rel.fetch('segment_id').astype(str).tolist()
    
    def set_segment_options(self, segment_options=None):
        self.segment_options = segment_options
    
    def clear_segment(self):
        self.children.SegSelectButton.set(description='Select')
        self.children.ResButtons.set(value='excitatory')
        self.children.SegSelectButton.set(value=False)
        self.children.SegField.set(value='')
        self.set_segment()
        self.set_neuroglancer_link()
        self.children.ResNote.set(value='')
        self.children.SegApp.set(disabled=False, exclude=self.children.SegSelectButton.name)
        self.children.ResSubmit.set(disabled=True)
        
    def get_random_segment(self):
        if self.segment_options is not None:
            segment = np.random.permutation(self.segment_options)[0]
            self.children.SegField.set(value=segment)
        
    def select_mat(self):
        if self.children.MatSelectButton.get1('value'):
            self.children.MatSelectButton.set(description='Deselect')
            self.children.MatApp.set(disabled=True, exclude=self.children.MatSelectButton.name)
            self.children.SegApp.set(disabled=False, exclude=self.children.SegSelectButton.name)
            self.set_ver(self.children.MatDropdown.get1('value'))
            self.set_segment_options(self.get_segment_options(self.ver))
        else:
            self.set_ver()
            self.set_segment_options()
            self.clear_segment()
            self.children.MatSelectButton.set(description='Select')
            self.children.MatApp.set(disabled=False)
            self.children.SegApp.set(disabled=True)
    
    def validate_segment(self):
        if self.children.SegField.get1('value') in self.segment_options:
            self.children.SegApp.set(disabled=False, include=self.children.SegSelectButton.name)
        else:
            self.children.SegApp.set(disabled=True, include=self.children.SegSelectButton.name)
    
    def select_segment(self):
        if self.children.SegSelectButton.get1('value'):
            seg_value = self.children.SegField.get1('value')
            if seg_value:
                if seg_value in self.segment_options:
                    self.set_segment(seg_value)
                    self.children.SegApp.set(disabled=True, exclude=self.children.SegSelectButton.name)
                    self.children.SegSelectButton.set(description='Clear')
                    self.set_neuroglancer_link(seg_value)
                    self.children.ResSubmit.set(disabled=False)
                else:
                    self.clear_segment()
                    self.message('segment not a valid option.')
        else:
            self.clear_segment()
                
    def set_neuroglancer_link(self, segment=None):
        if segment is not None:
            position = np.stack((m65mat.Nucleus.Info & {'ver': self.ver} & {'segment_id':segment}).fetch1('nucleus_x', 'nucleus_y', 'nucleus_z'), -1)
            if not position.tolist():
                position = None
            viewer = ngu.view_segments_in_neuroglancer(segments=segment, position=position)
            self.children.NgLink.make(link_kws=dict(src=viewer.as_url(), text='Link'))
        else:
            self.children.NgLink.make()
    
    def submit(self):
        def submit_to_database():
            with wr.Output():
                ctnr.UserInput.Submission.on_input(self.user, submission)
                ctnr.Submission.Maker.populate()
            self.clear_segment()
            feedback = {
                'Your choice': user_choice,
                'Solution': segment_cell_type,
                'Probability': str(segment_cell_type_prob)
            }
            self.msg(
                wra.Label(text='Submitted. <p>' + " <p> ".join(['<b>' + k + '</b>' + ': ' + v for k, v in feedback.items()])) - self.clear_button,
                with_clear_button=False
            )

        segment = self.segment
        segment_cell_type = self.segment_cell_type
        segment_cell_type_prob = self.segment_cell_type_prob
        user_choice = self.children.ResButtons.get1('value')
        user_note = self.children.ResNote.get1('value')
        
        submission = {
            'segment_id': segment, 
            'user_choice': user_choice, 
            'user_note': user_note, 
            'segment_cell_type': segment_cell_type, 
            'segment_cell_type_prob': segment_cell_type_prob
        }
        
        self.msg(
            wra.Label(text='Confirm submission:') + wra.Button(description='Confirm', button_style='success', on_interact=submit_to_database) + self.clear_button - \
            wra.Label(text=" <p> ".join(['<b>' + k + '</b>'+ ': ' + v for k, v in submission.items() if k in ['segment_id', 'user_choice', 'user_note']])), with_clear_button=False
        )