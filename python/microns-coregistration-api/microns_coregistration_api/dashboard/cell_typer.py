"""
Utility for training humans on cell type classification.
"""

from pathlib import Path
import time

import datajoint_plus as djp
import numpy as np
import pandas as pd
import wridgets as wr
import wridgets.app as wra

from microns_materialization_api.schemas import \
    minnie65_materialization as m65mat
from microns_materialization_api.dashboard.apps import \
    MaterializationManager
from microns_morphology_api.schemas import \
    minnie65_auto_proofreading as m65auto
from microns_dashboard_api.apps import \
    ProtocolManager, DataJointTableApp
import microns_utils.ng_utils as ngu
from microns_utils.misc_utils import classproperty

from ..schemas import cell_typer as database

logger = djp.getLogger(__name__)




class CellTyper(wra.App):
    store_config = [
        'ver',
        'protocol_id',
        'protocol_name',
        'segment',
        'segment_label',
        'on_submit',
        'on_submit_kws'
    ]
    
    def make(self, **kwargs):
        self._user = kwargs.get('user')
        self._user_app = kwargs.get('user_app')
        self.on_submit = kwargs.get('on_submit')
        self.on_submit_kws = {} if kwargs.get('on_submit_kws') is None else kwargs.get('on_submit_kws')

        # core apps
        header_app = wra.App(
            wra.Label(text='Cell Typer', fontsize=2, name='HeaderLabel') - \
            wra.Link(link_kws=dict(src='https://docs.google.com/presentation/d/1Gruve0SmsBkzVOpJA_IWXGIj88LkmFDzjBQSIsjuLb8/edit?usp=sharing', text='Training Link'), name='HeaderTrainingLink'),
            name='HeaderApp'
        )

        materialization_app = MaterializationManager(on_select=self.set_ver, set_button='stable', name='MatApp')

        protocol_app = ProtocolManager(source=database.Protocol.Manager, on_select=self.set_protocol, name='ProtocolApp')
                                             
        segment_app = wra.App(
                wra.Label(text='Segment ID', name='SegLabel') + \
                wra.Field(name='SegField', continuous_update=True, on_interact=self.on_field_change) + \
                wra.Button(name='RandomSegButton', description='Random', on_interact=self.get_random_segment) + \
                wra.ToggleButton(name='SegSelectButton', description='Select', button_style='info', on_interact=self.set_segment), 
            name='SegApp', 
            propagate=True
        )
        segment_app.set(disabled=True)
        
        neuroglancer_app = wra.App(
                wra.Label(text='Neuroglancer', name='NgLabel') + \
                wra.Link(name='NgLink'), 
            name='NgApp', 
            propagate=True
        )
        result_app = wra.App(
                (wra.Label(text='Submit', name='ResLabel') - \
                wra.SelectButtons(
                    wridget_type='RadioButtons',
                    options=[
                        ('Excitatory', 'excitatory'),
                        ('Inhibitory', 'inhibitory'),
                        ('Non-neuron', 'non_neuron'),
                        ('Uncertain', 'uncertain'),
                        ('Flag', 'flagged')
                    ], name='ResButtons', layout={'width': '150px'})) + \
                (wra.Label(text='Notes') - wra.Field(wridget_type='Textarea', layout={'width':'initial', 'height': 'initial'}, name='ResNote') - \
                wra.Button(description='Submit', on_interact=self.submit, name='ResSubmit', button_style='info', disabled=True)),
            name='ResApp',
            propagate=True
        )

        # build app
        self.core = header_app - materialization_app - protocol_app - segment_app - neuroglancer_app - result_app
        
        self.children.MatSelectButton.set(value=True)

        if kwargs.get('display'):
            self.display()
    
    @property
    def user(self):
        return self._user_app.user if self._user is None else self._user
    

    def set_ver(self):
        if self.children.MatSelectButton.get1('value'):
            self.children.MatSelectButton.set(description='Clear')
            self.children.MatApp.set(disabled=True, exclude=self.children.MatSelectButton.name)
            self.ver = self.children.MatApp.value
            self.toggle_seg_field(enable=True)
        else:
            self.children.MatSelectButton.set(description='Select')
            self.ver = None
            self.clear_segment()
            self.toggle_seg_field()
            self.children.MatApp.set(disabled=False)
            

    def set_protocol(self):
        if self.children.ProtocolSelectButton.get1('value'):
            self.children.ProtocolSelectButton.set(description='Clear')
            self.children.ProtocolApp.set(disabled=True, exclude=self.children.ProtocolSelectButton.name)
            self.protocol_id = self.children.ProtocolApp.protocol_id
            self.protocol_name = self.children.ProtocolApp.protocol_name
            self.toggle_seg_field(enable=True)
        else:
            self.children.ProtocolApp.set(disabled=False)
            self.children.ProtocolSelectButton.set(description='Select')
            self.protocol_id = None
            self.protocol_name = None
            self.clear_segment()
            self.toggle_seg_field()
    
    def toggle_seg_field(self, enable=False):
        if enable:
            if self.ver is not None and self.protocol_id is not None:
                self.children.SegApp.set(disabled=False, exclude=self.children.SegSelectButton.name) 
        else:
            self.children.SegApp.set(disabled=True)

    def get_random_segment(self):
        if 'train' in self.protocol_name:
            subtract = None
        else:
            subtract = database.Submission.CellType & {'user': self.user, 'protocol_id': self.protocol_id}
        segment = database.Protocol.get(
            {'protocol_id': self.protocol_id, 'ver': self.ver}, 
            n=1, 
            subtract=subtract
        ).segment_id.values
        if len(segment) == 1:
            segment = str(segment[0])
            self.children.SegField.set(value=segment)
        else:
            self.msg('No segments remaining.')
            

    def on_field_change(self):
        if self.children.SegField.get1('value'):
            self.children.SegSelectButton.set(disabled=False)
        else:
            self.children.SegSelectButton.set(disabled=True)

    def set_segment(self):
        if self.children.SegSelectButton.get1('value'):
            segment_df = database.Protocol.get({'protocol_id': self.protocol_id, 'ver': self.ver, 'segment_id': self.children.SegField.get1('value')}, n=1)
            if len(segment_df) == 1:
                self.children.SegSelectButton.set(description='Clear')
                self.segment = str(segment_df.segment_id.values[0])
                self.segment_label = segment_df.cell_type.values[0]
                self.set_neuroglancer_link(self.segment)
                self.children.ResSubmit.set(disabled=False)
                self.children.SegApp.set(disabled=True, exclude=self.children.SegSelectButton.name)
            else:
                self.children.SegSelectButton.set(value=False)
                self.msg('Segment ID not found.')
                
        else:
            self.clear_segment()

    def clear_segment(self):
        self.segment = None
        self.segment_label = None
        self.children.SegSelectButton.set(value=False)
        self.children.SegSelectButton.set(description='Select')
        self.children.ResButtons.set(value='excitatory')
        self.children.SegField.set(value='')
        self.set_neuroglancer_link()
        self.children.ResNote.set(value='')
        self.children.ResSubmit.set(disabled=True)
        self.children.SegApp.set(disabled=False, exclude=self.children.SegSelectButton.name)
                
    def set_neuroglancer_link(self, segment=None):
        if segment is not None:
            position = np.stack((m65mat.Nucleus.Info & {'ver': self.ver} & {'segment_id':segment}).fetch1('nucleus_x', 'nucleus_y', 'nucleus_z'), -1)
            if not position.tolist():
                position = None
            viewer = ngu.view_segments_in_neuroglancer(segments=[segment], position=position)
            self.children.NgLink.make(link_kws=dict(src=viewer.as_url(), text='Link'))
        else:
            self.children.NgLink.make()
    
    def submit(self):
        def submit_to_database():
            with wr.Output():
                event = database.Event.log_event('cell_type_submission', {'user': self.user}, data=submission)
            submission_id = (database.Submission.Maker * database.Submission.CellType() & {'event_id': event.id}).fetch1('submission_id')
            self.clear_segment()
            feedback = {
                    'Submission ID': submission_id,
                    'Segment ID': segment,
                    'Your choice': user_choice,
                }
            if 'train' in protocol_name:
                feedback.update({
                    'Label': segment_label,
                })
            
            feedback.update({'Note': note})

            # def flag_submission():
            #     with wr.Output():
            #         event = database.Event.log_event('flagged_submission', {'user': self.user}, data={'segment_id': segment})

            self.msg(
                wra.Label(text=" <p> ".join(['<b>' + k + '</b>' + ': ' + v for k, v in feedback.items()])) - self.clear_button, with_clear_button=False
            )

            if self.on_submit is not None:
                self.on_submit(submission_id)

        protocol_id = self.protocol_id
        protocol_name = self.protocol_name
        segment = self.segment
        segment_label = self.segment_label
        user_choice = self.children.ResButtons.get1('value')
        note = self.children.ResNote.get1('value')
        
        submission = {
            'protocol_id': protocol_id,
            'segment_id': segment, 
            'user_choice': user_choice, 
            'note': note if note else None, 
            'cell_type_label': segment_label, 
        }
        
        self.msg(
            wra.Label(text='Confirm submission:') + wra.Button(description='Confirm', button_style='success', on_interact=submit_to_database) + self.clear_button - \
            wra.Label(text=" <p> ".join(['<b>' + k + '</b>'+ ': ' + str(v) for k, v in submission.items() if k in ['segment_id', 'user_choice', 'note']])), with_clear_button=False
        )