import datajoint_plus as djp
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt

import wridgets as wr
from wridgets import Output, HBox, VBox, Label, Layout, HTML, display, clear_output

logger = djp.getLogger(__name__)

def DBox(items, dims='auto'):
    n_items = len(items)
    
    if dims=='auto':
        n_cols = 4
        n_rows = np.ceil(n_items / n_cols).astype(np.int)

    else:
        n_rows, n_cols = dims
        if n_rows * n_cols < n_items:
            logger.warning(f"Specified dimensions: {n_rows, n_cols} won't fit all {n_items} items")

    HBoxs = []
    for r in range(n_rows):
        HBoxs.append(
            HBox(
                *[items[slice(n_cols*r, n_cols*(r+1))]]
            )
        )
    return VBox([*HBoxs])


def namedtuple_with_defaults(nt, defaults=None, skip_extra_fields=False):
    """
    from Christos
    """
    if isinstance(defaults, dict):
        nt.__new__.__defaults__ = tuple(defaults.pop(field) for field in nt._fields)
        if not skip_extra_fields:
            if len(defaults) > 0:
                raise ValueError(f'For namedtuple {nt}, these defaults were supplied that don\'t correspond to fields: {defaults}')
    elif isinstance(defaults, list):
        nt.__new__.__defaults__ = tuple(defaults)
    else:
        nt.__new__.__defaults__ = (defaults,) * len(nt._fields)
    return nt


class StackByDepthLoader:
    def __init__(self, parent_table, depth_table, stack_key, stack_npy_path=None, load_source='datajoint', depth:int=None, padding:int=0, depth_range=None, load_mode='cache'):
        self.parent_table = parent_table
        self.dj_table = depth_table
        self.stack_key = stack_key
        self.stack_npy_path = stack_npy_path
        self.load_source = load_source
        self.load_mode = load_mode
        self.initialize_stack()
        self.get_stack_images(depth=depth, padding=padding, depth_range=depth_range, load_mode=load_mode)
    
    def initialize_stack(self):
        self.stack_x, self.stack_y, self.stack_z = self.get_stack_dimensions_in_voxels()
        self.empty_stack = np.empty((self.stack_z, self.stack_y, self.stack_x))
        self.loaded_stack = self.empty_stack
        self.loaded_stack_depth_tracker = np.zeros(self.stack_z)
        if self.load_source == 'numpy':
            self.mmap = np.load(self.stack_npy_path, mmap_mode='r')
                  
    def _prepare_stack_chunk(self, depth:int=None, padding:int=0, depth_range=None):
        """ 
        Prepares stack chunk. Provide depth or depth_range but not both. If no arguments are provided, no action is taken. 
        
        :param depth (optional): the depth (z-axis) to load around. 
        :param padding  (optional): the number of images that will be loaded both above and below depth value. Default is 0. Note: if depth not provided, padding will be ignored. 
        :param depth_range  (optional): the range of depths to load. All images from the min value of depth_range up to but not including the max value of depth_range will be loaded.
        """
        assert ~np.logical_and(depth is not None, depth_range is not None), 'Provide either depth/ padding or depth_range, but not both'
        
        if depth is not None:
            assert isinstance(depth, (int, np.integer)) and depth >=0, 'depth must be an integer greater than or equal to 0'
            assert isinstance(padding, (int, np.integer)) and padding >=0, 'padding must be an integer greater than or equal to 0'
            
            depth_min = np.maximum(depth - padding, 0)
            depth_max = np.minimum(depth + padding + 1, self.loaded_stack_depth_tracker.size - 1)
            
            return depth_min, depth_max
            
        if depth_range is not None:
            depth_min = np.maximum(np.min(depth_range), 0)
            depth_max = np.minimum(np.max(depth_range), self.loaded_stack_depth_tracker.size)
            
            assert isinstance(depth_min, (int, np.integer)) and depth_min >=0, 'depth_range must contain integers greater than or equal to 0'
            assert isinstance(depth_max, (int, np.integer)) and depth_max >=0, 'depth_range must contain integers greater than or equal to 0'
            
            return depth_min, depth_max
    
    def get_stack_images(self, depth:int=None, padding:int=0, depth_range=None, load_mode=None):
        depth_range = self._prepare_stack_chunk(depth=depth, padding=padding, depth_range=depth_range)
        
        if depth_range is not None:
            depth_min, depth_max = depth_range

            if load_mode is None:
                load_mode = self.load_mode

            not_loaded = np.where(self.loaded_stack_depth_tracker==0)[0]
            depths_to_load = [d for d in not_loaded if d >= depth_min and d < depth_max]
            
            if depths_to_load:
                if self.load_source == 'datajoint':
                    depth_restr = [{'depth': d} for d in depths_to_load]
                    images = np.stack((self.dj_table & self.stack_key & depth_restr).fetch('image'))

                elif self.load_source == 'numpy':
                    images = np.stack([self.mmap[depths_to_load]])
                
                else:
                    raise Exception('load_source not recognized. Choose "datajoint" or "numpy')

                if load_mode == 'cache':
                    self.loaded_stack[depths_to_load] = images
                    self.loaded_stack_depth_tracker[depths_to_load] = 1

                elif load_mode == 'view':
                    return np.squeeze(images)

                else:
                    raise Exception('"load_mode" not recognized. Choose "load" or "view". ')
    
    def load_stack_all(self):
        """ 
        Loads entire stack
        
        :param parent_table: table where entire stack is stored
        """
        if self.load_source == 'datajoint':
            self.loaded_stack = (self.parent_table & self.stack_key).fetch1('stack')
            self.loaded_stack_depth_tracker[:] = 1
        elif self.load_source == 'numpy':
            self.loaded_stack = np.load(self.stack_npy_path)
            self.loaded_stack_depth_tracker[:] = 1
        else:
            raise Exception('load_source not recognized. Choose "datajoint" or "numpy')
    
    def reset_stack(self):
        """
        Resets the loaded stack to the empty stack
        """
        self.loaded_stack = self.empty_stack
    
    
    def get_stack_dimensions_in_voxels(self):
        """
        Returns voxel dimensions of a resized stack in x, y, z format

        :param resized_stack_key: key to restrict Stack2PResized
        
        """

        attrs = np.reshape([x + y for x in ['resolution', 'length'] for y in ['_x', '_y', '_z']], (2,3))
        resolutions = np.stack((self.parent_table & self.stack_key).fetch(*attrs[0])) 
        lengths = np.stack((self.parent_table & self.stack_key).fetch(*attrs[1]))
        return (resolutions * lengths).squeeze().astype(np.int)
    
    def check_if_loaded(self, depth):
        """
        Checks if provided depth is already loaded. 
        
        :param depth: depth to check if already loaded
        
        returns bool
        """
        
        return bool(self.loaded_stack_depth_tracker[depth])


class Fig:
    def __init__(self, plot_functions, output=None, ax_layout='auto', fig_kws={}, plot_kws={}, initialize=True, scroll_up_action=None, scroll_down_action=None, button_press_action=None, draw_action=None, pick_action=None, resize_action=None, **kwargs):
        self.plot_functions = plot_functions
        self.output = output
        self.ax_layout = ax_layout
        self.fig_kws = fig_kws
        self.plot_kws = plot_kws
        self.is_initialized = False
        self.scroll_up_action = scroll_up_action
        self.scroll_down_action = scroll_down_action
        self.button_press_action = button_press_action
        self.draw_action = draw_action
        self.pick_action = pick_action
        self.resize_action = resize_action
        self.defaults = namedtuple('defaults', kwargs.keys())(*kwargs.values())
          
        if initialize:
            self.initialize()
            
    def initialize(self):
        self.n_axes = len(self.plot_functions)
        
        if self.ax_layout=='auto':
            self.n_rows = 1
            self.n_cols = np.ceil(self.n_axes / self.n_rows).astype(np.int)

        else:
            self.n_rows, self.n_cols = self.dims
            assert (self.n_rows * self.n_cols) == self.n_axes, f"Specified dimensions: {self.n_rows, self.n_cols} won't fit {self.n_axes} axes"
        
        if self.output is not None:
            with self.output:
                self._initialize()
        else:
            self._initialize()
                
    def _initialize(self):
        self.fig, self.axes = plt.subplots(self.n_rows, self.n_cols, **self.fig_kws)
        self.axes_function_mapping = {ax: f for ax, f in zip(self.fig.axes, self.plot_functions)}
        self.is_initialized=True
        self.update_plot()
    
    def update_plot(self, plot_kws={}):
        if not self.is_initialized:
            print("Plot is not initialized. Run 'initialize()' method first.")

        if not plot_kws:
            plot_kws = self.plot_kws
            
        for ax, f in self.axes_function_mapping.items():
            f(ax, **plot_kws)

    def add_scroll_event(self, scroll_up_action=None, scroll_down_action=None):
        if scroll_up_action is not None:
            self.scroll_up_action = scroll_up_action
        
        if scroll_down_action is not None:
            self.scroll_down_action = scroll_down_action
            
        assert self.scroll_up_action is not None and self.scroll_down_action is not None, 'Provide "scroll_up_action" and "scroll_down_action" functions to apply scroll event'

        def key_event(e):
            if e.button == 'up':
                self.scroll_up_action(e)
            elif e.button == 'down':
                self.scroll_down_action(e)
            else:
                return
    
        self.fig.canvas.mpl_connect('scroll_event', key_event)
        
    def add_button_press_event(self, button_press_action=None):
        if button_press_action is not None:
            self.button_press_action = button_press_action
            
        assert self.button_press_action is not None, 'Provide "button_press_action" function to apply pick event'
        
        def key_event(e):
            self.button_press_action(e)
    
        self.fig.canvas.mpl_connect('button_press_event', key_event)
    
    def add_pick_event(self, pick_action=None):
        if pick_action is not None:
            self.pick_action = pick_action
            
        assert self.pick_action is not None, 'Provide "pick_action" function to apply pick event'
        
        def key_event(e):
            self.pick_action(e)
    
        self.fig.canvas.mpl_connect('pick_event', key_event)
    
    def add_draw_event(self, draw_action=None):
        if draw_action is not None:
            self.draw_action = draw_action
            
        assert self.draw_action is not None, 'Provide "draw_action" function to apply pick event'
        
        def key_event(e):
            self.draw_action(e)
    
        self.fig.canvas.mpl_connect('draw_event', draw_action)

    def add_resize_event(self, resize_action=None):
        if resize_action is not None:
            self.resize_action = resize_action
        
        assert self.resize_action is not None, "Provide 'resize_action' function to apply resize event"

        def key_event(e):
            self.resize_action(e)

        self.fig.canvas.mpl_connect('resize_event', resize_action)


class DatajointTableWidget():
    def __init__(self, dj_table, archive_table=None, show=False, table_on=True, enable_mod=False, field_dims='auto', **kwargs):
        self.dj_table = dj_table
        self.archive_table = archive_table
        self.restricted_table = self.dj_table()
        self.restrs = []
        self.dtype_mapping = self.dj_table.heading.as_dtype.fields
        self.table_on=table_on
        self.enable_mod = enable_mod
        self.field_dims = field_dims
        self.defaults = kwargs
        
        # VIEW BUTTONS
        self.dj_table_out = wr.Output()
        self.show_table_button = wr.Button(description="Show Table", on_interact=self.show_table, output=self.dj_table_out)
        self.hide_table_button = wr.Button(description="Hide Table", on_interact=self.hide_table)
        self.restrict_button = wr.Button(description="Apply Restrictions", on_interact=self.apply_restrs)
        self.reset_restrs_button = wr.Button(description="Reset Restrictions", on_interact=self.reset_restrs)
        self.restrs_out = wr.Output()
        self.clear_fields_button = wr.Button(description="Clear Fields", on_interact=self.clear_fields)
        
        # MODIFY BUTTONS
        self.modification_out = wr.Output()
        self.enable_mod_check = wr.Checkbox(description="Enable Modification", on_true=self.toggle_mod_buttons, on_false=self.toggle_mod_buttons, indent=False)
        self.insert_button = wr.Button(description="Insert", on_interact=self.insert, output=self.modification_out, disabled=True)
        self.update_button = wr.Button(description="Update", on_interact=self.update, output=self.modification_out, disabled=True)
        self.archive_button = wr.Button(description="Archive", on_interact=self.archive, output=self.modification_out, disabled=True)
        self.archive_note_field = wr.Text(value='', description='Archive Note: ', disabled=True)
        self.delete_button = wr.Button(description="Delete", on_interact=self.delete, output=self.modification_out, disabled=True)
        self.clear_mod_button = wr.Button(description="Clear Feedback", on_interact=self.modification_out.clear_output, disabled=True)
        
        self.fields = {}
        for name in self.dj_table.heading.names:
            value = str(self.defaults[name]) if name in self.defaults else ''
            placeholder = self.dj_table.heading.attributes[name].comment
            style = {'description_width': 'max-content'}
#             layout = {'border':'0.5px solid gray'} if name in self.dj_table.heading.primary_key else {}
            self.fields[name] = wr.Text(value=value, description=name, placeholder=placeholder, \
                                continuous_update=True, style=style, layout={})
        
        if show:
            self.show()
            
    def show(self):
        fields = []
        for v in self.fields.values():
            fields.append(v.widget)
        
        view_module = wr.HBox([
                wr.Label(value=rf'$\large \text{{Restrict:}} $'),
                self.show_table_button.widget,
                self.hide_table_button.widget,
                self.restrict_button.widget, 
                self.reset_restrs_button.widget,
                self.clear_fields_button.widget
            ])
            
        modify_module = wr.HBox([
            wr.Label(value=rf'$\large \text{{Modify: }} $'),
            self.insert_button.widget,
            self.update_button.widget,
            self.delete_button.widget,
            self.archive_button.widget,
            self.archive_note_field.widget,
            self.clear_mod_button.widget
            
        ])
        
        options_module = wr.HBox([
            wr.Label(value=rf'$\large \text{{Options: }} $'),
            self.enable_mod_check.widget
        ])
        
        display(
            view_module,
            modify_module,
            options_module,
            self.modification_out,
            DBox(fields, self.field_dims), 
            wr.Label(value=rf'$\large \text{{Restrictions Applied:}} $'), 
            self.restrs_out,
            self.dj_table_out
            
        )
        
        self.show_restrs()
        
        if self.table_on:
            self.show_table_button.widget.click()
        
        if self.enable_mod:
            self.enable_mod_check.widget.value = True
        
    def show_table(self):
        display(self.restricted_table)
        self.table_on=True
        
    def hide_table(self):
        self.dj_table_out.clear_output()
        self.table_on=False
    
    def apply_restrs(self):
        source = self.dj_table()
        
        self.restrs = []
        for k, v in self.fields.items():
            if v.widget.value != '':
                dtype = self.dtype_mapping[k][0]
                restr = {k: dtype.type(v.widget.value)}
                source &= restr
                self.restrs.append(restr)
        
        self.restricted_table = source
        
        if self.table_on:
            self.show_table_button.widget.click()
        
        self.show_restrs()
    
    def reset_restrs(self):
        self.restricted_table = self.dj_table()
        self.restrs = []
        self.restrs_out.clear_output()
        
        if self.table_on:
            self.show_table_button.widget.click()
            
        self.show_restrs()
    
    def show_restrs(self):
        self.restrs_out.clear_output()
        with self.restrs_out:
            display(self.restrs)
        
    def clear_fields(self):
        for v in self.fields.values():
            v.widget.value = ''
            
    def toggle_mod_buttons(self):
        if self.enable_mod_check.widget.value:
            self.insert_button.widget.disabled = False
            self.delete_button.widget.disabled = False
            self.archive_button.widget.disabled = False
            self.archive_note_field.widget.disabled = False
            self.update_button.widget.disabled = False
            self.clear_mod_button.widget.disabled = False
        else:
            self.insert_button.widget.disabled = True
            self.delete_button.widget.disabled = True
            self.archive_button.widget.disabled = True
            self.archive_note_field.widget.disabled = True
            self.update_button.widget.disabled = True
            self.clear_mod_button.widget.disabled = True
            
    def insert(self):
        insert_dict = {}
        for k, v in self.dtype_mapping.items():
            input_value = self.fields[k].widget.value
            if input_value != '':
                insert_dict[k] = v[0].type(input_value)
        self.dj_table.insert1(insert_dict)
        print(f'Inserted: {insert_dict}')
        
        if self.table_on:
            self.show_table_button.widget.click()
    
    def delete(self):
        if len(self.restricted_table)!=1:
            print('Exactly one entry required to delete. Restrict table to a single entry. ')
            return 
        else:
            deleted_dict = self.restricted_table.fetch1()
            self.restricted_table.delete_quick()
            print(f'Entry deleted successfully: {deleted_dict}')
            
        if self.table_on:
            self.show_table_button.widget.click()
    
    def update(self):
        if len(self.restricted_table)!=1:
            print('Exactly one entry required to update. Restrict table to a single entry. ')
            return 
        
        for k in self.dj_table.heading.secondary_attributes:
            field_value = self.fields[k].widget.value
            if field_value != '':
                dtype = self.dtype_mapping[k][0]
                input_value = dtype.type(field_value)
                update_dict = {k: input_value}
                old_value = self.restricted_table.fetch1(k)
                old_dict = {k: old_value}
                self.restricted_table._update(k, input_value)
                print(f'Successfully update from {old_dict} to {update_dict}.')
        
        if self.table_on:
            self.show_table_button.widget.click()

    def archive(self):
        if self.archive_table is None:
            print('No Archive table specified. No action taken.')
        else:
            if len(self.restricted_table)!=1:
                print('Exactly one entry required to delete. Restrict table to a single entry. ')
                return 
            
            self.restricted_table.delete_quick()
            print('Entry archived. Deletion from Main table successful.')
            print('Archive successful.')
        
        if self.table_on:
            self.show_table_button.widget.click()
