from pathlib import Path
import inspect
import pandas as pd
import wridgets as wr
import wridgets.app as wra
import datajoint_plus as djp

# from microns_coregistration_api.schemas import dashboard as db
from microns_coregistration_api.schemas import minnie65_manual_match as m65man
from microns_morphology_api.schemas import minnie65_auto_proofreading as m65auto
from microns_nda_api.schemas import minnie_nda as nda
from microns_utils.datetime_utils import current_timestamp


class SchemaViewer(wra.App):
    schema_mapping = {
        nda.schema.database: nda
    }
    
    store_config = [
        ('dropdown_default_option', ['< Select >']),
        ('schemas', schema_mapping),
        ('schema_options', list(schema_mapping.keys())),
        'selected_schema_name',
        'selected_schema',
        'table_options',
        'selected_table',
        'filename',
        'df',
    ]
    
    def make(self, **kwargs):
        schema_label = wra.Label(text='Schema', name='SchemaLabel')
        schema_dropdown = wra.Dropdown(options = self.dropdown_default_option + self.schema_options, on_interact=self.set_schema, name='SchemaDropdown')
        table_label = wra.Label(text='Table', name='TableLabel')
        table_dropdown = wra.Dropdown(name='TableDropdown', on_interact=self.set_table, layout={'width': '200px'})
        download_table_button = wra.Button(name='DownloadTableButton', on_interact=self.download_table, description='Download CSV')
        self.core = (schema_label + schema_dropdown) - (table_label + table_dropdown + download_table_button)
    
    def set_schema(self):
        schema_name = self.children.SchemaDropdown.get1('value')
        if schema_name != self.dropdown_default_option[0]:
            self.selected_schema_name = schema_name
            self.selected_schema = self.schema_mapping[schema_name]
            self.set_table_options(self.selected_schema.schema.list_tables())
            
        else:
            self.selected_schema_name = None
            self.selected_schema = None
            self.children.TableDropdown.set(options=())
        
    def set_table_options(self, tables=None):
        if tables is not None:
            self.children.TableDropdown.set(options=self.dropdown_default_option + tables)
        else:
            self.children.TableDropdown.set(options=self.dropdown_default_option)
        
    def set_table(self):
        if self.selected_schema is not None:
            table_name = self.children.TableDropdown.get1('value')
            if table_name != self.dropdown_default_option[0]:
                self.selected_table = table_name
                wr.display(self.selected_schema.schema.free_table(self.selected_table))
            else:
                self.selected_table = None
    
    def download_table(self):
        if self.selected_table is not None:
            table = [table for name, table in inspect.getmembers(self.selected_schema) if getattr(table, 'is_user_table', False) and table.table_name==self.selected_table][0]
            self.df = pd.DataFrame(table.fetch())
            dldir = Path('./downloaded_tables')
            dldir.mkdir(exist_ok=True)
            self.filename = dldir.joinpath(self.selected_table + '_' + current_timestamp('US/Central', fmt="%Y-%m-%d_%H-%M-%S")).with_suffix('.csv')
            self.df.to_csv(self.filename)
            wr.HTML(value=(f"""<a href="{self.filename}" download="{self.filename.name}">Download Link</a>""")).display()
            