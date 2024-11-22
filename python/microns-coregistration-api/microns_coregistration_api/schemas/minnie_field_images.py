import datajoint as dj
import datajoint_plus as djp

from ..config import minnie_field_images as config

from microns_nda_api.schemas import minnie_nda as nda
import microns_utils.datajoint_utils as dju

config.register_externals()
config.register_adapters(context=locals())

schema = djp.schema(config.schema_name, create_schema=True)

@schema
class Tag(dju.VersionLookup):
    package = 'microns-coregistration-api'
    attr_name = 'tag'


@schema
class FieldImageMethod(djp.Lookup):
    hash_name = 'field_im_method'
    definition = f"""
    {hash_name} : varchar(6)
    """
        
    class EnhanceResizeNormalize(djp.Part):
        enable_hashing = True
        hash_name = 'field_im_method'
        hashed_attrs = 'lcn_kws', 'sharpen_2pimage_kws', 'resize_kws', Tag.attr_name
        definition = """
        -> master
        ---
        lcn_kws : varchar(1000) # kwargs to pass to microns_utils.transform_utils.lcn
        sharpen_2pimage_kws : varchar(1000) # kwargs to pass to microns_utils.transform_utils.sharpen_2pimage
        resize_kws : varchar(1000) # kwargs to pass to microns_coregistration_api.methods.registration.resize
        -> Tag
        """


@schema
class FieldImage(djp.Lookup):
    hash_name = 'field_im_id'
    definition = f"""
    {hash_name} : varchar(8)
    """
    class Store(djp.Part):
        hash_name = 'field_im_id'
        definition = """
        -> master
        -> FieldImageMethod
        -> nda.SummaryImages
        ---
        image : <field_image_files> # field image
        """

    class Maker(djp.Part, dj.Computed):
        enable_hashing = True
        hash_name = 'field_im_id'
        hashed_attrs = [FieldImageMethod.hash_name] + nda.SummaryImages.primary_key
        definition = """
        -> master
        -> FieldImageMethod
        -> nda.SummaryImages
        ---
        ts_computed=CURRENT_TIMESTAMP : timestamp
        """
            

@schema
class FieldGridMethod(djp.Lookup):
    hash_name = 'field_grid_method'
    definition = f"""
    {hash_name} : varchar(6)
    """
        
    class Affine(djp.Part):
        enable_hashing = True
        hash_name = 'field_grid_method'
        hashed_attrs = 'desired_res', Tag.attr_name
        definition = """
        -> master
        ---
        desired_res : float # desired resolution (um/px) for the output array
        -> Tag
        """

@schema
class FieldGrid(djp.Lookup):
    hash_name = 'field_grid_id'
    definition = f"""
    {hash_name} : varchar(8)
    """
    class Store(djp.Part):
        hash_name = 'field_grid_id'
        definition = """
        -> master
        -> FieldGridMethod
        -> nda.Registration.Affine
        ---
        grid : <field_grid_files> # field grid(s)
        """

    class Maker(djp.Part, dj.Computed):
        enable_hashing = True
        hash_name = 'field_grid_id'
        hashed_attrs = [FieldGridMethod.hash_name] + nda.Registration.Affine.primary_key
        definition = """
        -> master
        -> FieldGridMethod
        -> nda.Registration.Affine
        ---
        ts_computed=CURRENT_TIMESTAMP : timestamp
        """

schema.spawn_missing_classes()