import numpy as np
from nglui import statebuilder
from microns_utils.ap_utils import set_CAVEclient
from coregister.transform.transform import Transform
from coregister.utils import em_nm_to_voxels


class NgLinks:
    client = set_CAVEclient('minnie65_phase3_v1')
    em_src = client.info.image_source()
    seg_src = client.info.segmentation_source()
    nuc_src = client.materialize.get_table_metadata('nucleus_detection_v0')['flat_segmentation_source']
    em_2p_src = 'precomputed://gs://neuroglancer/alex/calcium/minnie/EM_phase3_2p_coords'
    vess_2p_src = 'precomputed://gs://neuroglancer/alex/calcium/minnie/2pstack_vessels_highres'
    nuc_seg_src = 'precomputed://gs://neuroglancer/alex/calcium/minnie/nuc_seg_phase3_2p_coords'
    # image layers
    em_layer = statebuilder.ImageLayerConfig(em_src, contrast_controls=True, black=0.35, white=0.7)
    seg_layer = statebuilder.SegmentationLayerConfig(seg_src,  name='seg')
    nuc_layer = statebuilder.SegmentationLayerConfig(nuc_src, name='nuclear-seg')


def em_nm_to_voxels_phase3(xyz, x_offset=31000, y_offset=500, z_offset=3150, inverse=False):
    """convert EM nanometers to neuroglancer voxels
    Parameters
    ----------
    xyz : :class:`numpy.ndarray`
        N x 3, the inut array in nm
    inverse : bool
        go from voxels to nm
    Returns
    -------
    vxyz : :class:`numpy.ndarray`
        N x 3, the output array in voxels
    """
    if inverse: 
        vxyz = np.zeros_like(xyz).astype(float)
        vxyz[:, 0] = (xyz[:, 0] - x_offset) * 4.0
        vxyz[:, 1] = (xyz[:, 1] - y_offset) * 4.0
        vxyz[:, 2] = (xyz[:, 2] + z_offset) * 40.0
        
    else: 
        vxyz = np.zeros_like(xyz).astype(float)
        vxyz[:, 0] = ((xyz[:, 0] / 4) + x_offset)
        vxyz[:, 1] = ((xyz[:, 1] / 4) + y_offset)
        vxyz[:, 2] = ((xyz[:, 2]/40.0) - z_offset)


def format_coords(coords_xyz, return_dim=1):
    # format coordinates 
    coords_xyz = np.array(coords_xyz)
    
    assert np.logical_or(return_dim==1, return_dim==2), '"ndim" must be 1 or 2'
    assert np.logical_or(coords_xyz.ndim == 1, coords_xyz.ndim == 2), 'Coordinate(s) must be 1D or 2D'
    assert coords_xyz.shape[-1] == 3, 'Coordinate(s) must have exactly x, y, and z'
    
    coords_xyz = coords_xyz if coords_xyz.ndim == return_dim else np.expand_dims(coords_xyz, 0)
        
    return coords_xyz


def normalize(image, newrange=[0,255], clip_bounds=None, astype=np.uint8):
    image = np.array(image)
    if clip_bounds is not None:
        image = np.clip(image,clip_bounds[0], clip_bounds[1]) 
    return (((image - image.min())*(newrange[1]-newrange[0])/(image.max() - image.min())) + newrange[0]).astype(astype)


def aibs_coreg_transform(coords, version=None, direction=None, transform_type=None, transform_solution=None):
    """ Transform provided coordinate according to parameters
    
    :param coords: 1D or 2D list or array of coordinates 
        if coords are 2P, units should be microns
        if coords are EM, units should be Neuroglancer voxels of the appropriate version
    :param transform_version: "phase2" or "phase3"
    :param transform_direction: 
        "EM2P" --> provided coordinate is EM and output is 2P
        "2PEM" --> provided coordinate is 2P and output is EM
    :param transform_type:
        "linear" --> more nonrigid transform
        "spline" --> more rigid transform
    :param transform_obj: option to provide transform_obj
        If transform_obj is None, then it will be fetched using fetch_coreg function and provided transform paramaters
    """ 
    if version is not None:
        assert np.logical_or(version=='phase2', version=='phase3'), "version must be 'phase2' or 'phase3'"
    if direction is not None:
        assert np.logical_or(direction=="EM2P", direction=="2PEM"), "direction must be 'EM2P' or '2PEM"
    if transform_type is not None:
        assert np.logical_or(transform_type=='linear', transform_type=='spline'), "version must be 'linear' or 'spline'"
    
    # format coord
    coords_xyz = format_coords(coords, return_dim=2)
    
    # make transformation object
    transform_obj = Transform(json=transform_solution)

    # perform transformations
    if version == 'phase2':
        if direction == '2PEM':
            return (em_nm_to_voxels(transform_obj.tform(coords_xyz/ 1000))).squeeze()
        
        elif direction == 'EM2P':
            return (transform_obj.tform(em_nm_to_voxels(coords_xyz, inverse=True))*1000).squeeze()
        
        else:
            raise Exception('Provide direction ("2PEM" or "EM2P")')
        
    elif version == 'phase3':
        if direction == '2PEM':
            coords_xyz[:,1] = 1322 - coords_xyz[:,1] # phase 3 does not invert y so have to manually do it
            return transform_obj.tform(coords_xyz/1000).squeeze()
        
        elif direction == 'EM2P':
            new_coords = transform_obj.tform(coords_xyz)*1000
            new_coords[:,1] = 1322 - new_coords[:,1] # phase 3 does not invert y so have to manually do it
            return new_coords.squeeze()
        
        else:
            raise Exception('Provide direction ("2PEM" or "EM2P")')
    else:
        raise Exception('Provide version ("phase2" or "phase3")')