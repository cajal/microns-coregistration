import pandas as pd
import numpy as np
import logging
import os
from pathlib import Path
from coregister.transform.transform import Transform
from coregister.utils import em_nm_to_voxels
from microns_utils.transform_utils import format_coords

logger = logging.getLogger(__name__)

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

aibs_transform_df = pd.read_pickle(Path(__location__).joinpath('./data/Coregistration.pkl'))

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


def fetch_aibs_transform(transform_id):
    row = aibs_transform_df.query(f'transform_id=={transform_id}')
    return row[['version', 'direction', 'transform_type', 'transform_solution']].to_dict(orient='records')[0]


def aibs_coreg_transform(coords, version, direction, transform_type, transform_solution):
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
    assert np.logical_or(version=='phase2', version=='phase3'), "version must be 'phase2' or 'phase3'"
    assert np.logical_or(direction=="EM2P", direction=="2PEM"), "direction must be 'EM2P' or '2PEM"
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