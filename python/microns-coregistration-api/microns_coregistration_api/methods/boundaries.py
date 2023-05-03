"""
Methods for computing layer and area boundaries
"""
import logging
from itertools import product

import numpy as np
from microns_utils.transform_utils import make_grid, normalize, run_kde, format_coords
from scipy.interpolate import LinearNDInterpolator, RBFInterpolator, interp1d
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

nm_per_vx = np.array([4, 4, 40]) # from precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/em data bounds
em_bounds_vx = np.array([[27648, 27648, 14816], [453632, 388096, 27904]]) # from graphene://https://minnie.microns-daf.com/segmentation/table/minnie3_v1
seg_bounds_vx = np.array([[52770, 60616, 14850], [437618, 322718, 27858]])
em_bounds_nm = em_bounds_vx * nm_per_vx
seg_bounds_nm = seg_bounds_vx * nm_per_vx

available_bounds = {
    'em': em_bounds_nm,
    'seg': seg_bounds_nm
}


def get_bounding_box(axis=None, source='em'):
    """
    Get bounding box of the source data along the specified axes in nanometers.

    Args:
        axis : None or int or tuple
            axis or axes for which to generate bounds, e.g: 1, (0, 2). 
            The default None will return the full bounding box of the source data.
        source : str 
            source of bounding box data. Must be one of:
                - em (default)
                - seg

    Returns: 
        numpy array with shape, 2 x len(axis) of:
            array([min_pts,
                   max_pts])
    """
    try:
        bounds = available_bounds[source]
    except KeyError:
        raise ValueError(f'source "{source}" invalid')
    if axis is None:
        return bounds
    if isinstance(axis, int):
        axis = axis,
    return bounds[:, axis]



def fit_best_threshold_logreg(data1, data2, **logreg_params):
    """
    adapted from Mara
    """
    X = np.concatenate([data1, data2]).reshape(-1, 1)
    y = np.concatenate([np.zeros_like(data1), np.ones_like(data2)])
    
    scaler = StandardScaler()
    scaler = scaler.fit(X)
    X = scaler.transform(X)
    
    logreg_params.setdefault('class_weight', 'balanced')
        
    logreg = LogisticRegression(**logreg_params)
    logreg.fit(X, y)
    
    result = - logreg.intercept_[0] / logreg.coef_[0]
    
    return scaler.inverse_transform(result.reshape(1, -1))


def fit_best_threshold_kde(data1, data2, **kde_params):
    """
    
    """
    kde_params.setdefault('nbins', 1000)
    nbins = kde_params.get('nbins')
    x1, y1 = run_kde(data1, **kde_params)
    x2, y2 = run_kde(data2, **kde_params)
    im1 = interp1d(x1, y1, fill_value ='nan', bounds_error=False)
    im2 = interp1d(x2, y2, fill_value ='nan', bounds_error=False)
    minx, maxx = np.min([np.min(x1), np.min(x2)]), np.max([np.max(x1), np.max(x2)])
    xvals = np.linspace(minx, maxx, nbins)
    a, b = im1(xvals), im2(xvals)           
    ama = np.nanargmax(a)
    amb = np.nanargmax(b)
    minr = np.minimum(ama, amb)
    maxr = np.maximum(ama, amb)
    na = a[minr: maxr]
    nb = b[minr: maxr]
    nx = xvals[minr: maxr]
    ny = np.abs((na - nb))
    return nx[np.nanargmin(ny)]


def fit_layer_boundaries(
    df, 
    xyz_column_names=('soma_x', 'soma_y', 'soma_z'), 
    pred_column_name='layer_prediction', 
    layer_names=('L23', 'L4', 'L5', 'L6'),
    grid_pts=(6, 4),
    grid_padding=0.01, 
    boundary_fit_method='logreg', 
    interp_method='LinearNDInterpolator',
    boundary_fit_method_params=None,
    interp_method_params=None
):
    """
    
    """
    boundary_fit_method_params = {} if boundary_fit_method_params is None else boundary_fit_method_params
    interp_method_params = {} if interp_method_params is None else interp_method_params
    
    xn, yn, zn = xyz_column_names

    x = df.loc[:, xn]
    y = df.loc[:, yn]
    z = df.loc[:, zn]
    
    # MAKE XZ GRID AND BINS
    xmin, xmax = x.min(), x.max()
    zmin, zmax = z.min(), z.max()
    xpad = (xmax - xmin) * grid_padding
    zpad = (zmax - zmin) * grid_padding

    bounds = np.array([
        [int(xmin - xpad), int(zmin - zpad)],
        [int(xmax + xpad), int(zmax + zpad)]]
    )
    grid = make_grid(bounds, npts=grid_pts)
    bin_inds = list(product(np.arange(grid.shape[0]-1),np.arange(grid.shape[1]-1)))
    init = np.zeros(np.array(grid.shape[:2]) - 1)
    
    # MAKE LAYER GROUPS
    qs = [f'{pred_column_name}=="{ln}"' for ln in layer_names]
    names = []
    queries = []
    for i in range(len(layer_names)-1):
        names.append(f'{layer_names[i]}/{layer_names[i+1]}')
        queries.append([qs[i], qs[i+1]])
    
    # SOLVE FOR LAYER BOUNDARY PER GRID BIN
    layer_boundary_grids = {}
    layer_boundary_models = {}
    for (q1, q2), name in zip(queries, names):
        subset_df = df.query(f'{q1} or {q2}')
        center_x = init.copy()
        center_z = init.copy()
        boundaries = init.copy()

        for x, z in bin_inds:
            # Retrieve all neurons within a grid cell.
            mins = grid[x, z]
            maxs = grid[x+1, z+1]
            center_x[x, z], center_z[x, z]= mins + (maxs - mins)//2
            bin_df = subset_df.query(f'{xn} >= {mins[0]} and {xn} <= {maxs[0]} and {zn} >= {mins[1]} and {zn} <= {maxs[1]}')

            if len(bin_df) > 0:
                # Soma depth of neurons per predicted layer.
                data1 = bin_df.query(q1)[yn].values
                data2 = bin_df.query(q2)[yn].values

                if boundary_fit_method == 'logreg':
                    boundaries[x, z] = fit_best_threshold_logreg(data1, data2, **boundary_fit_method_params)
                    
                elif boundary_fit_method == 'kde':
                    boundaries[x, z] = fit_best_threshold_kde(data1, data2, **boundary_fit_method_params)
                
                else:
                    raise AttributeError(f'boundary_fit_method {boundary_fit_method} not recognized.')
        

        center_points = np.stack([center_x, boundaries, center_z], -1).reshape(-1, 3)
        
        grid_x, grid_z = np.meshgrid(np.linspace(*bounds[:, 0], 100), np.linspace(*bounds[:, 1], 50), indexing='ij')
        
        boundary_points = np.stack([
            [grid[0, 0, 0], boundaries[0, 0], grid[0, 0, 1]],
            [grid[0, -1, 0], boundaries[0, -1], grid[0, -1, 1]],
            [grid[-1, 0, 0], boundaries[-1, 0], grid[-1, 0, 1]],
            [grid[-1, -1, 0], boundaries[-1, -1], grid[-1, 3, 1]],

        ])
        points = np.concatenate([center_points, boundary_points], axis=0)
        
        # SOLVE LAYER BOUNDARY MODELS AND MAKE GRIDS
        if interp_method == 'LinearNDInterpolator':
            layer_boundary_models[name] = LinearNDInterpolator(points[:, [0, 2]], points[:, 1], **interp_method_params)
            grid_y = layer_boundary_models.get(name)(grid_x, grid_z)
            assert ~np.any(np.isnan(grid_y))
            layer_boundary_grids[name] = np.stack([grid_x, grid_y, grid_z], axis=-1) 
            
        elif interp_method == 'RBFInterpolator':
            layer_boundary_models[name] = RBFInterpolator(points[:, [0, 2]], points[:, [1]], **interp_method_params)
            grid_xz = np.stack([grid_x, grid_z], -1).reshape(-1, 2)
            grid_y = layer_boundary_models.get(name)(grid_xz)
            layer_boundary_grids[name] = np.hstack([
                grid_xz[:, [0]],
                grid_y,
                grid_xz[:, [1]]
            ])
        else:
            raise AttributeError(f'interp_method {interp_method} not recognized.')
            
    return layer_boundary_models, layer_boundary_grids


def assign_layer(coords, layer_names, models):
    """
    
    """
    assert len(models) == len(layer_names)-1, 'The number of models must be one less than the number of layers'
    # assert np.shape(coords)[-1] == 3, 'coordinates must be 3D'
    # if np.ndim(coords) == 1:
    #     coords = np.expand_dims(coords, 0)
    coords = format_coords(coords, return_dim=2)
    xz = coords[:, [0, 2]]
    y = coords[:, [1]]

    signs = []
    for i, m in enumerate(models):
        ybound = m(xz)
        if np.ndim(ybound) == 1:
            ybound = np.expand_dims(ybound, 1)
        signs.append(np.sign(y - ybound))

    signs = np.hstack(signs)
    positions = signs.sum(1)
    nbounds = len(layer_names)-1
    layer_inds = normalize(positions, -nbounds, nbounds, 0, nbounds).astype(int)
    return np.array(layer_names)[layer_inds].tolist()