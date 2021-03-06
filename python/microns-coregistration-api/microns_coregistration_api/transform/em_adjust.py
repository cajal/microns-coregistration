"""
"""

import logging
import os
from pathlib import Path
import numpy as np
from microns_utils.model_utils import InterpModel, PolyModel
from microns_utils.transform_utils import normalize
from microns_utils.misc_utils import classproperty, unwrap

logger = logging.getLogger(__name__)

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))


em_nm_per_vx = np.array([4,4,40])
em_bounds_vx = np.array([[27648, 27648, 14816], [453632, 388096, 27904]])
em_bounds_nm =  em_bounds_vx * em_nm_per_vx 


def em_bounding_box_2D(axes, return_as='nm', vx_res=None):
    """
    Generate 2D bounding box of the EM volume.
    
    :param axes: tuple of shape (2,) containing the axes to use in the grid
    :param return_as: (str) Units to return bounding_box in
        Options: 
            nm - nanometers 
            vx - voxels
    :param vs_res: (array) voxel resolution if return_as="vx" 
        Default: 4 x 4 x 40 nm / vx

    :returns: 2 x 2 bounding box array in nanometers with the form:
        
        array([
                [axes[0] min pt, axes[0] max pt],
                [axes[1] min pt, axes[1] max pt]
            ]
        )
    """
    vx_res = em_nm_per_vx if vx_res is None else vx_res

    if return_as == 'nm':
        em_bounds = em_bounds_nm
    elif return_as == 'vx':
        em_bounds = em_bounds_nm * vx_res
    else:
        raise AttributeError(f'return_as {return_as} not recognized. Valid entries are "nm" and "vx".' )

    return np.stack([em_bounds[:, axes[0]], em_bounds[:, axes[1]]])


def em_grid_2D(axes, spacing=(25, 25)):
    """
    Generate 2D grid from the bounding box of the EM volume.  
    
    :param axes: tuple of shape (2,) containing the axes to use in the grid
    :param spacing: (int) spacing in microns between grid points
    
    :returns: N x 2 array of grid points in nanometers
    """
    bb_um = em_bounding_box_2D(axes) / 1000
    return np.stack(np.meshgrid(np.arange(*bb_um[0], spacing[0]), np.arange(*bb_um[1], spacing[1]), indexing='ij'), -1).reshape(-1,2) * 1000


def em_grid_3D(xrange='full', yrange='full', zrange='full', spacing=(25, 25, 25), return_as='nm', vx_res=None):
    """
    Generate a 3D array of grid points in the EM volume
    
    :param xrange: array of shape (n,) of EM x-values
        Default : 'full' - uses entire range of EM x-dimension
    :param yrange: array of shape (n,) of EM y-values
        Default : 'full' - uses entire range of EM y-dimension
    :param zrange: array of shape (n,) of EM z-values
        Default : 'full' - uses entire range of EM z-dimension
    :param spacing: tuple of shape (3,) with grid point spacing in microns for x, y, and z axis
    :param return_as: (str) units to return array as
        Options : 
            - 'nm' (default) - nanometers
            - 'vx' - voxels
    :param vx_res: tuple of shape (3,) with voxel resolution if return_as='vx'
        Default : em_adjust.em_bounds_vx
    
    :returns: N x 3 array of EM coordinates
    """
    em_bounds = em_bounds_nm
    ranges = []
    for r, b, s in zip([xrange, yrange, zrange], em_bounds.T, spacing):
        if isinstance(r, str) and r == 'full':
            r = b
            r = r / 1000
        else:
            r = r / 1000
            ranges.append(r)
            continue
        ranges.append(np.arange(*r, s))
    grid = np.stack(np.meshgrid(*ranges, indexing='ij'), -1).reshape(-1, 3) * 1000
    if return_as == 'nm':
        return grid 
    elif return_as == 'vx':
        vx_res = em_nm_per_vx if vx_res is None else vx_res
        return grid / vx_res
    
    
def run_2D_grid(model, axes):
    """
    Generates a regular 2D grid and runs through provided model.
    """
    reg_grid = em_grid_2D(axes)
    points = np.hstack([reg_grid[:, 0][:, None], model.run(reg_grid), reg_grid[:, 1][:, None]])
    return points[~np.isnan(points[:, 1])] # remove nans


def rotate_points(points, degree=5):
    """ Credit: Sven Dorkenwald, Forrest Collman
    
    Rotates points in the microns dataset
    
    points: Nx3 numpy array
        coordinates in nm
    degree: int
        degrees of rotation
        
    returns: Nx3 numpy array
        rotated points
    """
    angle = degree * np.pi / 180
    x = points[..., 0] * np.cos(angle) - points[..., 1] * np.sin(angle)
    y = points[..., 0] * np.sin(angle) + points[..., 1] * np.cos(angle)
    
    corrected_points = points.copy()
    corrected_points[..., 0] = x
    corrected_points[..., 1] = y
    
    return corrected_points


class EMAdjust:
    """
    Adjust the EM coordinate system to apply an optional rotation about the z-axis and 
        normalize the depth of each point relative to it's distance from pia and white matter.

    Usage:

        To use default settings in EMAdjust.defaults:
            
            EMAdjust.solve_and_adjust(points),
                where points is an Nx3 array of x, y, z EM coordinates in nanometers.

            Returns: adjusted Nx3 coordinate array

        To update default settings:

            EMAdjust.update_defaults(update_dict),
                where update_dict is a dict containing updated params.

                Param options:
                    pia_pts_npy_path - Path to npy file with pia points in nanometers to solve for pia model
                    wm_pts_npy_path - Path to npy file with white matter points in nanometers to solve for white matter model
                    method - method to use for solving pia and white matter.
                        types :
                            - "linear" - microns_utils.model_utils.PolyModel
                            - "quadratic" - microns_utils.model_utils.PolyModel 
                            - "interpolation" - microns_utils.model_utils.InterpModel with 
                    method_params - dict of dictionaries
                        "linear" - param dict to pass to linear model function
                        "quadratic" - param dict to pass to quadratic model function
                        "interpolation" - param dict to pass to interpolation function
                    rotation - the desired rotation (degrees) to apply to coordinate system before solving for pia/ wm and adjusting
                    grid_spacing - the spacing in microns between grid points for :func:`make_grid`
                    grid_axes - the axes to use for making grid in :func:`make_grid`

        To solve the model once and then adjust points:
            To solve once:
                adjuster = EMAdjust() # to use defaults in EMAdjust.defaults
            Or
                adjuster = EMAdjust(method, rotation) # to overwrite defaults

            To adjust points:
                adjuster.adjust(points)

    """ 
    _defaults = {}

    @classproperty
    def defaults(cls):
        cls._defaults.setdefault('pia_pts_npy_path', Path(__location__).joinpath('./data/pia_pts.npy'))
        cls._defaults.setdefault('wm_pts_npy_path', Path(__location__).joinpath('./data/wm_pts.npy'),)
        cls._defaults.setdefault('method', 'linear')
        cls._defaults.setdefault('rotation', 3.5)
        cls._defaults.setdefault('method_params', {
            'linear': {'model': 'x + y'},
            'quadratic': {'model': 'x + y + x*y + x^2 + y^2'},
            'interpolation': {'method': 'rbf'},
        })
        cls._defaults.setdefault('grid_spacing', (25, 25))
        cls._defaults.setdefault('grid_axes', (0, 2))
        return cls._defaults

    @classmethod
    def update_defaults(cls, update_dict:dict):
        cls._defaults.update(update_dict)
        logger.info('Default params updated.')
       
    @classproperty
    def pia_pts(cls):
        return np.load(cls.defaults['pia_pts_npy_path'])
    
    @classproperty
    def wm_pts(cls):
        return np.load(cls.defaults['wm_pts_npy_path'])
    
    @classproperty
    def pia_mean_y(cls):
        return cls.pia_pts.mean(0)[1]
    
    @classproperty
    def wm_mean_y(cls):
        return cls.wm_pts.mean(0)[1]

    def __init__(self, method='default', rotation='default', method_params='default'):
        """
        Solve model for pia and white matter (wm) with optional rotation.

        :param method: (str) method to solve pia/ wm model
            Default : "default" - uses EMAdjust.defaults["method"]
            Options : 
                - "linear" - microns_utils.model_utils.PolyModel solved with EMAdjust.defaults['linear_method_params']
                - "quadratic" - microns_utils.model_utils.PolyModel solved with EMAdjust.defaults['quadratic_method_params']
                - "interpolation" - microns_utils.model_utils.InterpModel solved with EMAdjust.defaults['interp_method_params']
            
        :param rotation: (float) rotation in degrees to use for solver, set rotation=None or rotation=0 for no rotation.
            Default : "default" - uses EMAdjust.defaults["rotation"]
        
        :param method_params: (dict of dictionaries) arguments to pass to solver
            Default : "default" - uses EMAdjust.defaults["method_params"]
        
        """
        method, rotation, method_params = self._check_for_defaults(**{'method':method, 'rotation':rotation, 'method_params':method_params})
        
        self.method = self._validate_method(method)
        self.rotation = rotation
        self.method_params = method_params

        self.pia_pts = self.pia_pts if self.rotation is None else rotate_points(self.pia_pts, degree=self.rotation)
        self.wm_pts = self.wm_pts if self.rotation is None else rotate_points(self.wm_pts, degree=self.rotation)
        
        # solve models
        self.pia_model = self.solve(self.pia_pts, self.method, self.method_params)
        self.wm_model = self.solve(self.wm_pts, self.method, self.method_params)
    
    @classmethod
    def _validate_points(cls, points):
        points = np.array(points)
        assert points.ndim == 2 and points.shape[-1] == 3, 'Array must be 2D with shape N x 3'
        return points

    @classmethod
    def _validate_method(cls, method):
        methods = ['linear', 'quadratic', 'interpolation']
        if method not in methods:
            raise AttributeError(f'method not recognized. Options are {methods}.' )
        return method

    @classmethod
    def _check_for_defaults(cls, **kwargs):
        return unwrap([cls.defaults[k] if v=='default' else v for k, v in kwargs.items()])

    @classmethod
    def solve(cls, data, method='default', method_params='default'):
        """
        Solves for a model of data with given method.
        
        :param method: (str) method to use for solving
            Options : 
                - "linear" - microns_utils.model_utils.PolyModel solved with EMAdjust.defaults['linear_method_params']
                - "quadratic" - microns_utils.model_utils.PolyModel solved with EMAdjust.defaults['quadratic_method_params']
                - "interpolation" - microns_utils.model_utils.InterpModel solved with EMAdjust.defaults['interp_method_params']
        :param data: N x 3 array of x, y, z EM coordinates in nanometers
        :param method_params: (dict) params to pass to method

        :returns: solved model
        """
        data = cls._validate_points(data)
        method, method_params = cls._check_for_defaults(**{'method':method, 'method_params':method_params})
        method = cls._validate_method(method)

        if method == 'linear':
            return PolyModel(data[:, [0, 2]], data[:, [1]], **method_params[method])
            
        if method == 'quadratic':
            return PolyModel(data[:, [0, 2]], data[:, [1]], **method_params[method])
            
        if method == 'interpolation':
            return InterpModel(data[:, [0, 2]], data[:, [1]], **method_params[method])
    
    @classmethod
    def _normalize(cls, points, pia_y, wm_y):
        """
        Normalize points between pia and white matter

        :param points: N x 3 array of x, y, z EM points in nanometers
        :param pia_y: N x 1 array of y values for pia at x, z in points
        :param wm_y: N x 1 array of y values for white matter at x, z in points

        :returns: N x 3 array of normalized EM points in nanometers
        """
        return np.hstack([
                points[:, [0]], 
                normalize(points[:, [1]], pia_y, wm_y, cls.pia_mean_y, cls.wm_mean_y), 
                points[:, [2]]
        ])

    @classmethod
    def solve_and_adjust(cls, points, method='default', rotation='default', method_params='default'):
        """
        Solves model and adjusts points in one step.

        :param points: N x 3 array of x, y, z EM coordinates to adjust in nanometers
        :param method: model to use to solve. See EMAdjust.solve
            Default - "default" : uses EMAdjust.defaults["method"]
        :param rotation: rotation in degrees to use for solver, set rotation=None or rotation=0 for no rotation.
            Default - "default" : uses EMAdjust.defaults["rotation"]

        :returns: N x 3 array of adjusted grid points in nanometers 
        """
        points = cls._validate_points(points)
        rotation = cls._check_for_defaults(**{'rotation': rotation})

        if rotation is not None:
            pia_pts = rotate_points(cls.pia_pts, degree=rotation)
            wm_pts = rotate_points(cls.wm_pts, degree=rotation)
            points = rotate_points(points, degree=rotation)
        else:
            pia_pts = cls.pia_pts
            wm_pts = cls.wm_pts

        pia_model = cls.solve(pia_pts, method, method_params)
        wm_model = cls.solve(wm_pts, method, method_params)
        pia_y = pia_model.run(points[:, [0, 2]])
        wm_y = wm_model.run(points[:, [0, 2]])
        return cls._normalize(points, pia_y, wm_y)

    def adjust(self, points):
        """
        Adjusts points with solved model. 

        :param points: N x 3 array of x, y, z EM coordinates to adjust in nanometers

        :returns: N x 3 array of adjusted grid points in nanometers 
        """
        points = self._validate_points(points)

        if self.rotation is not None:
            points = rotate_points(points, degree=self.rotation)
        
        pia_y = self.pia_model.run(points[:, [0, 2]])
        wm_y = self.wm_model.run(points[:, [0, 2]])
        
        return self._normalize(points, pia_y, wm_y)

    def make_pia_grid(self, grid_spacing=(25, 25, 25)):
        """
        Returns Nx3 array of EM coordinates in nanometers corresponding to adjusted pia grid
        """
        grid = rotate_points(em_grid_3D(yrange=self.pia_mean_y, spacing=grid_spacing), degree=self.rotation)
        return np.hstack([
            grid[:, [0]], 
            self.pia_model.run(grid[:, [0, 2]]), 
            grid[:, [2]]
        ])

    def make_wm_grid(self, grid_spacing=(25, 25, 25)):
        """
        Returns Nx3 array of EM coordinates in nanometers corresponding to adjusted wm grid
        """
        grid = rotate_points(em_grid_3D(yrange=self.wm_mean_y, spacing=grid_spacing), degree=self.rotation)
        return np.hstack([
            grid[:, [0]], 
            self.wm_model.run(grid[:, [0, 2]]), 
            grid[:, [2]]
        ])