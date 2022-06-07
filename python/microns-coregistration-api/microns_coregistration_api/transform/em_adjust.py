"""
"""

import logging
import os
from pathlib import Path
import numpy as np
from microns_utils.model_utils import InterpModel, PolyModel
from microns_utils.transform_utils import normalize
from microns_utils.misc_utils import classproperty

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
                    model - model type to use for solving pia and white matter.
                        types :
                            - linear - microns_utils.model_utils.PolyModel
                            - quadratic - microns_utils.model_utils.PolyModel 
                            - interpolation - microns_utils.model_utils.InterpModel with 
                    linear_model_params - params to pass to linear model function
                    quadratic_model_params - params to pass to quadratic model function
                    interp_model_params - params to pass to interpolation function
                    rotation - the desired rotation (degrees) to apply to coordinate system before solving for pia/ wm and adjusting
                    grid_spacing - the spacing in microns between grid points for :func:`make_grid`
                    grid_axes - the axes to use for making grid in :func:`make_grid`

        To solve the model once and then adjust points:
            adjuster = EMAdjust(method, rotation)
            adjuster.adjust(points)

    """ 
    _defaults = {}

    @classproperty
    def defaults(cls):
        cls._defaults.setdefault('pia_pts_npy_path', Path(__location__).joinpath('./data/pia_pts.npy'))
        cls._defaults.setdefault('wm_pts_npy_path', Path(__location__).joinpath('./data/wm_pts.npy'),)
        cls._defaults.setdefault('model', 'linear')
        cls._defaults.setdefault('linear_model_params', {'model': 'x + y'})
        cls._defaults.setdefault('quadratic_model_params', {'model': 'x + y + x*y + x^2 + y^2'})
        cls._defaults.setdefault('interp_model_params', {'method': 'rbf'})
        cls._defaults.setdefault('rotation', 3.5)
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

    def __init__(self, method, rotation=None):
        """
        Solve model for pia and white matter (wm) with optional rotation.

        :param method: (str) method to solve pia/ wm model
            Options : 
                - linear - microns_utils.model_utils.PolyModel solved with EMAdjust.defaults['linear_model_params']
                - quadratic - microns_utils.model_utils.PolyModel solved with EMAdjust.defaults['quadratic_model_params']
                - interpolation - microns_utils.model_utils.InterpModel solved with EMAdjust.defaults['interp_model_params']
            
        :param rotation: (float) optional rotation (degrees) to apply before solving
            default - None

        Usage: 
            adjuster = EMAdjust(method, rotation)
            adjuster.adjust(points),
                where points is an Nx3 array of x, y, z EM coordinates in nanometers
        """
        self.method = method
        self.rotation = rotation
        
        # solve models
        self.pia_model = self.solve(self.method, self.pia_pts if self.rotation is None else rotate_points(self.pia_pts, degree=self.rotation))
        self.wm_model = self.solve(self.method, self.wm_pts if self.rotation is None else rotate_points(self.wm_pts, degree=self.rotation))
    
    @classmethod
    def validate_points(cls, points):
        points = np.array(points)
        assert points.ndim == 2 and points.shape[-1] == 3, 'Array must be 2D with shape N x 3'
        return points

    @classmethod
    def solve(cls, method, data):
        """
        Solves for a model of data with given method.
        
        :param method: (str) method to use for solving
            Options : 
                - linear - microns_utils.model_utils.PolyModel solved with EMAdjust.defaults['linear_model_params']
                - quadratic - microns_utils.model_utils.PolyModel solved with EMAdjust.defaults['quadratic_model_params']
                - interpolation - microns_utils.model_utils.InterpModel solved with EMAdjust.defaults['interp_model_params']
        
        :param data: N x 3 array of x, y, z EM coordinates in nanometers

        :returns: solved model
        """
        data = cls.validate_points(data)

        if method == 'linear':
            return PolyModel(data[:, [0,2]], data[:, [1]], **cls.defaults['linear_model_params'])
            
        if method == 'quadratic':
            return PolyModel(data[:, [0,2]], data[:, [1]], **cls.defaults['quadratic_model_params'])
            
        if method == 'interpolation':
            return InterpModel(data[:, [0,2]], data[:, [1]], **cls.defaults['interp_model_params'])
        
        raise AttributeError('method not recognized. Options are "linear", "quadratic", "interpolation".' )
    
    @classmethod
    def normalize(cls, points, pia_y, wm_y):
        """
        Normalize points between pia and white matter

        :param points: N x 3 array of x, y, z EM points in nanometers
        :param pia_y: N x 1 array of y values for pia at x, z in points
        :param wm_y: N x 1 array of y values for white matter at x, z in points

        :returns: N x 3 array of normalized EM points in nanometers
        """
        points = cls.validate_points(points)

        return np.hstack([
                points[:, [0]], 
                normalize(points[:, [1]], pia_y, wm_y, cls.pia_mean_y, cls.wm_mean_y), 
                points[:, [2]]
        ])

    @classmethod
    def make_grid(cls, method, data):
        """
        Makes 2D grid of EM x, y,z coordinates after running through model.

        :param method: Method to run solver with. See EMAdjust.solve
        :param data: (N x 3) array of EM coordinates in nanometers to solve model
            Optionally can also pass the following strings:
                - "pia" : solves model with EMAdjust.pia_pts
                - "wm" : solves model with EMAdjust.wm_pts

        Additional params:
            grid axes specified by EMAdjust.defaults['grid_axes']
            grid spacing (microns) specified by EMAdjust.defaults['grid_spacing']

        :returns: N x 3 array of adjusted grid points in nanometers
        """
        if isinstance(data, str) and data  == 'pia':
            data = cls.pia_pts
        
        if isinstance(data, str) and data == 'wm':
            data = cls.wm_pts
            
        grid = em_grid_2D(axes=cls.defaults['grid_axes'], spacing=cls.defaults['grid_spacing'])
        
        data = cls.validate_points(data)

        model = cls.solve(method, data)

        return np.hstack([
            grid[:, [0]], 
            model.run(grid), 
            grid[:, [1]]
        ])

    @classmethod
    def solve_and_adjust(cls, points, model='default', rotation='default'):
        """
        Solves model and adjusts points in one step.

        :param points: N x 3 array of x, y, z EM coordinates to adjust in nanometers
        :param model: model to use to solve. See EMAdjust.solve
            Default - "default" : uses EMAdjust.defaults["model"]
        :param rotation: rotation to use for solver.
            Default - "default" : uses EMAdjust.defaults["rotation"]

        :returns: N x 3 array of adjusted grid points in nanometers 
        """
        points = cls.validate_points(points)

        if model == 'default':
            model = cls.defaults['model']
        
        if rotation == 'default':
            rotation = cls.defaults['rotation']

        pia_pts = rotate_points(cls.pia_pts, degree=rotation)
        wm_pts = rotate_points(cls.wm_pts, degree=rotation)
        pia_model = cls.solve(model, pia_pts)
        wm_model = cls.solve(model, wm_pts)
        pia_y = pia_model.run(points[:, [0, 2]])
        wm_y = wm_model.run(points[:, [0, 2]])
        return cls.normalize(points, pia_y, wm_y)

    def adjust(self, points):
        """
        Adjusts points with solved model. 

        :param points: N x 3 array of x, y, z EM coordinates to adjust in nanometers

        :returns: N x 3 array of adjusted grid points in nanometers 
        """
        points = self.validate_points(points)

        if self.rotation is not None:
            points = rotate_points(points, degree=self.rotation)
            
        pia_y = self.pia_model.run(points[:, [0, 2]])
        wm_y = self.wm_model.run(points[:, [0, 2]])
        
        return self.normalize(points, pia_y, wm_y)

    @property
    def pia_grid(self):
        """
        Returns Nx3 array of EM coordinates in nanometers corresponding to adjusted pia grid
        """
        return self.adjust(self.make_grid(self.method, self.pia_pts, self.defaults['grid_spacing']))
    
    @property
    def wm_grid(self):
        """
        Returns Nx3 array of EM coordinates in nanometers corresponding to adjusted white matter grid
        """
        return self.adjust(self.make_grid(self.method, self.wm_pts, self.defaults['grid_spacing']))