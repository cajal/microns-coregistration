"""
"""

import logging
import os
from pathlib import Path

import numpy as np
from microns_utils.misc_utils import classproperty
from microns_utils.model_utils import InterpModel, PolyModel
from microns_utils.transform_utils import (make_grid, normalize,
                                           rotate_points_3d)

from .boundaries import get_bounding_box

logger = logging.getLogger(__name__)

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))


class M65Adjust:
    """
    Adjust minnie65 EM coordinates via rotation and/ normalization, 
        with options to normalize y-values (depth) to pia and white matter (wm) depths

    NOTE: All input coordinates must be in nanometers        

    Basic Usage

        The following transforms can be sequentially applied to adjust user-provided 
            coordinates:
            1. Rotation
            2. Normalization

        The default transform performs the following:
            1. apply a 3.5 degree rotation about the z-axis
            2. solve a linear model for the rotated pia with an included set of 
                manually placed points
            3. solve a linear model for the rotated wm with an included set of 
                manually placed points
            4. solve for the pia and wm depth of each user-provided coordinate 
                via the models
            5. normalize each user-provided y-value according to its pia and wm 
                depth, setting the new min and new max to the mean pia depth and 
                mean wm depth, respectively

        To adjust with the default transform:

            >>> M65Adjust.default_adjust(points)


        To adjust with rotation only, ignoring all defaults:

            >>> M65Adjust.rotate_points(points, rotate_z=3.5)

            >>> M65Adjust.rotate_points(points, rotate_x=14, rotate_z=3.5)


        To adjust with user-specified params:

            NOTE: Any params not provided by user will use default params

            Override default params (Recommended):

                >>> tform = M65Adjust(**param_dict)
                >>> tform.adjust(points)

            Overwrite default params:

                >>> M65Adjust.update_default_params(**param_dict)
                >>> M65Adjust.default_adjust(points)

            Restore default params:

                >>> M65Adjust.restore_default_params()


    Advanced Usage        

        Rotation

            Params "rotate_x", "rotate_y", "rotate_z" toggle rotation about 
                each axis separately
            
            Set to the desired degree, or None for no rotation:

                >>> tform = M65Adjust(rotate_x=5, rotate_y=None, rotate_z=10)
                >>> tform.adjust(points) 


        Pia/ WM solvers

            Pia and WM solvers are needed for some methods of normalization. 
                See Normalization section below for more detail. 

            param "solve_method" controls which method will generate the pia and wm models
                'poly' - solve for the constants of a polynomial via least squares
                'interp' - interpolation

                If the class is initialized, the solved models can be found:

                    >>> tform = M65Adjust()
                    >>> tform.pia_model
                    >>> tform.wm_model

                Arbitrary x-z points (Nx2) can be run through the model to predict 
                    the y-values at those points:

                    >>> tform.pia_model.run(xz_points)

            Poly method

                Param "poly_method" controls the polynomial equation used
                    'linear' --> 'x + y'
                    'quadratic' --> 'x + y + x*y + x^2 + y^2'

                Param "poly_model" can be passed if a different equation is desired. 
                    In this case "poly_method" is ignored

                    >>> tform = M65Adjust(poly_model='x + y^2')

                To pass in the constants of a solved model use param "poly_constants". 
                    In this case, "poly_method" or "poly_model" specifying the equation 
                    that was used to solve the model is still required. 

                    >>> tform = M65Adjust(poly_method='quadratic', poly_constants=constants))
                    >>> tform = M65Adjust(poly_model='x + y^2', poly_constants=constants)

                constants and r**2 can be accessed via the solved model:

                    >>> tform.pia_model.constants
                    >>> tform.pia_model.r2

                NOTE: the first element of constants is the bias term, the rest are the 
                    constants for each term in the equation, in the order of the equation

                For more info on the polynomial method see PolyModel

            Interpolation method

                Param "interp_method" controls the type of interpolation used
                    - 'griddata': the interpolation solver with use scipy.interpolate.griddata
                    - 'rbf': the interpolation will use scipy.interpolate.RBFInterpolator

                For more info on the interpolation method see InterpModel

            Solve Pia and WM models with separate methods

                All params pertaining to solve methods can be passed with the prefix 'pia_' or 'wm_' 
                    to target pia or wm, respectively. Params with a prefix take priority over params without. 

                This will use a quadratic model for wm but will use the default method for pia:

                    >>> tform = M65Adjust(wm_poly_method='quadratic')

        Normalization

            Each column of x, y, and z is normalized separately or not at all, and 
                y (depth) has additional options

            Basic normalization

                Params "normalize_x", "normalize_y", and "normalize_z" toggle normalization of x, y, and z
                    By default: 
                        "normalize_x" = False
                        "normalize_y" = True
                        "normalize_z" = False 

                This will skip all solve and normalization and rotate only:

                    >>> tform = M65Adjust(normalize_y=False) 

                The formula for normalization is:

                    (points - from_min) * ((to_max - to_max) / (from_max - from_min)) + to_min

                Normalization rescales points from bounds (from_min, from_max) to (to_min, to_max)

                To control from_min, from_max, to_min, to_max separately pass in the desired values. 
                    Using x as an example:

                    >>> param_dict = {
                        'normalize_x': True,
                        'normalize_xmin_from': 1000,
                        'normalize_xmax_from': 10000000,
                        'normalize_xmin_to': 0,
                        'normalize_xmax_to': 1,
                    }
                    >>> tform = M65Adjust(**param_dict)

                If "normalize_xmin_from" or "normalize_xmax_from" are None, then the original 
                    data bounds specified by "normalize_bounds" will be used.

                This will rescale the points from the original bounds to 0, 1:

                    >>> param_dict = {
                        'normalize_x': True,
                        'normalize_xmin_to': 0,
                        'normalize_xmax_to': 1,
                    }
                    >>> tform = M65Adjust(**param_dict)  

                If only one of "normalize_xmin_to" or "normalize_xmax_to" is None, then 
                    the other will default to the bounds specified by "normalize_bounds".

                This will rescale the points from the original bounds to 0 and the original max bound:

                    >>> param_dict = {
                        'normalize_x': True,
                        'normalize_xmin_to': 0,
                    }
                    >>> tform = M65Adjust(**param_dict)

                If both "normalize_xmin_to" and "normalize_xmax_to" are None, then normalization for x
                    will be skipped.

                To normalize the entire volume to 0, 1 without taking into account pia and wm depths:

                    >>> param_dict = {
                        'normalize_x': True,
                        'normalize_xmin_to': 0,
                        'normalize_xmax_to': 1,
                        'normalize_y': True,
                        'normalize_ymin_to': 0,
                        'normalize_ymax_to': 1,
                        'normalize_z': True,
                        'normalize_zmin_to': 0,
                        'normalize_zmax_to': 1,
                    }
                    >>> tform = M65Adjust(**param_dict)

            Depth normalization to Pia or WM

                normalization of the y column (depth) has the additional option to adjust 
                    the min values to the coordinates of the pia or the max values to the 
                    coordinates of the white matter.

                available methods:
                    - pia_per_point - normalizes the min of each point separately to the depth 
                        of the rotated pia at that point
                    - pia_mean - normalizes the min of all points to the mean depth of the 
                        rotated pia
                    - wm_per_point - normalizes the max of each point separately to the depth 
                        of the rotated wm at that point
                    - wm_mean - normalizes the max of all points to the mean depth of the 
                        rotated pia


                The default method normalizes each point by its respective pia, wm depth and 
                    rescales to the mean pia, wm depths. It is equivalent to:

                    >>> param_dict = {
                        'normalize_y': True,
                        'normalize_ymin_from': 'pia_per_point',
                        'normalize_ymax_from': 'wm_per_point',
                        'normalize_ymin_to': 'pia_mean',
                        'normalize_ymax_to': 'wm_mean',
                    }
                    >>> tform = M65Adjust(**param_dict)  

                To normalize each point to 0, 1 after the point-specific pia/ wm adjustment:

                    >>> param_dict = {
                        'normalize_y': True,
                        'normalize_ymin_from': 'pia_per_point',
                        'normalize_ymax_from': 'wm_per_point',
                        'normalize_ymin_to': 0,
                        'normalize_ymax_to': 1,
                    }
                    >>> tform = M65Adjust(**param_dict)

                To normalize all points from the mean rotated pia and mean rotated wm depths to 0, 1:

                    >>> param_dict = {
                        'normalize_y': True,
                        'normalize_ymin_from': 'pia_mean',
                        'normalize_ymax_from': 'wm_mean',
                        'normalize_ymin_to': 0,
                        'normalize_ymax_to': 1,
                    }
                    >>> tform = M65Adjust(**param_dict)

                Any of these can be combined with x and z normalization:

                    >>> param_dict = {
                        'normalize_x': True,
                        'normalize_xmin_to': 0,
                        'normalize_xmax_to': 1,
                        'normalize_y': True,
                        'normalize_ymin_from': 'pia_mean',
                        'normalize_ymax_from': 'wm_mean',
                        'normalize_ymin_to': 0,
                        'normalize_ymax_to': 1,
                        'normalize_z': True,
                        'normalize_zmin_to': 0,
                        'normalize_zmax_to': 1,
                    }
                    >>> tform = M65Adjust(**param_dict)

        User-provided pia or wm points
            param "npy_path_to_pia_pts" or "npy_path_to_wm_pts" specifies file location of pia or wm points
                Note: file must be compatible with np.load


    Params

        npy_path_to_pia_pts (Path or str)
            path to file to pass to np.load for pia coordinates in nm

        npy_path_to_wm_pts (Path or str)
            path to file to pass to np.load for wm coordinates in nm

        rotate_x (None or float)
            degrees to rotate coordinates about x-axis, or skip if None

        rotate_y (None or float)
            degrees to rotate coordinates about y-axis, or skip if None

        rotate_z (None or float)
            degrees to rotate coordinates about z-axis, or skip if None

        solve_method (str)
            solver methods 
                options:
                    - 'poly': fit a polynomial model via least squares
                    - 'interp': interpolation
            can also be passed as "pia_solve_method" or "wm_solve_method"

        solve_method_kws (dict)
            dict with kwargs to pass to the solver
            can also be passed as "pia_solve_method_kws" or "wm_solve_method_kws"

        poly_method (str)
            methods for determining the model to pass to the polynomial solver. 
            ignored if an alternative solve_method is used.
                options:
                    - 'linear': solve for the constants of a 
                        linear model (x + y)
                    - 'quadratic': solve for the constants of a 
                        quadratic model (x + y + x*y + x^2 + y^2)
            can also be passed as "pia_poly_method" or "wm_poly_method"

        poly_model (str)
            model to pass to the polynomial solver. 
            If provided, poly_method will be ignored.
            can also be passed as "pia_model" or "wm_model"

        poly_constants (ndarray)
            the constants to initialize a solved model. 
            'poly_method' or 'poly_model' must also be provided.
            can also be passed as "pia_poly_constants" or "wm_poly_constants"

        interp_method (str)
            methods for the interpolation solver. 
            ignored if an alternative solve_method is used.
                options:
                    - 'griddata': the interpolation solver 
                        will use scipy.interpolate.griddata
                    - 'rbf': the interpolation will use 
                        scipy.interpolate.RBFInterpolator
            can also be passed as "pia_interp_method" or "wm_interp_method"

        normalize_bounds (str)
            the bounding box to use for the EM in the case that normalization requires it.
                options:
                    - 'em': the em bounding box
                    - 'seg': the segmentation bounding box

        normalize_x (bool)
            toggles normalization of x

        normalize_y (bool)
            toggles normalization of y. If False all solve and normalization params are ignored.

        normalize_z (bool)
            toggles normalization of z

        normalize_xmin_from (None or int/ float)
            min value to normalize x-values from. 
            If None, and if param "normalize_xmin_to" or "normalize_xmax_to" is not None, 
                then the xmin of the bounding box specified by param "normalize_bounds" is used

        normalize_xmax_from (None or int/ float)
            max value to normalize x-values from. 
            If None, and if param "normalize_xmin_to" or "normalize_xmax_to" is not None, 
                then the xmax of the bounding box specified by param "normalize_bounds" is used

        normalize_xmin_to (None or int/ float)
            min value to normalize x-values to. 
            If None, and if param "normalize_xmax_to" is not None, 
                then the xmin of the bounding box specified by param "normalize_bounds" is used

        normalize_xmax_to (None or int/ float)
            max value to normalize x-values to. 
            If None, and if param "normalize_xmin_to" is not None, 
                then the xmax of the bounding box specified by param "normalize_bounds" is used

        normalize_ymin_from (None or str or int/ float)
            min value(s) to normalize y-values from
            options:
                - int or float: the value to use
                - 'pia_per_point': the min value for each point is computed from the pia model, 
                        or if no pia model was solved, then defaults to 'pia_mean' below
                - 'pia_mean': the mean pia depth is used for all points
            If None, and if param "normalize_ymin_to" or "normalize_ymax_to" is not None, 
                then the ymin of the bounding box specified by param "normalize_bounds" is used

        normalize_ymax_from (None or str or int/ float)
            max value(s) to normalize y-values from
            options:
                - int or float: the value to use
                - 'wm_per_point': the max value for each point is computed from the wm model, 
                        or if no wm model was solved, then defaults to 'wm_mean' below
                - 'wm_mean': the mean wm depth is used for all points
            If None, and if param "normalize_ymin_to" or "normalize_ymax_to" is not None, 
                then the ymax of the bounding box specified by param "normalize_bounds" is used

        normalize_ymin_to (None or str or int/ float)
            min value to normalize y-values to
            options:
                - int or float: the value to use
                - 'pia_mean': the mean pia depth is used
            If None, and if param "normalize_ymax_to" is not None, 
                then the ymin of the bounding box specified by param "normalize_bounds" is used

        normalize_ymax_to (None or str or int/ float)
            max value to normalize y-values to
            options:
                - int or float: the value to use
                - 'wm_mean': the mean wm depth is used
            If None, and if param "normalize_ymin_to" is not None, 
                then the ymax of the bounding box specified by param "normalize_bounds" is used

        normalize_zmin_from (None or int/ float)
            min value to normalize z-values from. 
            If None, and if param "normalize_zmin_to" or "normalize_zmax_to" is not None, 
                then the zmin of the bounding box specified by param "normalize_bounds" is used

        normalize_zmax_from (None or int/ float)
            max value to normalize z-values from. 
            If None, and if param "normalize_zmin_to" or "normalize_zmax_to" is not None, 
                then the zmax of the bounding box specified by param "normalize_bounds" is used

        normalize_zmin_to (None or int/ float)
            min value to normalize z-values to. 
            If None, and if param "normalize_zmax_to" is not None, 
                then the zmin of the bounding box specified by param "normalize_bounds" is used

        normalize_zmax_to (None or int/ float)
            max value to normalize z-values to. 
            If None, and if param "normalize_zmin_to" is not None, 
                then the zmax of the bounding box specified by param "normalize_bounds" is used
    """
    _default_params = {}
    solve_methods = ('poly', 'interp')
    poly_methods = ('linear', 'quadratic')
    poly_models = {'linear': 'x + y', 'quadratic': 'x + y + x*y + x^2 + y^2'}
    interp_methods = ('griddata', 'rbf')
    bound_options = ('em', 'seg')

    @classproperty
    def default_params(cls):
        if not cls._default_params:
            cls._set_default_params()
        return cls._default_params

    @classmethod
    def _set_default_params(cls):
        # DATA PATHS
        cls._default_params.update(
            {'npy_path_to_pia_pts': Path(__location__).joinpath('./data/pia_pts.npy')})
        cls._default_params.update(
            {'npy_path_to_wm_pts': Path(__location__).joinpath('./data/wm_pts.npy')})

        # ROTATIONS
        cls._default_params.update({'rotate_x': None})
        cls._default_params.update({'rotate_y': None})
        cls._default_params.update({'rotate_z': 3.5})

        # SOLVE METHOD
        cls._default_params.update({'solve_method': 'poly'})
        cls._default_params.update({'solve_method_kws': None})

        # POLY METHOD
        cls._default_params.update({'poly_method': 'linear'})
        cls._default_params.update({'poly_constants': None})

        # INTERP METHOD
        cls._default_params.update({'interp_method': 'rbf'})

        # NORMALIZE METHOD
        cls._default_params
        cls._default_params.update({'normalize_bounds': 'seg'})
        cls._default_params.update({'normalize_x': False})
        cls._default_params.update({'normalize_y': True})
        cls._default_params.update({'normalize_ymin_from': 'pia_per_point'})
        cls._default_params.update({'normalize_ymax_from': 'wm_per_point'})
        cls._default_params.update({'normalize_ymin_to': 'pia_mean'})
        cls._default_params.update({'normalize_ymax_to': 'wm_mean'})
        cls._default_params.update({'normalize_z': False})

    @classmethod
    def update_default_params(cls, **param_dict):
        cls._default_params.update(param_dict)
        logger.info('Default params updated.')

    @classmethod
    def restore_default_params(cls):
        cls._set_default_params()
        logger.info('Default params restored.')

    @classproperty
    def pia_pts_nm(cls):
        """ manually placed pia points in nanometers"""
        return np.load(Path(cls.default_params.get('npy_path_to_pia_pts')))

    @classproperty
    def wm_pts_nm(cls):
        """ manually placed white matter points in nanometers"""
        return np.load(Path(cls.default_params.get('npy_path_to_wm_pts')))

    @classmethod
    def rotate_points(cls, points, rotate_x=None, rotate_y=None, rotate_z=None, rotate_kws=None, info=None):
        """
        Rotate points

        Args
            points (Nx3 array)
                 points to rotate
            rotate_x (None or int)
                degrees to rotate points about x-axis
                    If None, 0 degrees will be used
            rotate_y (None or int)
                degrees to rotate points about y-axis
                    If None, 0 degrees will be used
            rotate_z (None or int)
                degrees to rotate points about z-axis
                    If None, 0 degrees will be used
            rotate_kws (None or dict)
                kwargs to pass to rotation func
            info (None or str)
                str to pass to logger
        Returns
            Rotated points (Nx3 array)
        """
        if info is not None:
            logger.info(info)
        logger.info(
            f'rotating about x-axis: {rotate_x}, y-axis: {rotate_y}, z-axis: {rotate_z} degrees')
        return rotate_points_3d(
            points,
            cols=(0, 1, 2),
            degrees=(
                rotate_x or 0,
                rotate_y or 0,
                rotate_z or 0
            ),
            **rotate_kws if rotate_kws is not None else {}
        )

    @classmethod
    def solve(cls,
              solve_method,
              features=None,
              targets=None,
              poly_method=None,
              poly_model=None,
              poly_constants=None,
              interp_method=None,
              method_kws=None,
              info=None
              ):
        """
        Solve or initialize a model

        Args
            solve_method (str)
                solver method
                    options:
                        - 'poly': fit a polynomial model via least squares
                        - 'interp': interpolation
            features (None or array)
                feature array to pass to solver. If None, then method_kws must be able to initialize a solved model 
            targets (None or array)
                target array to pass to solver. If None, then method_kws must be able to initialize a solved model
            poly_method (None or str)
                methods for determining the model to pass to the polynomial solver. 
                    ignored if an alternative solve_method is used.
                    options:
                        - 'linear': solve for the constants of a linear model (x + y)
                        - 'quadratic': solve for the constants of a quadratic model (x + y + x*y + x^2 + y^2)
            poly_model (None or str)
                model to pass to the polynomial solver. If provided, poly_method will be ignored.
            poly_constants (array)
                the constants to initialize a solved model. 'poly_method' or 'poly_model' must also be provided.
            interp_method (None or str)
                methods for the interpolation solver. ignored if an alternative solve_method is used.
                    options:
                        - 'griddata': the interpolation solver with use scipy.interpolate.griddata
                        - 'rbf': the interpolation will use scipy.interpolate.RBFInterpolator
            method_kws (None or dict)
                dict with kwargs to pass to the solver
            info (None or str)
                str to pass to logger
        """
        if info is not None:
            logger.info(info)
        assert solve_method in cls.solve_methods, f'solve_method options: {cls.solve_methods}'
        if poly_method is not None:
            assert poly_method in cls.poly_methods, f'poly_method options: {cls.poly_methods}'
        if interp_method is not None:
            assert interp_method in cls.interp_methods, f'interp_method options: {cls.interp_methods}'

        method_kws = method_kws if method_kws is not None else {}

        if solve_method == 'poly':
            if poly_model is None:
                if poly_method is None:
                    raise AttributeError(f'Provide poly_method')
                model = cls.poly_models.get(poly_method)
            else:
                model = poly_model
                if poly_method is not None:
                    logger.warning(
                        'ignoring poly_method because poly_model was provided.')

            if poly_constants is not None:
                logger.info(
                    f'initializing PolyModel with model {model}, constants {poly_constants} and kws {method_kws}')
                return PolyModel(
                    model,
                    constants=poly_constants,
                    **method_kws
                )
            else:
                logger.info(
                    f'solving PolyModel with model {model} and kws {method_kws}')
                return PolyModel(
                    model,
                    features=features,
                    targets=targets,
                    **method_kws
                )
        if solve_method == 'interp':
            logger.info(
                f'solving InterpModel with method {interp_method} and kws {method_kws}')
            return InterpModel(
                points=features,
                values=targets,
                method=interp_method,
                method_kws=method_kws
            )

    @classmethod
    def normalize_column(cls, col, points, from_min, from_max, to_min, to_max):
        """
        Normalize a column of Nx3 array

        Args
            col (int)
                desired column to normalize, one of 0, 1, or 2
            points (N x 3 array)
                 the col that will be rotated is: points[:, col]
            from_min (int or float or array)
                min(s) of points[:, col] to normalize from
            from_max (int or float or array)
                max(s) of points[:, col] to normalize from
            to_min (int or float or array)
                min(s) to normalize to
            to_max (int or float)
                max(s) to normalize to
        Returns
            Nx3 array of normalized points
        """
        m = {0: 'x', 1: 'y', 2: 'z'}
        logger.info(f'normalizing {m[col]} values')
        if col == 0:
            return np.hstack([
                normalize(points[:, [0]], from_min, from_max, to_min, to_max),
                points[:, [1]],
                points[:, [2]]
            ])
        elif col == 1:
            return np.hstack([
                points[:, [0]],
                normalize(points[:, [1]], from_min, from_max, to_min, to_max),
                points[:, [2]]
            ])
        elif col == 2:
            return np.hstack([
                points[:, [0]],
                points[:, [1]],
                normalize(points[:, [2]], from_min, from_max, to_min, to_max),
            ])
        else:
            raise AttributeError(
                f'col {col} not allowed. col must be 0, 1, or 2.')

    @classmethod
    def _apply_rotation(cls, params):
        if np.any([params.get('rotate_x'), params.get('rotate_y'), params.get('rotate_z')]):
            return True

    @classmethod
    def _use_bounds(cls, params):
        if np.any([params.get('normalize_x'), params.get('normalize_y'), params.get('normalize_z')]):
            return True

    @classmethod
    def _apply_normalize_column(cls, col, params):
        xyz = {0: 'x', 1: 'y', 2: 'z'}[col]
        if params.get(f'normalize_{xyz}'):
            if params.get(f'normalize_{xyz}min_to') is not None or params.get(f'normalize_{xyz}max_to') is not None:
                return True

    @classmethod
    def _use_pia(cls, params):
        if params.get('normalize_y'):
            if 'pia' in str(params.get('normalize_ymin_from')) or 'pia' in str(params.get('normalize_ymin_to')):
                return True

    @classmethod
    def _use_wm(cls, params):
        if params.get('normalize_y'):
            if 'wm' in str(params.get('normalize_ymax_from')) or 'wm' in str(params.get('normalize_ymax_to')):
                return True

    @classmethod
    def _adjust(cls, points_nm, params, pia_pts_nm=None, wm_pts_nm=None, pia_model=None, wm_model=None, disable_rotation=False, disable_normalization=False):
        """
        Adjust points
        """
        bounds = get_bounding_box(source=params.get('normalize_bounds'))
        
        # ROTATE
        if not disable_rotation:
            if cls._apply_rotation(params):
                points_nm = cls.rotate_points(
                    points=points_nm,
                    rotate_x=params.get('rotate_x'),
                    rotate_y=params.get('rotate_y'),
                    rotate_z=params.get('rotate_z'),
                    info='--> Rotating points...'
                )
            if cls._use_bounds(params):
                if cls._apply_rotation(params):
                    bounds = cls.rotate_points(
                        points=bounds,
                        rotate_x=params.get('rotate_x'),
                        rotate_y=params.get('rotate_y'),
                        rotate_z=params.get('rotate_z'),
                        info='--> Rotating bounds for normalization...'
                    )

        # NORMALIZE
        if not disable_normalization:
            if cls._apply_normalize_column(0, params):
                xmin, xmax = bounds[:, 0]
                fmin = xmin if params.get(
                    'normalize_xmin_from') is None else params.get('normalize_xmin_from')
                fmax = xmax if params.get(
                    'normalize_xmax_from') is None else params.get('normalize_xmax_from')
                tmin = xmin if params.get(
                    'normalize_xmin_to') is None else params.get('normalize_xmin_to')
                tmax = xmax if params.get(
                    'normalize_xmax_to') is None else params.get('normalize_xmax_to')
                points_nm = cls.normalize_column(
                    col=0,
                    points=points_nm,
                    from_min=fmin,
                    from_max=fmax,
                    to_min=tmin,
                    to_max=tmax,
                )
            if cls._apply_normalize_column(1, params):
                ymin, ymax = bounds[:, 1]
                fmin = ymin if params.get(
                    'normalize_ymin_from') is None else params.get('normalize_ymin_from')
                fmax = ymax if params.get(
                    'normalize_ymax_from') is None else params.get('normalize_ymax_from')
                tmin = ymin if params.get(
                    'normalize_ymin_to') is None else params.get('normalize_ymin_to')
                tmax = ymax if params.get(
                    'normalize_ymax_to') is None else params.get('normalize_ymax_to')

                if cls._use_pia(params):
                    pia_mean = pia_pts_nm[:, 1].mean()

                    if isinstance(fmin, str):
                        if fmin == 'pia_per_point':
                            fmin = pia_model.run(points_nm[:, [0, 2]])
                        elif fmin == 'pia_mean':
                            fmin = pia_mean
                        else:
                            raise AttributeError(
                                f'normalize_ymin_from "{fmin}" not recognized.')

                    if isinstance(tmin, str):
                        if tmin == 'pia_mean':
                            tmin = pia_mean
                        else:
                            raise AttributeError(
                                f'normalize_ymin_to "{tmin}" not recognized.')

                if cls._use_wm(params):
                    wm_mean = wm_pts_nm[:, 1].mean()

                    if isinstance(fmax, str):
                        if fmax == 'wm_per_point':
                            fmax = wm_model.run(points_nm[:, [0, 2]])
                        elif fmax == 'wm_mean':
                            fmax = wm_mean
                        else:
                            raise AttributeError(
                                f'normalize_ymax_from "{fmax}" not recognized.')

                    if isinstance(tmax, str):
                        if tmax == 'wm_mean':
                            tmax = wm_mean
                        else:
                            raise AttributeError(
                                f'normalize_ymax_to "{tmax}" not recognized.')

                points_nm = cls.normalize_column(
                    col=1,
                    points=points_nm,
                    from_min=fmin,
                    from_max=fmax,
                    to_min=tmin,
                    to_max=tmax,
                )
            if cls._apply_normalize_column(2, params):
                zmin, zmax = bounds[:, 2]
                fmin = zmin if params.get(
                    'normalize_zmin_from') is None else params.get('normalize_zmin_from')
                fmax = zmax if params.get(
                    'normalize_zmax_from') is None else params.get('normalize_zmax_from')
                tmin = zmin if params.get(
                    'normalize_zmin_to') is None else params.get('normalize_zmin_to')
                tmax = zmax if params.get(
                    'normalize_zmax_to') is None else params.get('normalize_zmax_to')
                points_nm = cls.normalize_column(
                    col=2,
                    points=points_nm,
                    from_min=fmin,
                    from_max=fmax,
                    to_min=tmin,
                    to_max=tmax,
                )
        return points_nm

    @classmethod
    def default_adjust(cls, points_nm):
        """
        Adjust points with default parameters

        Args
            points_nm (Nx3 array)
                EM coordinates to adjust in nanometers
        Returns
            adjusted points (Nx3 array)
        """
        logger.info('--> Adjusting points with default params')

        if cls._use_pia(cls.default_params):
            if cls._apply_rotation(cls.default_params):
                # Rotate pia
                pia_pts_nm = cls.rotate_points(
                    cls.pia_pts_nm,
                    rotate_x=cls.default_params.get('rotate_x'),
                    rotate_y=cls.default_params.get('rotate_y'),
                    rotate_z=cls.default_params.get('rotate_z'),
                    info='--> Rotating PIA...'
                )
            # Solve pia model
            pia_model = cls.solve(
                solve_method=cls.default_params.get(
                    'pia_solve_method', cls.default_params.get('solve_method')),
                features=pia_pts_nm[:, [0, 2]],
                targets=pia_pts_nm[:, [1]],
                poly_method=cls.default_params.get(
                    'pia_poly_method', cls.default_params.get('poly_method')),
                poly_model=cls.default_params.get(
                    'pia_poly_model', cls.default_params.get('poly_model')),
                poly_constants=cls.default_params.get(
                    'pia_poly_constants', cls.default_params.get('poly_constants')),
                interp_method=cls.default_params.get(
                    'pia_interp_method', cls.default_params.get('interp_method')),
                method_kws=cls.default_params.get(
                    'pia_solve_method_kws', cls.default_params.get('solve_method_kws')),
                info='--> Generating PIA model...'
            )
        else:
            pia_pts_nm = None
            pia_model = None

        if cls._use_wm(cls.default_params):
            # Rotate wm
            if cls._apply_rotation(cls.default_params):
                wm_pts_nm = cls.rotate_points(
                    cls.wm_pts_nm,
                    rotate_x=cls.default_params.get('rotate_x'),
                    rotate_y=cls.default_params.get('rotate_y'),
                    rotate_z=cls.default_params.get('rotate_z'),
                    info='--> Rotating WM...'
                )
            # Solve wm model
            wm_model = cls.solve(
                solve_method=cls.default_params.get(
                    'wm_solve_method', cls.default_params.get('solve_method')),
                features=wm_pts_nm[:, [0, 2]],
                targets=wm_pts_nm[:, [1]],
                poly_method=cls.default_params.get(
                    'wm_poly_method', cls.default_params.get('poly_method')),
                poly_model=cls.default_params.get(
                    'wm_poly_model', cls.default_params.get('poly_model')),
                poly_constants=cls.default_params.get(
                    'wm_poly_constants', cls.default_params.get('poly_constants')),
                interp_method=cls.default_params.get(
                    'wm_interp_method', cls.default_params.get('interp_method')),
                method_kws=cls.default_params.get(
                    'wm_solve_method_kws', cls.default_params.get('solve_method_kws')),
                info='--> Generating WM model...'
            )
        else:
            wm_pts_nm = None
            wm_model = None

        return cls._adjust(
            points_nm=points_nm,
            params=cls.default_params,
            pia_pts_nm=pia_pts_nm,
            wm_pts_nm=wm_pts_nm,
            pia_model=pia_model,
            wm_model=wm_model
        )

    @property
    def params(self):
        return self._params

    def __init__(self, **param_dict):
        """
        See class documentation for usage
        """
        self._params = self.default_params.copy()
        self.params.update(param_dict)

        if self._use_pia(self.params):
            # load pia pts
            self.pia_pts_nm = np.load(
                Path(self.params.get('npy_path_to_pia_pts')))
            
            if self._apply_rotation(self.params):
                # Rotate pia
                self.pia_pts_nm = self.rotate_points(
                    self.pia_pts_nm,
                    rotate_x=self.params.get('rotate_x'),
                    rotate_y=self.params.get('rotate_y'),
                    rotate_z=self.params.get('rotate_z'),
                    info='--> Rotating PIA...'
                )

            # Solve pia model
            self.pia_model = self.solve(
                solve_method=self.params.get(
                    'pia_solve_method', self.params.get('solve_method')),
                features=self.pia_pts_nm[:, [0, 2]],
                targets=self.pia_pts_nm[:, [1]],
                poly_method=self.params.get(
                    'pia_poly_method', self.params.get('poly_method')),
                poly_model=self.params.get(
                    'pia_poly_model', self.params.get('poly_model')),
                poly_constants=self.params.get(
                    'pia_poly_constants', self.params.get('poly_constants')),
                interp_method=self.params.get(
                    'pia_interp_method', self.params.get('interp_method')),
                method_kws=self.params.get(
                    'pia_solve_method_kws', self.params.get('solve_method_kws')),
                info='--> Generating PIA model...'
            )
        else:
            self.pia_pts_nm = None
            self.pia_model = None

        if self._use_wm(self.params):
            # load wm pts
            self.wm_pts_nm = np.load(
                Path(self.params.get('npy_path_to_wm_pts')))

            if self._apply_rotation(self.params):
                # Rotate wm
                self.wm_pts_nm = self.rotate_points(
                    self.wm_pts_nm,
                    rotate_x=self.params.get('rotate_x'),
                    rotate_y=self.params.get('rotate_y'),
                    rotate_z=self.params.get('rotate_z'),
                    info='--> Rotating WM...'
                )
            # Solve wm model
            self.wm_model = self.solve(
                solve_method=self.params.get(
                    'wm_solve_method', self.params.get('solve_method')),
                features=self.wm_pts_nm[:, [0, 2]],
                targets=self.wm_pts_nm[:, [1]],
                poly_method=self.params.get(
                    'wm_poly_method', self.params.get('poly_method')),
                poly_model=self.params.get(
                    'wm_poly_model', self.params.get('poly_model')),
                poly_constants=self.params.get(
                    'wm_poly_constants', self.params.get('poly_constants')),
                interp_method=self.params.get(
                    'wm_interp_method', self.params.get('interp_method')),
                method_kws=self.params.get(
                    'wm_solve_method_kws', self.params.get('solve_method_kws')),
                info='--> Generating WM model...'
            )
        else:
            self.wm_pts_nm = None
            self.wm_model = None

    def adjust(self, points_nm):
        """
        Adjust points with initialized parameters

        Args
            points_nm (Nx3 array)
                EM coordinates to adjust in nanometers
        Returns
            adjusted points (Nx3 array)
        """
        logger.info('--> Adjusting points with initialized params')
        return self._adjust(
            points_nm=points_nm,
            params=self.params,
            pia_pts_nm=self.pia_pts_nm,
            wm_pts_nm=self.wm_pts_nm,
            pia_model=self.pia_model,
            wm_model=self.wm_model)

    def make_model_grid(self, source, step=25000, npts=None, grid=None, normalize=False):
        """
        Run x-z grid through pia or wm models

        Args
            source (str)
                model to use
                options:
                    'pia' - use pia_model
                    'wm' - use wm_model
            step (None or int or tuple)
                desired spacing of grid points in nanometers. 
                    int will be applied to both x-z axes, or pass 
                    tuple with shape (2,) to adjust each axis separately
                if step is not None, npts must be None
            npts (int or None)
                the number of grid points to make
                    int will be applied to both x-z axes, or pass
                    tuple with shape (2,) to adjust each axis separately
                if npts is not None, step must be None
            grid (None or Nx2 array)
                desired grid to pass through model
                    If provided, step and npts will be ignored
        Returns 
            grid points (Nx3 array)
        """
        assert source in ('pia', 'wm'), f'model source {source} not available'
        model = getattr(self, '_'.join([source, 'model']))
        assert model is not None, f'No {source} model found.'
        if grid is None:
            bounds = get_bounding_box(source=self.params.get('normalize_bounds'))
            grid = make_grid(
                bounds=bounds,
                axis=(0, 2),
                step=step,
                npts=npts
            ).reshape(-1, 2)
        model_grid = np.hstack([
            grid[:, [0]],
            model.run(grid),
            grid[:, [1]]
        ])
        if normalize:
            model_grid = self._adjust(
                points_nm=model_grid, 
                params=self.params, 
                pia_pts_nm=self.pia_pts_nm,
                wm_pts_nm=self.wm_pts_nm,
                pia_model=self.pia_model,
                wm_model=self.wm_model,
                disable_rotation=True
            )
        return model_grid
