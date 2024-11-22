# microns-coregistration
microns functional / EM data coregistration

### WARNING: 
Python native blobs get enabled for DataJoint when api is imported.

# Install API
```
pip install git+https://github.com/cajal/microns-coregistration.git#subdirectory=python/microns-coregistration-api
```

# Install API with specified tag
```
pip install git+https://github.com/cajal/microns-coregistration.git@TAG#subdirectory=python/microns-coregistration-api
```

# Using the AIBS transform
adapted from https://github.com/alleninstitute/em_coregistration/tree/phase3

```python
from microns_coregistration_api.methods import fetch_aibs_transform, aibs_coreg_transform
```
There are 8 transforms indexed by transform_id:

    1. phase2 	2P->EM 	  spline  (deprecated)
    2. phase2 	EM->2P 	  spline  (deprecated)
    3. phase2 	2P->EM 	  linear  (deprecated)
    4. phase2 	EM->2P 	  linear  (deprecated)
    5. phase3 	2P->EM 	  spline
    6. phase3 	EM->2P 	  spline
    7. phase3 	2P->EM 	  linear
    8. phase3 	EM->2P 	  linear

The "full" transform as designed by AIBS is type "spline" (transform_id's 5 and 6).
A more rigid version of the transform is type "linear" (transform_id's 7 and 8)

## Usage: 

Inputs to the EM->2P direction are EM coordinates with voxel resolution (4nm x 4nm x 40nm), and outputs are units of microns in the 2P Stack space (1um x 1um x 1um)

Inputs to the 2P->EM direction are 2P coordinates in 2P stack space with voxel resolution (1um x 1um x 1um), and outputs are units of EM voxels (4nm x 4nm x 40nm)


```python
em_coords_vx = aibs_coreg_transform(tp_coords_um, **fetch_aibs_transform(transform_id=5))
tp_coords_um = aibs_coreg_transform(em_coords_vx, **fetch_aibs_transform(transform_id=6))
em_coords_vx = aibs_coreg_transform(tp_coords_um, **fetch_aibs_transform(transform_id=7))
tp_coords_um = aibs_coreg_transform(em_coords_vx, **fetch_aibs_transform(transform_id=8))
```

# Adjusting EM Coordinates
```python
from microns_coregistration_api.methods import M65Adjust
```

The following transforms can be sequentially applied to adjust user-provided 
    coordinates in units of nanometers:

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

```python
M65Adjust.default_adjust(points)
```

To adjust with rotation only, ignoring all defaults:

```python
M65Adjust.rotate_points(points, rotate_z=3.5)
```

To adjust with user-specified params:

NOTE: Any params not provided by user will use default params

Override default params (Recommended):

```python
tform = M65Adjust(**param_dict)
tform.adjust(points)
```

Overwrite default params:

```python
M65Adjust.update_default_params(**param_dict)
M65Adjust.default_adjust(points)
```

Restore default params:
```python
M65Adjust.restore_default_params()
```

To access the manually placed pia and wm points:

```python
M65Adjust.pia_pts_nm
M65Adjust.wm_pts_nm
```

For advanced usage see `M65Adjust` class documentation