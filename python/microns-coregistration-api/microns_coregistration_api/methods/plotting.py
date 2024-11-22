"""
Plotting functions
"""
import numpy as np
import matplotlib.pyplot as plt
from .boundaries import tp_bounds_um, em_bounds_nm, seg_bounds_nm


def fig_3d(
    proj=1, 
    bounds='em_seg', 
    elev=None,
    azim=None,
    roll=None,
    vertical_axis='z',
    zoom=0.9,
    ticks_off=False,
    invert_yaxis=True,
    **fig_kws, 
):
    """
    
    """
    if isinstance(bounds, str):
        if bounds == 'em':
            bounds = em_bounds_nm
        elif bounds == 'em_seg':
            bounds = seg_bounds_nm
        elif bounds == 'tp':
            bounds = tp_bounds_um
        else:
            raise AttributeError(f'bounds {bounds} not recognized.')
    else:
        bounds = bounds
    
    fig_kws.setdefault('dpi', 150)
    fig_kws.setdefault('figsize', (8,7))
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, **fig_kws)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(*bounds[:, 0])
    ax.set_ylim(*bounds[:, 1])
    ax.set_zlim(*bounds[:, 2])
    
    if ticks_off:
        ax.set_xticks([]) 
        ax.set_yticks([]) 
        ax.set_zticks([])

    if invert_yaxis:
        ax.invert_yaxis()

    ax.set_box_aspect(bounds[1] - bounds[0], zoom=zoom)
    
    # VIEW
    if any([elev, azim, roll]):
        ax.view_init(
            elev=elev or 0, 
            azim=azim or 0, 
            roll=roll or 0,
            vertical_axis=vertical_axis
        )
    elif proj == 1:
        ## xy v1-->hva
        ax.view_init(elev=90, azim=90, roll=180, vertical_axis=vertical_axis)
    elif proj == 2:
        ## zy - v1 front
        ax.view_init(elev=0, azim=180, roll=270, vertical_axis=vertical_axis)
    elif proj == 3:
        ## xz - top down
        ax.view_init(elev=0, azim=90, roll=180, vertical_axis=vertical_axis)
    elif proj == 4:
        ## xy hva--> v1
        ax.view_init(elev=270, azim=270, roll=180, vertical_axis=vertical_axis)
    elif proj == 5:
        ## zy - hva front
        ax.view_init(elev=0, azim=0, roll=90, vertical_axis=vertical_axis)
    elif proj == 6:
        ## xz - bottom up
        ax.view_init(elev=180, azim=90, roll=180, vertical_axis=vertical_axis)
    else:
        raise AttributeError(f'proj {proj} not recognized')
    
    return fig, ax