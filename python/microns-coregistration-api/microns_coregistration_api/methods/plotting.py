"""
Plotting functions
"""

import matplotlib.pyplot as plt
from .boundaries import seg_bounds_nm

def em_fig(
    proj=1, 
    bounds_to_use=None, 
    plot_kws=None,
    subplots_kws=None, 
):
    bounds = seg_bounds_nm if bounds_to_use is None else bounds_to_use
    
    plot_kws = {} if plot_kws is None else plot_kws
    plot_kws.setdefault('zoom', 0.9)
    plot_kws.setdefault('ticks_on', True)
    plot_kws.setdefault('tight_layout', True)
    
    subplots_kws = {} if subplots_kws is None else subplots_kws
    subplots_kws.setdefault('dpi', 250)
    subplots_kws.setdefault('figsize', (8,7))
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, **subplots_kws)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(*bounds[:, 0])
    ax.set_ylim(*bounds[:, 1])
    ax.set_zlim(*bounds[:, 2])
    
    if plot_kws.get('ticks_on'):
        if plot_kws.get('xticks') is not None:
            ax.set_xticks(plot_kws.get('xticks'))
        if plot_kws.get('yticks') is not None:
            ax.set_yticks(plot_kws.get('yticks'))
        if plot_kws.get('zticks') is not None:
            ax.set_zticks(plot_kws.get('zticks'))
    else:
        ax.set_xticks([]) 
        ax.set_yticks([]) 
        ax.set_zticks([])

    ax.invert_yaxis()
    ax.set_box_aspect(bounds[1] - bounds[0], zoom=plot_kws.get('zoom'))
    
    if plot_kws.get('tight_layout'):
        fig.set_tight_layout('tight')
    
    # VIEW
    if isinstance(proj, dict):
        ax.view_init(
            elev=proj.get('elev') or 0, 
            azim=proj.get('azim') or 0, 
            roll=proj.get('roll') or 0,
            vertical_axis=proj.get('vertical_axis') or 'z'
        )
    elif proj == 1:
        ## xy v1-->hva
        ax.view_init(elev=90, azim=90, roll=180)
    elif proj == 2:
        ## zy - v1 front
        ax.view_init(elev=0, azim=180, roll=270)
    elif proj == 3:
        ## xz - top down
        ax.view_init(elev=0, azim=90, roll=180)
    elif proj == 4:
        ## xy hva--> v1
        ax.view_init(elev=270, azim=270, roll=180)
    elif proj == 5:
        ## zy - hva front
        ax.view_init(elev=0, azim=0, roll=90)
    elif proj == 6:
        ## xz - bottom up
        ax.view_init(elev=180, azim=90, roll=180)
    else:
        raise AttributeError(f'proj {proj} not recognized')
    
    return fig, ax