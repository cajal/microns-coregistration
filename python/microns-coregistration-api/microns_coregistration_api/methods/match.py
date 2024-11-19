import numpy as np
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import LabelEncoder
from scipy.special import comb
import pandas as pd
import logging
from scipy.spatial.distance import cdist
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path


logger = logging.getLogger(__name__)

def proximity_match(a_array, b_array, max_match_dist, a_labels=None, b_labels=None):
    """
    Computes optimal nearest match assignment between 2 sets of centroids.
    
    Args:
        a_array (np.array): First set of NxD centroids.
        b_array (np.array): Second set of MxD centroids.
        max_match_dist (float): Maximum distance to consider a match.
        a_labels (np.array, optional): Labels of a to index for match array.
            If None, uses indices of a.
        b_labels (np.array, optional): Labels of b to index for match array.
            If None, uses indices of b.
            
    Returns:
        tuple: Contains three arrays:
            - Array of labels or indices in a_array with matches to b_array.
            - Array of labels or indices in b_array with matches to a_array.
            - Array of distances between matched a_centroid and b_centroid.
    """
    # Calculate the distance matrix between each pair of centroids
    dist_mat = distance_matrix(a_array, b_array)
    
    # Solve the linear sum assignment problem
    rows, cols = linear_sum_assignment(dist_mat)
    
    # Filter matches by max_match_dist
    match_mask = dist_mat[rows, cols] < max_match_dist
    
    # Handle None labels, default to indices if labels are not provided
    if a_labels is None:
        a_labels = np.arange(a_array.shape[0])
    if b_labels is None:
        b_labels = np.arange(b_array.shape[0])
    
    # Apply mask to select the matched labels/indices and residuals
    a_matches = a_labels[rows][match_mask]
    b_matches = b_labels[cols][match_mask]
    residuals = dist_mat[rows, cols][match_mask]
    
    return a_matches, b_matches, residuals

def add_residual_to_match_df(match_df, und_df):
    """
    Adds residual distance information to `match_df` given unit-nucleus distances from `und_df`

    Parameters:
    - match_df (pd.DataFrame): DataFrame containing matches with columns:
        'scan_session', 'scan_idx', 'field', 'unit_id', 'nucleus_id'
    - und_df (pd.DataFrame): DataFrame containing precomputed distances with columns:
        'scan_session', 'scan_idx', 'field', 'unit_id', 'nucleus_id', 'distance'

    Returns:
    - pd.DataFrame: The updated `match_df` with a new 'residual' renamed from 'distance' in `und_df` 

    Note:
        NaN values for the 'residual' can be returned if the distance for that row is not present in `und_df`.
    """
    # format dfs and ensure necessary columns are present
    match_df = match_df[['scan_session', 'scan_idx', 'field', 'unit_id', 'nucleus_id']]
    und_df = und_df[['scan_session', 'scan_idx', 'field', 'unit_id', 'nucleus_id', 'distance']]
    return match_df.merge(und_df, how='left', validate="1:1").rename(columns={'distance': 'residual'})


def add_separation_to_match_df(match_df, und_df):
    """
    Adds separation distance information to `match_df` given unit-nucleus distances from `und_df`. 
    The separation is calculated per field, both for nucleus and unit separation. 

    Parameters:
    - match_df (pd.DataFrame): DataFrame containing matches with columns:
        'scan_session', 'scan_idx', 'field', 'unit_id', 'nucleus_id', 'residual'
    - und_df (pd.DataFrame): DataFrame containing precomputed unit-nucleus distances with columns:
        'scan_session', 'scan_idx', 'field', 'unit_id', 'nucleus_id', 'distance'

    The function calculates two types of separation:
    - Nucleus separation (`nuc_sep`): The difference between the distance and residual for the nearest 
      non-matched nucleus_id to a given unit_id.
    - Unit separation (`unit_sep`): The difference between the distance and residual for the neareast
      non-matched unit_id to a given nucleus_id

    Returns:
    - pd.DataFrame: `match_df` with four new columns added:
        - 'nuc_sep' - nucleus separation
        - 'unit_sep' - unit separation
        - 'nucleus_id_sep' - The nucleus_id used to calculate `nuc_sep`
        - 'unit_id_sep' - The unit_id used to calculate `unit_sep`

    Note : 
        NaN values can be returned if provided residual is NaN or no nearby unit or nucleus is present in und_df to calculate separation
    """
    # format dfs and ensure necessary columns are present
    match_df = match_df[['scan_session', 'scan_idx', 'field', 'unit_id', 'nucleus_id', 'residual']]
    und_df = und_df[['scan_session', 'scan_idx', 'field', 'unit_id', 'nucleus_id', 'distance']]
    
    dfs = []
    # calculate separation per field
    for (ss, si, f), und_sub_df in und_df.groupby(['scan_session', 'scan_idx', 'field']):
        match_sub_df = match_df.query(f'scan_session == {ss} and scan_idx == {si} and field == {f}')
        s = match_sub_df.merge(und_sub_df, how='left', on=['scan_session', 'scan_idx', 'field', 'unit_id', 'nucleus_id']).drop(columns='distance')

        # add nucleus separation
        a = match_sub_df.merge(und_sub_df, on=['scan_session', 'scan_idx', 'field', 'unit_id'], suffixes=('_match', '_sep'))
        b = a.query('nucleus_id_match!=nucleus_id_sep')
        c = b.sort_values(by=['unit_id', 'nucleus_id_match', 'distance'])
        d = c.drop_duplicates(['unit_id', 'nucleus_id_match'])
        e = d.sort_values('unit_id')
        e['nuc_sep'] = e['distance'] - e['residual']
        f = e.drop(columns=['residual', 'distance']).rename(columns={'nucleus_id_match': 'nucleus_id'})
        s1 = s.merge(f, how='left', validate="1:1")

        # add unit separation
        a = match_sub_df.merge(und_sub_df, on=['scan_session', 'scan_idx', 'field', 'nucleus_id'], suffixes=('_match', '_sep'))
        b = a.query('unit_id_match!=unit_id_sep')
        c = b.sort_values(by=['nucleus_id', 'unit_id_match', 'distance'])
        d = c.drop_duplicates(['nucleus_id', 'unit_id_match'])
        e = d.sort_values('unit_id_match')
        e['unit_sep'] = e['distance'] - e['residual']
        f = e.drop(columns=['residual', 'distance']).rename(columns={'unit_id_match': 'unit_id'})
        s2 = s1.merge(f, how='left', validate="1:1")
    
        dfs.append(s2)
    return match_df.merge(pd.concat(dfs, 0))

def drop_match_disagreements(df, on='unit_id'):
    """
    Remove match disagreements from a match dataframe.

    Parameters:
    - df (DataFrame): A Pandas DataFrame with the following columns: 
        - 'scan_session', 'scan_idx', 'field', 'unit_id', 'nucleus_id'
    - on (str, optional): The column to test agreement:
        - 'unit_id': drop any units with at least two nuclei matches that disagree
        - 'nucleus_id': drop any nuclei at least two unit matches that disagree
        - 'both': same as 'unit_id' and 'nucleus_id'

    Returns:
    - DataFrame: A Pandas DataFrame with the same columns as the input dataframe, 
        but with all disagreement rows dropped.
    """
    def _drop(df, on):
        assert on in ['unit_id', 'nucleus_id', 'both'], 'on must be "unit_id" or "nucleus_id" or "both"'
        sattrs = ['scan_session', 'scan_idx', 'field', on]
        _, disagree_df, _ = split_match_attempts_by_agreement(df, on=on)
        return df.merge(disagree_df, on=sattrs, how='left', indicator=True).query('_merge=="left_only"').drop(columns='_merge')
    if on == 'both':
        return _drop(_drop(df, 'unit_id'), 'nucleus_id').merge(_drop(_drop(df, 'nucleus_id'), 'unit_id'))
    else:
        return _drop(df, on)
    

def sparse_distance_matrix(unit_df, nucleus_df, dmax):
    """
    Calculate the sparse distance matrix between units and nuclei based on a maximum distance threshold per axis.

    This function identifies nuclei that are within a specified maximum distance (`dmax`) from each unit
    along each axis (x, y, z) individually. It then computes the Euclidean distance between each unit and
    the filtered nuclei. The result includes only those unit-nucleus pairs where the nucleus is within `dmax`
    units of the unit along each axis separately. 

    Parameters:
    - unit_df (pd.DataFrame): A DataFrame containing unit information with the following columns:
      'scan_session', 'scan_idx', 'field', 'unit_id', 'unit_x', 'unit_y', 'unit_z'.
    - nucleus_df (pd.DataFrame): A DataFrame containing nucleus information with columns:
      'nucleus_id', 'nucleus_x', 'nucleus_y', 'nucleus_z'.
    - dmax (float): The maximum distance threshold along each axis (x, y, z) for considering a nucleus
      to be within range of a unit.

    Returns:
    - pd.DataFrame: A DataFrame with columns 'scan_session', 'scan_idx', 'field', 'unit_id', 'nucleus_id',
      and 'distance', where each row represents a unit-nucleus pair. The distance is the Euclidean distance
      between the unit and nucleus, included only if the nucleus is within `dmax` distance from the unit
      along each axis separately.

    Note:
    - The function filters nuclei by checking if their distance from a unit along each axis (x, y, z) is
      less than or equal to `dmax`, before calculating the Euclidean distances for the filtered pairs.
    - This approach ensures that only nuclei within a cube defined by `dmax` around each unit are considered.
    """
    unit_df = unit_df[['scan_session', 'scan_idx', 'field', 'unit_id', 'unit_x', 'unit_y', 'unit_z']]
    nucleus_df = nucleus_df[['nucleus_id', 'nucleus_x', 'nucleus_y', 'nucleus_z']]
    
    unit_labels = unit_df[['scan_session', 'scan_idx', 'field', 'unit_id']].values
    unit_coords = unit_df[['unit_x', 'unit_y', 'unit_z']].values
    nuc_coords = nucleus_df[['nucleus_x', 'nucleus_y', 'nucleus_z']].values 
    nuc_labels = nucleus_df.nucleus_id.values
    
    dfs = []
    for uxyz, (ss, si, f, u) in tqdm(zip(unit_coords, unit_labels), total=len(unit_coords)):
        xs, ys, zs = np.abs(uxyz - nuc_coords).T
        nearest_mask = (1 * (xs <= dmax) * 1 * (ys <= dmax) * 1*(zs <= dmax)).astype(bool)
        nucs = nuc_labels[nearest_mask]
        dxyz = (cdist(uxyz[None, :], nuc_coords[nearest_mask]))[0]
        # make dataframe
        inds = np.argsort(dxyz)
        dfs.append(
            pd.DataFrame({
                'scan_session': np.repeat(ss, len(inds)),
                'scan_idx': np.repeat(si, len(inds)),
                'field': np.repeat(f, len(inds)),
                'unit_id': np.repeat(u, len(inds)),
                'nucleus_id': nucs[inds],
                'distance': dxyz[inds],
            }))
    return pd.concat(dfs).reset_index(drop=True)


def split_match_attempts_by_agreement(df, on='unit_id'):
    """
    Split match dataframe by agreement on a given column.

    Given the columns 'scan_session', 'scan_idx', and 'field' and either 'unit_id' or 'nucleus_id',
        this function splits the dataframe into three dataframes based on agreement:
        - agree_df: The unique rows where more than one match attempt was made and all attempts agree.
        - disagree_df: The unique rows where more than one match attempt was made and any two or more attempts disagree.
        - total_df: The unique rows where more than one match attempt was made.

    Parameters:
    - df (DataFrame): A Pandas DataFrame with the following columns: 
        - 'scan_session', 'scan_idx', 'field', 'unit_id', 'nucleus_id'
    - on (str, optional): The column to test agreement:
        - 'unit_id': test whether multiple nuclei that were matched to a given a unit_id 
            are identical
        - 'nucleus_id': test whether multiple unit_ids that were matched to a given a nucleus_id 
            are identical

    Returns:
    A tuple with three pandas dataframes: (agree_df, disagree_df, total_df)
    """
    assert on in ['unit_id', 'nucleus_id'], 'on must be "unit_id" or "nucleus_id"'
    sattrs = ['scan_session', 'scan_idx', 'field', on]
    other = 'nucleus_id' if on == 'unit_id' else 'unit_id'

    # get all matches with more than 1 attempt
    total_df = df.groupby(sattrs, as_index=False).count().query(f'{other}>1')[sattrs]

    # get all matches with more than 1 attempt and those attempts disagree
    disagree_df = df.groupby(sattrs, as_index=False).nunique().query(f'{other}>1')[sattrs]
    
    # get agreements by subtracting disagree_df from total_df
    merged_df = total_df.merge(disagree_df, on=sattrs, how='left', indicator=True)
    agree_df = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge'])
    
    return agree_df, disagree_df, total_df


def compare_match_agreement(ref_df, test_df, keep_disagreements_from=None, keep_unique_attempts=False, disagree_types=None, names=None, verbose=False):
    ref_name = 'ref' if names is None else names[0]
    test_name = 'test' if names is None else names[1]

    for df, name in zip([ref_df, test_df], [ref_name, test_name]):
        assert not df.duplicated().any(), f'{name} dataframe has duplicate rows.'

    if disagree_types is None:
        disagree_types = [
            (1, 1, 1),
            (1, 1, 2),
            (1, 2, 1),
            (1, 2, 2),
            (3, 2, 2),
            (3, 2, 3),
            (3, 3, 2),
            (3, 3, 3),
        ]
        if not keep_unique_attempts:
            _ = disagree_types.pop(0)
            _ = disagree_types.pop(-1)
            
        if keep_disagreements_from is not None:
            if keep_disagreements_from == ref_name:
                disagree_types = [d for d in disagree_types if d[0] == 1]
            elif keep_disagreements_from == test_name:
                disagree_types = [d for d in disagree_types if d[0] == 3]
            else:
                raise ValueError(f'keep_disagreements_from "{keep_disagreements_from}" not recognized.')

    unit_key = ['scan_session', 'scan_idx', 'field', 'unit_id']
    nuc_key = ['scan_session', 'scan_idx', 'field', 'nucleus_id']
    match_key = unit_key + ['nucleus_id']

    match_merge_df = ref_df.merge(test_df, on=match_key, how='outer', indicator='match_type')[match_key + ['match_type']]
    match_merge_df['match_type'] = match_merge_df['match_type'].map({'left_only': 1, 'both': 2, 'right_only': 3})
    
    unit_merge_df = ref_df.merge(test_df, on=unit_key, how='outer', indicator='unit_type')[unit_key + ['unit_type']]
    unit_merge_df['unit_type'] = unit_merge_df['unit_type'].map({'left_only': 1, 'both': 2, 'right_only': 3})
    
    nuc_merge_df = ref_df.merge(test_df, on=nuc_key, how='outer', indicator='nuc_type')[nuc_key + ['nuc_type']]
    nuc_merge_df['nuc_type'] = nuc_merge_df['nuc_type'].map({'left_only': 1, 'both': 2, 'right_only': 3})
    
    comb_df = match_merge_df.merge(unit_merge_df).merge(nuc_merge_df)
    
    query_type = lambda mt, ut, nt: f'match_type=={mt} and unit_type=={ut} and nuc_type=={nt}'
    
    agree_match_df = comb_df.query(query_type(*(2, 2, 2))).merge(ref_df)
    agree_match_df['source'] = 'both'
    
    disagree_dfs = []
    for dt in disagree_types:
        sub_df = comb_df.query(query_type(*dt))
        if verbose:
            print(dt, len(sub_df))
        disagree_dfs.append(sub_df)
    disagree_match_df = pd.concat(disagree_dfs)
    
    disagree_ref_match_df = disagree_match_df.merge(ref_df)
    disagree_ref_match_df['source'] = ref_name
    
    disagree_test_match_df = disagree_match_df.merge(test_df)
    disagree_test_match_df['source'] = test_name
    
    disagree_match_df = pd.concat([
        disagree_ref_match_df,
        disagree_test_match_df
    ])
    
    total_match_df = pd.concat([
        agree_match_df,
        disagree_match_df
    ])

    return agree_match_df, disagree_match_df, total_match_df


def evaluate_match_self_agreement(df, on='unit_id'):
    """
    Split match dataframe by agreement with split_match_attempts_by_agreement 
        and return len(agree_df) and len(total_df).

    Parameters:
    - df (DataFrame): A Pandas DataFrame with the following columns: 
        - 'scan_session', 'scan_idx', 'field', 'unit_id', 'nucleus_id'
    - on (str, optional): The column to test agreement:
        - 'unit_id': test whether multiple nuclei matched to a given a unit_id 
            are identical
        - 'nucleus_id': test whether multiple unit_ids matched to a given a nucleus_id 
            are identical
    """
    agree_df, _, total_df = split_match_attempts_by_agreement(df, on=on)
    return len(agree_df), len(total_df)


def plot_match_agree_vs_disagree_hist(
    match_df, 
    on, 
    metric,
    groupby_method='mean',
    agree_color='blue',
    disagree_color='orange',
    xlabel=None,
    ylabel=None,
    xtick_params=None,
    ytick_params=None,
    hist_kws=None, 
    title=None,
    legend=True,
    despine=True,
    save_dir=None,
    save_fn=None,
    savefig_kws=None
):    
    agree_df, disagree_df, _ = split_match_attempts_by_agreement(match_df, on=on)
    agree_groupby_df = match_df.merge(agree_df).groupby(['scan_session', 'scan_idx', 'field', on], as_index=False)
    disagree_groupby_df = match_df.merge(disagree_df).groupby(['scan_session', 'scan_idx', 'field', on], as_index=False)
    
    if groupby_method == 'mean':
        agree_match_df = agree_groupby_df.mean()
        disagree_match_df = disagree_groupby_df.mean()
    elif groupby_method == 'sample':
        agree_match_df = agree_groupby_df.apply(lambda g: g.sample(1))
        disagree_match_df = disagree_groupby_df.apply(lambda g: g.sample(1))
    else:
        raise ValueError('groupby_method not recognized. Must be "mean" or "sample".')


    hist_kws = {} if hist_kws is None else hist_kws
    hist_kws.setdefault('bins', 25)
    hist_kws.setdefault('density', True)
    hist_kws.setdefault('alpha', 0.5)
    
    fig, ax = plt.subplots()
                        
    _ = ax.hist(agree_match_df[metric], color=agree_color, label='agree', **hist_kws)
    _ = ax.hist(disagree_match_df[metric], color=disagree_color, label='disagree', **hist_kws)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)
    
    if xtick_params is not None:
        xtick_start, xtick_end, xtick_spacing = xtick_params
        ax.set_xlim(xtick_start, xtick_end)
        ax.set_xticks(np.arange(xtick_start, xtick_end, xtick_spacing))
    
    if ytick_params is not None:
        ytick_start, ytick_end, ytick_spacing = ytick_params
        ax.set_ylim(ytick_start, ytick_end)
        ax.set_yticks(np.arange(ytick_start, ytick_end, ytick_spacing))
    
    if title is not None:
        ax.set_title(title)
    
    if legend:
        ax.legend()
    
    if despine:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    if save_dir is not None:
        savefig_kws = {} if savefig_kws is None else savefig_kws
        save_fn = f'agree_vs_disagree_dist_on_{on}_{metric}.png' if save_fn is None else save_fn
        savepath = Path(save_dir).joinpath(save_fn)
        savefig_kws.setdefault('bbox_inches', 'tight')
        savefig_kws.setdefault('pad_inches', 0.5)
        fig.savefig(savepath, **savefig_kws)

    return fig, ax


def plot_fraction_agreement_by_metric(
    match_df,
    on,
    metric,
    bin_params,
    min_total,
    groupby_method='mean',
    bar_color='skyblue',
    bar_edge_color = 'k',
    add_annotations=True,
    annotation_fs=8,
    anno_kws=None,
    xlabel=None,
    ylabel=None,
    xtick_params=None,
    ytick_params=None,
    add_ygrid=True,
    title=None,
    despine=True,
    save_dir=None,
    save_fn=None,
    savefig_kws=None
):

    agree_df, _, total_df = split_match_attempts_by_agreement(match_df, on=on)
    agree_groupby_df = match_df.merge(agree_df).groupby(['scan_session', 'scan_idx', 'field', on], as_index=False)
    total_groupby_df = match_df.merge(total_df).groupby(['scan_session', 'scan_idx', 'field', on], as_index=False)

    if groupby_method == 'mean':
        agree_match_df = agree_groupby_df.mean()
        total_match_df = total_groupby_df.mean()
    elif groupby_method == 'sample':
        agree_match_df = agree_groupby_df.apply(lambda g: g.sample(1))
        total_match_df = total_groupby_df.apply(lambda g: g.sample(1))
    else:
        raise ValueError('groupby_method not recognized. Must be "mean" or "sample".')

    bin_start, bin_end, bin_width = bin_params
    bins = np.arange(bin_start, bin_end, bin_width)
    agree_counts, agree_bins = np.histogram(
        agree_match_df[metric][~np.isnan(agree_match_df[metric])], bins=bins
    )
    total_counts, total_bins = np.histogram(
        total_match_df[metric][~np.isnan(total_match_df[metric])], bins=bins
    )

    total_mask = total_counts >= min_total
    agree_counts = agree_counts[total_mask]
    total_counts = total_counts[total_mask]
    agree_bins = agree_bins[:-1][total_mask]
    total_bins = total_bins[:-1][total_mask]
    
    fig, ax = plt.subplots()
    bars = ax.bar(
        agree_bins + bin_width / 2, 
        agree_counts / total_counts, 
        width=bin_width, 
        edgecolor=bar_edge_color, 
        color=bar_color, 
        zorder=3
    )
    if add_annotations:
        for a, t, bar in zip(agree_counts, total_counts, bars):
            xy_bot = (bar.get_x() + bar.get_width() / 2, 0)
            xy_top = (bar.get_x() + bar.get_width() / 2, bar.get_height())
            anno_kws = {} if anno_kws is None else anno_kws
            anno_kws.setdefault('textcoords', 'offset points')
            anno_kws.setdefault('ha', 'center')
            anno_kws.setdefault('va', 'bottom')
            anno_kws.setdefault('fontsize', annotation_fs)
            ax.annotate(
                f'{int(a)/int(t):.2f}',
                xy=xy_top,
                xytext=(0, 3),
                **anno_kws
            )
            ax.annotate(
                f'{int(a)}',
                xy=xy_bot,
                xytext=(0, 20),
                **anno_kws
            )
            ax.annotate(
                f'____',
                xy=xy_bot,
                xytext=(0, 15),
                **anno_kws
            )
            ax.annotate(
                f'{int(t)}',
                xy=xy_bot,
                xytext=(0, 3),
                **anno_kws
            )

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)
    
    if xtick_params is not None:
        xtick_start, xtick_end, xtick_spacing = xtick_params
        ax.set_xlim(xtick_start, xtick_end)
        ax.set_xticks(np.arange(xtick_start, xtick_end, xtick_spacing))
    
    if ytick_params is not None:
        ytick_start, ytick_end, ytick_spacing = ytick_params
        ax.set_ylim(ytick_start, ytick_end)
        ax.set_yticks(np.arange(ytick_start, ytick_end, ytick_spacing))
    
    if add_ygrid:
        ax.grid(axis='y', zorder=1)

    if title is not None:
        ax.set_title(title)
    
    if despine:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    if save_dir is not None:
        savefig_kws = {} if savefig_kws is None else savefig_kws
        save_fn = f'fraction_agreement_on_{on}_{metric}.png' if save_fn is None else save_fn
        savepath = Path(save_dir).joinpath(save_fn)
        savefig_kws.setdefault('bbox_inches', 'tight')
        savefig_kws.setdefault('pad_inches', 0.5)
        fig.savefig(savepath, **savefig_kws)

    return fig, ax


def repetition_score(data, axis=None):
    """
    Calculate the repetition score of the data along a specified axis.
    
    :param data: List or NumPy array of data.
    :param axis: Axis along which to calculate repetition. 
                 If None, the data is flattened.
    """
    data = np.array(data)  # Ensure data is a NumPy array
    
    if axis is None:
        n = data.size
        nu = len(np.unique(data))
    else:
        # For multidimensional data, count unique elements along the specified axis.
        n = data.shape[axis]
        nu = np.unique(data, axis=axis).shape[axis]
    
    return 1 - ((nu - 1) / (n - 1))

def identical_combination_score(data, k=2, axis=None):
    """
    Calculate the identical combination score of a dataset.

    This function determines the likelihood that a randomly chosen subset of k items
    from the dataset consists of identical elements. It achieves this by computing the
    proportion of possible combinations of k identical items out of the total possible
    combinations of k items from the entire dataset.

    Parameters:
    data (array_like): An array of data points, possibly with repetitions.
    k (int): The number of items to choose from the dataset. Default is 2.
    axis: The axis to compute score. Passed to np.unique.

    Returns:
    float: The combination score, which is the ratio of the sum of combinations of k
    identical items from each unique value to the total combinations of k items from
    the dataset. Returns 0 if comb(n, k) is zero, where n is the total count of items.

    Notes:
    - The score is 0 if `k` is greater than the number of items in `data`.
    - This function uses scipy.special.comb to compute binomial coefficients, which
      can handle large numbers and arrays efficiently.
    """
    _, counts = np.unique(data, return_counts=True, axis=axis)
    total_combinations = comb(counts.sum(), k)
    if total_combinations == 0:
        return 0
    return np.sum([comb(c, k) for c in counts if c >= k]) / total_combinations

def add_composite_id(df, keys, name='composite_id'):
    """
    Adds a new column to the DataFrame that contains a composite identifier built from specified key columns.

    This function creates a new column in the DataFrame by concatenating the string representations of values
    from specified key columns. This composite identifier can be used to uniquely identify rows based on the
    combination of these values.

    Parameters:
        df (pd.DataFrame): The DataFrame to which the ID will be added.
        keys (list of str): A list of column names whose values will be concatenated to form the composite identifier.
        name (str, optional): The name of the new column to hold the composite identifier. Defaults to 'composite_id'.

    Returns:
        pd.DataFrame: The modified DataFrame with an additional column containing the composite identifiers.

    Example:
        >>> df = pd.DataFrame({
        >>>     'scan_session': [1, 1],
        >>>     'scan_idx': [101, 102],
        >>>     'field': [1, 2],
        >>>     'unit_id': [3001, 3002]
        >>> })
        >>> add_composite_id(df, keys=['scan_session', 'scan_idx', 'field', 'unit_id'], name='func_id')
    """
    df[name] = df[keys].apply(lambda r: '_'.join([str(r[k]) for k in keys]), axis=1)
    return df

def custom_pandas_groupby_agg(df, by, agg, **agg_kws):
    """
    Performs grouped aggregation on a DataFrame with additional custom aggregation functions.

    Parameters:
        df (pd.DataFrame): The DataFrame to perform the aggregation on.
        by (str or list): The column(s) to group the DataFrame by.
        agg (str): The column to perform aggregations on.
        **agg_kws: Additional keyword arguments for custom aggregations.

    Returns:
        pd.DataFrame: A DataFrame containing the count, number of unique values, and any
                      custom aggregations specified by `agg_kws` for the grouped column.
    """
    return df.groupby(by)[agg].agg(
        count='count',
        nunique='nunique',
        **agg_kws
    )


def compute_unanimity_score_on_df(df, by, agg):
    """
    Computes a unanimity score for each group in the DataFrame, indicating whether all
    entries in a group are identical based on the specified aggregation column.

    Parameters:
        df (pd.DataFrame): The DataFrame to compute the score on.
        by (str or list): The column(s) to group the DataFrame by.
        agg (str): The column to apply the aggregation to.

    Returns:
        pd.DataFrame: The grouped DataFrame with an added 'un_score' column where a score of 1
                      indicates all aggregated entries are identical, and 0 otherwise.
    """
    result_df = custom_pandas_groupby_agg(
        df=df,
        by=by,
        agg=agg,
    )
    result_df['un_score'] = 1 * (result_df['nunique'] <= 1)
    return result_df


def compute_identical_combination_score_on_df(df, by, agg):
    """
    Computes a score for each group in the DataFrame based on
    'identical_combination_score', which assesses the likelihood 
    that a randomly chosen subset of k items consists of identical elements.

    Parameters:
        df (pd.DataFrame): The DataFrame to compute the score on.
        by (str or list): The column(s) to group the DataFrame by.
        agg (str): The column to apply the aggregation to.

    Returns:
        pd.DataFrame: The grouped DataFrame with additional custom aggregation column 
            called "ic_score" that has the result of calling
            'identical_combination_score' on the aggregated entries.
    """
    return custom_pandas_groupby_agg(
        df=df,
        by=by,
        agg=agg,
        ic_score=lambda x: identical_combination_score(x)
    )


def agreement_score_on_multi_attempted_attrs(df, by, agg, method):
    """
    Computes an agreement score for a DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame to calculate the score on.
        by (str or list): The column(s) to group the DataFrame by.
        agg (str): The column to apply the aggregation to.
        method (str): The method to use for calculating the agreement score, supported methods
                      include 'un_score' and 'ic_score'.

    Returns:
        tuple: A tuple containing the average score of the specified method for groups with more
               than one entry, and the result DataFrame with the calculated scores.

    Raises:
        AssertionError: If the specified method is not recognized.
    """
    methods = {
        'un_score': compute_unanimity_score_on_df,
        'ic_score': compute_identical_combination_score_on_df
    }
    assert method in methods, f'method {method} not recognized.'

    result_df = methods[method](df=df, by=by, agg=agg).query('count>1')
    return result_df[method].mean(), result_df 


def filter_dataframes_by_common_attrs(df1, df2, common_attrs):
    """
    Filters and returns rows from two dataframes based on common values in specified attributes.

    This function identifies common rows between two dataframes based on two specified attributes. 
    It checks each attribute independently in both dataframes and returns rows from each dataframe 
    that have at least one identical attribute value in the other dataframe.

    Parameters:
    - df1 (pd.DataFrame): The first dataframe to compare.
    - df2 (pd.DataFrame): The second dataframe to compare.
    - common_attrs (tuple of str): A tuple containing the names of the two attributes to compare. 
      The first element is the attribute used from `df1` to compare against `df2` and vice versa 
          for the second attribute.

    Returns:
    - tuple of pd.DataFrame: A tuple containing two dataframes. 
        The first dataframe contains rows from `df1` that have common values in either of the 
            specified attributes with `df2`. 
        The second dataframe contains rows from `df2` that have common values in either of 
            the specified attributes with `df1`.
    """
    attr1, attr2 = common_attrs
    mask1 = df1[attr1].isin(df2[attr1]) | df1[attr2].isin(df2[attr2])
    mask2 = df2[attr1].isin(df1[attr1]) | df2[attr2].isin(df1[attr2])
    return df1[mask1], df2[mask2]



def filter_dataframes_by_common_attrs_and_merge(left, right, common_attrs, how='outer', **merge_kws):
    """
    Filters two dataframes for common attributes and then merges them according to the specified join method.

    This function first filters two dataframes by checking if they share common values in specified attributes,
    then merges the filtered dataframes using the specified type of join. Additional keyword arguments for the merge
    can be passed.

    Parameters:
        left (pd.DataFrame): The left dataframe to merge.
        right (pd.DataFrame): The right dataframe to merge.
        common_attrs (tuple of str): Attributes based on which the dataframes are first filtered and then merged.
        how (str, optional): Type of merge to be performed (e.g., 'inner', 'outer', 'left', 'right'). Defaults to 'outer'.
        **merge_kws: Additional keyword arguments to be passed to pandas merge function.

    Returns:
        pd.DataFrame: The merged dataframe after filtering for common attributes.
    """
    return pd.merge(
        *filter_dataframes_by_common_attrs(left, right, common_attrs=common_attrs), 
        on=common_attrs, 
        how='outer', 
        indicator=True,
        **merge_kws
    )

# def match_agreement_score_on_common_attrs(ref_df, test_df, match_attrs=('func_id', 'em_id'), score_method='recall'):
#     """
#     Calculates the agreement score based on the overlap of common attributes between two dataframes.

#     This function merges two dataframes based on common attributes and then calculates the agreement
#     score as the proportion of matching entries to the total entries from the reference dataframe that
#     are present in the merged dataframe.

#     Parameters:
#         ref_df (pd.DataFrame): The reference dataframe.
#         test_df (pd.DataFrame): The test dataframe to compare with the reference dataframe.
#         match_attrs (tuple of str): The attributes used for matching the dataframes.
#         score_method (str, optional): The method to use for calculating the agreement score.
#             Supported methods include 'recall' and 'precision'. Defaults to 'recall'.
#     Returns:
#         float: The agreement score, calculated as the ratio of matching entries (present in both dataframes)
#                to the total entries from the reference dataframe present in the merged dataframe.

#     """
#     merge_df = filter_dataframes_by_common_attrs_and_merge(
#         ref_df,
#         test_df,
#         common_attrs=match_attrs
#     )
#     if score_method == 'recall':
#         score = len(merge_df.query('_merge=="both"')) / len(merge_df.query('_merge=="both" or _merge=="left_only"'))
#     elif score_method == 'precision':
#         score = len(merge_df.query('_merge=="both"')) / len(merge_df.query('_merge=="both" or _merge=="right_only"'))
#     else:
#         raise ValueError(f'score_method {score_method} not recognized.')
#     return score, merge_df

def match_agreement_score_on_common_attrs(ref_df, test_df, match_attrs=('func_id', 'em_id'), score_method='recall'):
    """
    Calculates the agreement score based on the overlap of common attributes between two dataframes.

    This function merges two dataframes based on common attributes and then calculates the agreement
    score as the proportion of matching entries to the total entries from the reference dataframe that
    are present in the merged dataframe.

    Parameters:
        ref_df (pd.DataFrame): The reference dataframe.
        test_df (pd.DataFrame): The test dataframe to compare with the reference dataframe.
        match_attrs (tuple of str): The attributes used for matching the dataframes.
        score_method (str, optional): The method to use for calculating the agreement score.
            Supported methods include 'recall' and 'precision'. Defaults to 'recall'.
    Returns:
        float: The agreement score, calculated as the ratio of matching entries (present in both dataframes)
               to the total entries from the reference dataframe present in the merged dataframe.

    """
    
    if score_method == 'recall':
        merge_df = pd.merge(
            ref_df,
            test_df,
            on=match_attrs, 
        )
        score = len(merge_df) / len(ref_df)
    
    elif score_method == 'precision':
        merge_df = filter_dataframes_by_common_attrs_and_merge(
            ref_df,
            test_df,
            common_attrs=match_attrs
        )
        score = len(merge_df.query('_merge=="both"')) / len(merge_df.query('_merge=="both" or _merge=="right_only"'))
    else:
        raise ValueError(f'score_method {score_method} not recognized.')
    return score, merge_df


def compute_match_score_between_df1_and_filtered_df2_single_metric(df1, df2, metric, min_value, max_value, bin_width, cumulative_operator=None, score_method='recall'):
    """
    Compute the match agreement score between two DataFrames based on a single metric over a range of values.

    This function iterates over a range of values defined by `min_value` and `max_value`, with each iteration
    filtered by `bin_width`. For each bin, it computes the match agreement score by comparing `df1` with the filtered
    `df2` based on the specified `metric`. If `cumulative_operator` is provided, it computes the score cumulatively.

    Parameters:
    - df1 (DataFrame): The first DataFrame to compare.
    - df2 (DataFrame): The second DataFrame to filter and compare against `df1`.
    - metric (str): The column name in `df2` to use for filtering.
    - min_value (int/float): The minimum value to start binning from.
    - max_value (int/float): The maximum value to end binning.
    - bin_width (int/float): The width of each bin.
    - cumulative_operator (str, optional): The operator ('<=', '>=', etc.) to apply for cumulative filtering. If None, non-cumulative bins are used.
    - score_method (str, optional): The method to use for calculating the agreement score. Supported methods include 'recall' and 'precision'.

    Returns:
    - bins (list): The midpoints of the bins used for filtering.
    - scores (list): The match agreement scores for each bin.
    - len_df2s (list): The number of entries in `df2` for each bin.
    """
    
    bins = []
    scores = []
    filtered_df2s = []

    for b in np.arange(min_value, max_value, bin_width):
        if cumulative_operator is None:
            filtered_df2 = df2.query(f'{metric} >= @b and {metric} < @b + @bin_width')
        else:
            filtered_df2 = df2.query(f'{metric} {cumulative_operator} @b')
        
        try:
            score, _ = match_agreement_score_on_common_attrs(df1, filtered_df2, score_method=score_method)
        except ZeroDivisionError:
            score = np.nan
        
        scores.append(score)
        # bins.append(b + bin_width / 2)
        bins.append(b)
        filtered_df2s.append(filtered_df2)
        
    return bins, scores, filtered_df2s


def compute_match_score_between_df1_and_filtered_df2_joint_metric(
    df1, df2, metric1, metric2, min_value1, max_value1, bin_width1, min_value2, max_value2, bin_width2, cumulative_operator1=None, cumulative_operator2=None, score_method='recall'
):
    """
    Compute the match agreement score between two DataFrames based on joint metrics over a range of values.

    This function creates a 2D grid based on two metrics and their respective ranges and bin widths.
    It then filters `df2` for each grid cell and computes the match agreement score by comparing `df1`
    with the filtered `df2`. Cumulative filtering can be applied by specifying both `cumulative_operator1` and
    `cumulative_operator2`. If only one of `cumulative_operator1` or `cumulative_operator2` is provided, it 
    is ignored. 

    Parameters:
    - df1 (DataFrame): The first DataFrame to compare.
    - df2 (DataFrame): The second DataFrame to filter and compare against `df1`.
    - metric1 (str): The first column name in `df2` to use for filtering.
    - metric2 (str): The second column name in `df2` to use for filtering.
    - min_value1 (int/float): The minimum value for binning `metric1`.
    - max_value1 (int/float): The maximum value for binning `metric1`.
    - bin_width1 (int/float): The width of each bin for `metric1`.
    - min_value2 (int/float): The minimum value for binning `metric2`.
    - max_value2 (int/float): The maximum value for binning `metric2`.
    - bin_width2 (int/float): The width of each bin for `metric2`.
    - cumulative_operator1 (str, optional): The operator for cumulative filtering of `metric1`.
    - cumulative_operator2 (str, optional): The operator for cumulative filtering of `metric2`.
    - score_method (str, optional): The method to use for calculating the agreement score. Supported methods include 'recall' and 'precision'.

    Returns:
    - bins1 (numpy.ndarray): The bins used for `metric1`.
    - bins2 (numpy.ndarray): The bins used for `metric2`.
    - scores (numpy.ndarray): The match agreement scores for each cell in the grid.
    - len_df2s (numpy.ndarray): The number of entries in `df2` for each cell in the grid.
    """
    
    bins1 = np.arange(min_value1, max_value1, bin_width1)
    bins2 = np.arange(min_value2, max_value2, bin_width2)
    
    scores = np.full((len(bins1), len(bins2)), np.nan) 
    df2s = {}
    
    for i, b1 in enumerate(bins1):
        for j, b2 in enumerate(bins2):
            if cumulative_operator1 is not None and cumulative_operator2 is not None:
                filtered_df2 = df2.query(f'({metric1} {cumulative_operator1} @b1) and {metric2} {cumulative_operator2} @b2')
            else:
                filtered_df2 = df2.query(f'({metric1} >= @b1 and {metric1} < @b1 + @bin_width1) and ({metric2} >= @b2 and {metric2} < @b2 + @bin_width2)')
            try:
                score, _ = match_agreement_score_on_common_attrs(df1, filtered_df2, score_method=score_method)
            except ZeroDivisionError:
                score = np.nan

            scores[i, j] = score
            df2s[(i, j)] = filtered_df2
    return bins1, bins2, scores, df2s


def sparse_linear_sum_assignment_from_edge_list(df, attr1, attr2, cost_attr, maximize=False):
    """
    Perform linear sum assignment on a bipartite graph defined by an edge list.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the edge list.
    attr1 (str): The column name representing the first set of vertices.
    attr2 (str): The column name representing the second set of vertices.
    cost_attr (str): The column name representing the weights/costs of the edges.
    maximize (bool): Whether to maximize the assignment. Defaults to False (minimize).

    Returns:
    pd.DataFrame: A DataFrame containing the optimal matching with columns:
                  - attr1: The first set of vertices.
                  - attr2: The second set of vertices.
                  - cost_attr: The weights/costs of the matched edges.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()
    
    # Encode 'attr1' and 'attr2' as numeric indices
    encoder1 = LabelEncoder()
    encoder2 = LabelEncoder()

    df_copy.loc[:, 'index1'] = encoder1.fit_transform(df_copy[attr1])
    df_copy.loc[:, 'index2'] = encoder2.fit_transform(df_copy[attr2])

    # Extract the encoded indices and weights
    rows = df_copy['index1']
    cols = df_copy['index2']
    weights = df_copy[cost_attr]

    # Determine the size of the cost matrix
    size1 = len(encoder1.classes_)
    size2 = len(encoder2.classes_)

    # Set a high cost (penalty) for missing edges
    if maximize:
        high_cost = -weights.max() * 10
    else:
        high_cost = weights.max() * 10

    # Create the dense cost matrix with high cost for missing edges
    cost_matrix = np.full((size1, size2), high_cost)
    cost_matrix[rows, cols] = weights

    # Use the Hungarian algorithm (linear_sum_assignment) to find the optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=maximize)

    # Get the optimal matching
    match_df = pd.DataFrame({
        attr1: encoder1.inverse_transform(row_ind),
        attr2: encoder2.inverse_transform(col_ind),
        cost_attr: cost_matrix[row_ind, col_ind]
    })

    return match_df


def resolve_match_disagreements(df1, df2, attr1, attr2, cost_attr, maximize=False, source_names=None, field_attrs=('scan_session', 'scan_idx', 'field')):
    """
    Combine two DataFrames by matching their entries using linear sum assignment.

    Parameters:
    df1 (pd.DataFrame): The first DataFrame to be combined.
    df2 (pd.DataFrame): The second DataFrame to be combined.
    attr1 (str): The column name representing the first set of vertices for matching.
    attr2 (str): The column name representing the second set of vertices for matching.
    cost_attr (str): The column name representing the weights/costs of the edges for matching.
    maximize (bool): Whether to maximize the assignment. Defaults to False (minimize).
    source_names (list, optional): List of source names for the DataFrames. Defaults to None, which assigns 'df1' and 'df2'.
    field_attrs (tuple): The column names representing fields to group by for separate matching. Defaults to ('scan_session', 'scan_idx', 'field').

    Returns:
    pd.DataFrame: A DataFrame containing the combined results with optimal matching.
    """
    df1_copy = df1.copy()
    df1_copy['source'] = 'df1' if source_names is None else source_names[0]
    
    df2_copy = df2.copy()
    df2_copy['source'] = 'df2' if source_names is None else source_names[1]
    
    concat_df = pd.concat([df1_copy, df2_copy])
    concat_df.loc[concat_df.duplicated([attr1, attr2], keep=False), 'source'] = 'both'
    
    # agree_df = concat_df.query('source=="both"').sort_values([attr1, attr2, cost_attr], ascending=not maximize).drop_duplicates([attr1, attr2])
    agree_df = concat_df.query('source=="both"').groupby([attr1, attr2], as_index=False).mean()
    disagree_df = concat_df.query('source!="both"')
    
    field_dfs = []
    for _, field_df in disagree_df.groupby(list(field_attrs)):
        field_dfs.append(sparse_linear_sum_assignment_from_edge_list(field_df, attr1, attr2, cost_attr, maximize=maximize))
    
    rematch_df = pd.concat(field_dfs)
    remerge_df = pd.concat([
        df1_copy.merge(rematch_df, on=[attr1, attr2]),
        df2_copy.merge(rematch_df, on=[attr1, attr2])
    ]).drop(columns=f'{cost_attr}_x').rename(columns={f'{cost_attr}_y': cost_attr})
    
    return pd.concat([agree_df, remerge_df])