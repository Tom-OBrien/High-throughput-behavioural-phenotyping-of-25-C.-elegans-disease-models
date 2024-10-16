#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 15:22:27 2022

This script takes raw files of individual worm tracks extracted by Tierpsy
and makes a timeseries plot of the speed of N2 and imb-2 treated with paraquat.
**Note: This figure is saved within the file containing the data as 'speed_plot.pdf'

The raw data files are ~100gb in size, so this script also saves a condensed
version of the speed feature with timestamps for each worm track.
For ease of rerunning this .csv file is saved in the data repository and 
is called in this script to remake the figures shown in the paper.

@author: tobrien
"""

#%% Imports
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes
from tierpsytools.preprocessing.filter_data import (filter_nan_inf)
from tierpsytools.read_data.get_timeseries import read_timeseries

sys.path.insert(0, '/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Code/Helper_Functions_and_Scripts')
from helper import (read_disease_data,
                    select_strains,
                    filter_features_no_annotations,
                    make_colormaps,
                    find_window, 
                    plot_colormap,
                    plot_cmap_text)

#%% Paths to data directory, save location and data files 
PROJECT_DIR = "/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/TNPO2_sensitisation/Paraquat_timeseries"
SAVE_DIR = "/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/TNPO2_sensitisation/Paraquat_timeseries"
FEAT_FILE = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/TNPO2_sensitisation/Paraquat_timeseries/featurematrix.csv')
FNAME_FILE = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/TNPO2_sensitisation/Paraquat_timeseries/filenames.csv')
METADATA_FILE = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/TNPO2_sensitisation/Paraquat_timeseries/metadata.csv')
# Path to windowed summaries
WINDOW_FILES = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/TNPO2_sensitisation/Paraquat_timeseries/window_summaries')
## Providing a path to a folder with raw data (on our group's server)- this is
## too big to be included in the datasets, so isn't necessary to re-run code
# RAW_DATA_DIR = Path('/Volumes/behavgenom$/Tom/Data/Hydra/imb-2/paraquat_homo')

# How many wells used in the imaging plate
N_WELLS = 6
# Frames per second of analysis
FPS = 25
# Control for pairwise comparison
CONTROL = 'N2_Paraquat'
# Nan thresholding
nan_threshold_row = 0.5
nan_threshold_col = 0.05
# Features we want to plot
FEATURE_SET = ['speed_50th']
# Set information about the blue light periods
BLUELIGHT_TIMEPOINTS_SECONDS = [(60, 70),(160, 170),(260, 270)]
BLUELIGHT_FRAMES = [(1500,1751),(4000,4251),(6500,6751)]
# Creat a window dictionary of the bluelight periods
WINDOW_DICT = {0:(50,60),
               1:(65,75),
               2:(75,85),
               3:(150,160),
               4:(165,175),
               5:(175,185),
               6:(250,260),
               7:(265,275),
               8:(275,286)}
WINDOW_NAME_DICT = {0:"pre-stimulus 1", 
                    1: "blue light 1",
                    2:"post-stimulus 1", 
                    3: "pre-stimulus 2",
                    4:"blue light 2", 
                    5: "post-stimulus 2",
                    6: "pre-stimulus 3",
                    7:"blue light 3",
                    8:"post-stimulus 3"}
# Choose to remove any strains from analysis
strains_done = []
                
# %% Defining all functions
def arousal_stats(metadata,
                  features,
                  group_by='treatment',
                  control=CONTROL,
                  save_dir=None,
                  feature_set=None,
                  pvalue_threshold=0.05,
                  fdr_method='fdr_bh'):
    
    # check case-sensitivity
    assert len(metadata[group_by].unique()) == len(metadata[group_by].str.upper().unique())
    
    if feature_set is not None:
        feature_set = [feature_set] if isinstance(feature_set, str) else feature_set
        assert isinstance(feature_set, list)
        assert(all(f in features.columns for f in feature_set))
    else:
        feature_set = features.columns.tolist()
        
    features = features[feature_set].reindex(metadata.index)

    # print mean sample size
    sample_size = metadata.groupby(group_by).count()
    print("Mean sample size of %s: %d" % (group_by, int(sample_size[sample_size.columns[-1]].mean())))

    # Perform t-tests
    stats_t, pvals_t, reject_t = univariate_tests(X=features,
                                                  y=metadata[group_by],
                                                  control=control,
                                                  test='t-test',
                                                  comparison_type='binary_each_group',
                                                  multitest_correction=fdr_method,
                                                  alpha=pvalue_threshold)
    
    effect_sizes_t = get_effect_sizes(X=features,
                                      y=metadata[group_by],
                                      control=control,
                                      linked_test='t-test')
    
    stats_t.columns = ['stats_' + str(c) for c in stats_t.columns]
    pvals_t.columns = ['pvals_' + str(c) for c in pvals_t.columns]
    reject_t.columns = ['reject_' + str(c) for c in reject_t.columns]
    effect_sizes_t.columns = ['effect_size_' + str(c) for c in effect_sizes_t.columns]
    ttest_results = pd.concat([stats_t, pvals_t, reject_t, effect_sizes_t], axis=1)
    
    # save results
    ttest_path = Path(save_dir) / 't-test_results.csv'
    ttest_path.parent.mkdir(exist_ok=True, parents=True)
    ttest_results.to_csv(ttest_path, header=True, index=True)
    
    nsig = sum(reject_t.sum(axis=1) > 0)
    print("%d significant features between any %s vs %s (t-test, P<%.2f, %s)" %\
          (nsig, group_by, control, pvalue_threshold, fdr_method))

    return

# %%
if __name__ == '__main__':

    window_files = list(WINDOW_FILES.rglob('*_window_*'))
    window_feat_files = [f for f in window_files if 'features' in str(f)]
    window_feat_files.sort(key=find_window)
    window_fname_files = [f for f in window_files if 'filenames' in str(f)]
    window_fname_files.sort(key=find_window)
    
    assert (find_window(f[0]) == find_window(f[1]) for f in list(zip(
            window_feat_files, window_fname_files)))
        
    # Use Ida's helper function to read in window files and concat into DF
    feat_windows = []
    meta_windows = []
    for c,f in enumerate(list(zip(window_feat_files, window_fname_files))):
            _feat, _meta = read_disease_data(f[0],
                                              f[1],
                                              METADATA_FILE,
                                              drop_nans=True,
                                              align_blue_light=False
                                              )
            _meta['window'] = find_window(f[0])
            
            meta_windows.append(_meta)
            feat_windows.append(_feat)

    meta_windows = pd.concat(meta_windows)
    meta_windows.reset_index(drop=True,
                                 inplace=True)
        
    # Concatenate feature windows
    feat_windows = pd.concat(feat_windows)
    feat_windows.reset_index(drop=True,
                                 inplace=True)

    meta = meta_windows
    feat = feat_windows
    
    # Create an analysis column containing information about drug and genes
    meta['analysis'] = meta['worm_gene'] + '_' + meta['drug_type'] 
    meta.analysis.replace({'imb-2_No_compound': 'imb-2',
                           'N2_No_compound': 'N2'}, inplace=True)
    meta['worm_gene'] = meta['analysis']

    # Filter out nan's within specified columns and print .csv of these    
    nan_worms = meta[meta.worm_gene.isna()][['featuresN_filename',
                                             'well_name',
                                             'imaging_plate_id',
                                             'instrument_name',
                                             'date_yyyymmdd']]
    nan_worms.to_csv(METADATA_FILE.parent / 'nan_worms.csv', index=False)
    print('{} nan worms'.format(nan_worms.shape[0]))
    feat = feat.drop(index=nan_worms.index)
    meta = meta.drop(index=nan_worms.index)    

    genes = [g for g in meta.worm_gene.unique() if g != CONTROL]
    genes = list(set(genes) - set(strains_done))
            
    genes.sort()
    strain_numbers = []
    #% Filter nans with tierpsy tools function
    feat = filter_nan_inf(feat, 0.5, axis=1, verbose=True)
    meta = meta.loc[feat.index]
    feat = filter_nan_inf(feat, 0.05, axis=0, verbose=True)
    feat = feat.fillna(feat.mean())
    # Some raw data in this file is corrupted, so drop from analysis
    plate_drop = ['Para_02']
    meta = meta[meta['imaging_plate_id'].isin(plate_drop)]
    # Select wells containing our strain of interest
    wells = ['A2', 'B2']
    meta = meta[meta['well_name'].isin(wells)]

    #%% Keep only blue light videos (we're looking at window summaries for these only)
    metadata = meta[meta['imgstore_name'].str.contains('bluelight')]
    features = feat.loc[metadata.index]
    
    # Now subset the data for the features we're interested in
    features = features[FEATURE_SET].copy()
    # Build a feature list of these
    feature_list = features.columns.tolist()

    # To keep with Saul's script, I'm simply adding my strain info to treatment col
    treatment_cols = ['worm_gene']
    metadata['treatment'] = metadata[treatment_cols].astype(str).agg('-'.join, axis=1)
    
    #%% Analyse individual conditions with controls set independently 
   
    for count, g in enumerate(genes):
        print('Analysing {} {}/{}'.format(g, count+1, len(genes)))
        candidate_gene = 'imb-2_Paraquat'
        
        # set the results to save in a different location for each condition
        saveto = Path(SAVE_DIR) / candidate_gene
        saveto.mkdir(exist_ok=True)
        
        control = CONTROL
        print('Control = {}'.format(CONTROL))
    
        meta['window'] = meta['window'].astype(int)
        window_list = list(meta['window'].unique())
        
        # Generation of a hardcoded strain colour map
        strain_lut = {}
        # candidate_gene_colour = STRAIN_cmap[candidate_gene]
        strain_lut = {candidate_gene: (0.8, 0.4, 0.0),
                      CONTROL : (0.0, 0.4, 0.8)}

       # Use helper function to select the control and candidate gene of
       # interest and save as new feat/meta dataframes
        feat_df, meta_df, idx, gene_list = select_strains([candidate_gene],
                                                              CONTROL,
                                                              feat_df=features,
                                                              meta_df=metadata)
            
        # Double check filtering (in theory 0 samples/feats should drop)
        feat_df, meta_df, featsets = filter_features_no_annotations(feat_df,
                                                          meta_df)
            
       # Generation of stim and strain lut, note: I hardcode colours above
        strain_lut_, stim_lut, feat_lut = make_colormaps(gene_list,
                                                             featlist=featsets['all'],
                                                             idx=idx,
                                                             candidate_gene=[candidate_gene],
                                                             # candidate_gene=None
                                                             )
         # Save colour maps as legends/figure keys
        plot_colormap(strain_lut)
        plt.tight_layout()
        plt.savefig(saveto / 'strain_cmap.png', bbox_inches='tight')
        plot_cmap_text(strain_lut)
        plt.tight_layout()
        plt.savefig(saveto / 'strain_cmap_text.png', bbox_inches='tight')
        plt.close('all')
# %% Time series functions
        def write_list_to_file(list_to_save, save_path):
            """ Write a list to text file """
            
            Path(save_path).parent.mkdir(exist_ok=True, parents=True)
            
            with open(str(save_path), 'w') as fid:
                for line in list_to_save:
                    fid.write("%s\n" % line)
                    
            return
        
        def add_bluelight_to_plot(ax, bluelight_frames=BLUELIGHT_FRAMES, alpha=0.5):
                """ Add lines to plot to indicate video frames where bluelight stimulus was delivered 
                
                    Inputs
                    ------
                    fig, ax : figure and axes from plt.subplots()
                    bluelight_frames : list of tuples (start, end) 
                """
                assert type(bluelight_frames) == list or type(bluelight_frames) == tuple
                if not type(bluelight_frames) == list:
                    bluelight_frames = [bluelight_frames]
                
                for (start, stop) in bluelight_frames:
                    ax.axvspan(start, stop, facecolor='lightblue', alpha=alpha)
                 
                return ax
        
        def _bootstrapped_ci(x, function=np.mean, n_boot=100, which_ci=95, axis=None):
            """ Wrapper for tierpsytools bootstrapped_ci function, which encounters name space / 
                variable scope conflicts when used in combination with pandas apply function 
            """
            from tierpsytools.analysis.statistical_tests import bootstrapped_ci
            
            lower, upper = bootstrapped_ci(x, func=function, n_boot=n_boot, which_ci=which_ci, axis=axis)
            
            return lower, upper
        
        def bootstrapped_ci(x, n_boot=100, alpha=0.95):
            """ Wrapper for applying bootstrap function to sample array """
        
            from sklearn.utils import resample
            
            means = []
            for i in range(n_boot):
                s = resample(x, n_samples=int(len(x)))
                m = np.mean(s)
                means.append(m)
            # plt.hist(means); plt.show()
            
            # confidence intervals
            p_lower = ((1.0 - alpha) / 2.0) * 100
            lower = np.percentile(means, p_lower)
            p_upper = (alpha + ((1.0 - alpha) / 2.0)) * 100
            upper = np.percentile(means, p_upper)
                                
            return lower, upper
            
            
        def get_strain_timeseries(metadata, 
                                  project_dir, 
                                  strain=candidate_gene, 
                                  group_by='treatment',
                                  feature_list=['motion_mode',
                                                # 'speed'
                                                ],
                                  save_dir=None,
                                  n_wells=96,
                                  verbose=True,
                                  return_error_log=False):
            """ Load saved timeseries reults for strain, or compile from featuresN timeseries data """
        
            strain_timeseries = True
            
            if save_dir is not None:
                save_path = Path(save_dir) / '{0}_timeseries.csv'.format(strain)
                if save_path.exists():
                    if verbose:
                        print("Loading timeseries data for %s..." % strain)
                    strain_timeseries = pd.read_csv(save_path)
                    assert all(f in strain_timeseries.columns for f in feature_list)
        
            if strain_timeseries is None: 
                print("Compiling timeseries for %s..." % strain)
                strain_meta = metadata.groupby(group_by).get_group(strain)
                            
                # make dict of video imgstore names and wells we need to extract for strain data
                video_list = sorted(strain_meta['imgstore_name'].unique())
                grouped_video = strain_meta.groupby('imgstore_name')
                video_dict = {vid : sorted(grouped_video.get_group(vid)['well_name'].unique()) 
                              for vid in video_list}     
                  
                feature_list = [feature_list] if isinstance(feature_list, str) else feature_list
                assert isinstance(feature_list, list)
                colnames = ['worm_index','timestamp','well_name']
                colnames.extend(feature_list)
                
                error_log = []
                strain_timeseries_list = []
                for imgstore, wells in tqdm(video_dict.items()):
                    
                    filename = RAW_DATA_DIR / 'Results' / imgstore / 'metadata_featuresN.hdf5'
        
                    try:
                        df = read_timeseries(filename, 
                                             names=colnames,
                                             only_wells=wells if n_wells != 6 else None)
                        
                        df['filename'] = filename
                        if len(wells) == 1:
                            df['well_name'] = wells[0]
                                            
                        strain_timeseries_list.append(df)
                        
                    except Exception as E:
                        if verbose:
                            print("ERROR reading file! %s" % filename)
                            print(E)
                        error_log.append(filename)
                        
                # compile timeseries data for strain 
                strain_timeseries = pd.concat(strain_timeseries_list, axis=0, ignore_index=True)
                
                # save timeseries dataframe to file
                if save_dir is not None:
                    if verbose:
                        print("Saving timeseries data for %s..." % strain)
                    save_dir.mkdir(exist_ok=True, parents=True)
                    strain_timeseries.to_csv(save_path, index=False)
                         
                    if len(error_log) > 0:
                        write_list_to_file(error_log, Path(save_dir) / 'error_log.txt')
                        
            return strain_timeseries
        
        def plot_timeseries(df, 
                            feature='speed', 
                            error=True, 
                            max_n_frames=None, 
                            smoothing=1, 
                            ax=None,
                            bluelight_frames=None, 
                            title=None, 
                            saveAs=None, 
                            colour=None):
            """ Plot timeseries for any feature in HDF5 timeseries data EXCEPT from motion mode or turn 
                features. For motion mode, please use 'plot_timeseries_motion_mode' """
            
            # from time_series.plot_timeseries import add_bluelight_to_plot #, _bootstrapped_ci
            
            grouped_timestamp = df.groupby(['timestamp'])[feature]
        
            plot_df = grouped_timestamp.mean().reset_index()
        
            # mean and bootstrap CI error for each timestamp
            if error:            
                                        
                conf_ints = grouped_timestamp.apply(bootstrapped_ci, n_boot=100)
                conf_ints = pd.concat([pd.Series([x[0] for x in conf_ints], index=conf_ints.index), 
                                       pd.Series([x[1] for x in conf_ints], index=conf_ints.index)], 
                                      axis=1)
                conf_ints = conf_ints.rename(columns={0:'lower',1:'upper'}).reset_index()
                                    
                plot_df = pd.merge(plot_df, conf_ints, on='timestamp')
                #plot_df = plot_df.dropna(axis=0, how='any')
        
            plot_df = plot_df.set_index('timestamp').rolling(window=smoothing, 
                                                             center=True).mean().reset_index()
                
            # crop timeseries data to standard video length (optional)
            if max_n_frames:
                plot_df = plot_df[plot_df['timestamp'] <= max_n_frames]
        
            if ax is None:
                fig, ax = plt.subplots(figsize=(15,6))
        
            sns.lineplot(data=plot_df,
                         x='timestamp',
                         y=feature,
                         ax=ax,
                         ls='-',
                         hue=None,
                         palette=None,
                         color=colour)
            if error:
                ax.fill_between(plot_df['timestamp'], plot_df['lower'], plot_df['upper'], 
                                color=colour, edgecolor=None, alpha=0.25)
            
            # add decorations
            if bluelight_frames is not None:
                ax = add_bluelight_to_plot(ax, bluelight_frames=bluelight_frames, alpha=0.5)
        
            if title:
                plt.title(title, pad=10)
        
            if saveAs is not None:
                Path(saveAs).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(saveAs)
            
            if ax is None:
                return fig, ax
            else:
                return ax
        my_pal = {worm_gene: "lightcoral" if worm_gene == "imb-2" else "lightskyblue" for worm_gene in meta.worm_gene.unique()}

        def plot_timeseries_feature(metadata,
                                    project_dir,
                                    save_dir,
                                    # feature='speed',
                                    feature='speed',
                                    group_by='treatment',
                                    control=control,
                                    groups_list=None,
                                    n_wells=96,
                                    bluelight_stim_type='bluelight',
                                    video_length_seconds=360,
                                    bluelight_timepoints_seconds=[(65, 75),(165, 175),(265, 275)],
                                    smoothing=10,
                                    fps=25,
                                    ylim_minmax=None,
                                    pallette=my_pal):
                
            if groups_list is not None:
                assert isinstance(groups_list, list) 
                assert all(g in metadata[group_by].unique() for g in groups_list)
            else:
                groups_list = sorted(metadata[group_by].unique())
            groups_list = [g for g in groups_list if g != control]
            assert control in metadata[group_by].unique()
            
            if bluelight_stim_type is not None and 'window' not in metadata.columns:
                metadata['imgstore_name'] = metadata['imgstore_name_{}'.format(bluelight_stim_type)]
                
            if 'window' in metadata.columns:
                assert bluelight_stim_type is not None
                stimtype_videos = [i for i in metadata['imgstore_name'] if bluelight_stim_type in i]
                metadata = metadata[metadata['imgstore_name'].isin(stimtype_videos)]
            
            if bluelight_timepoints_seconds is not None:
                bluelight_frames = [(i*fps, j*fps) for (i, j) in bluelight_timepoints_seconds]
            
            # get control timeseries
            control_ts = get_strain_timeseries(metadata,
                                               project_dir=project_dir,
                                               strain=CONTROL,
                                               group_by=group_by,
                                               feature_list=[feature],#['motion_mode','speed']
                                               save_dir=save_dir,
                                               n_wells=n_wells,
                                               verbose=True)
        
            for group in tqdm(groups_list):
                ts_plot_dir = saveto
                # ts_plot_dir = save_dir / 'Plots' / '{0}'.format(group)
                ts_plot_dir.mkdir(exist_ok=True, parents=True)
                save_path = ts_plot_dir / '{0}_{1}.pdf'.format(feature, bluelight_stim_type)
                
                if not save_path.exists():
                    group_ts = get_strain_timeseries(metadata,
                                                     project_dir=project_dir,
                                                     strain=group,
                                                     group_by=group_by,
                                                     feature_list=[feature],
                                                     save_dir=save_dir,
                                                     n_wells=n_wells,
                                                     verbose=True)
                    
                    print("Plotting '%s' timeseries for %s vs %s" % (feature, group, control))
                    col_dict = dict(zip([control, group], sns.color_palette('tab10', 2)))
                    
                    plt.close('all')
                    fig, ax = plt.subplots(figsize=(15,6), dpi=300)
                    ax = plot_timeseries(df=control_ts,
                                         feature=feature,
                                         error=True, 
                                         max_n_frames=video_length_seconds*fps, 
                                         smoothing=smoothing*fps, 
                                         ax=ax,
                                         bluelight_frames=(bluelight_frames if 
                                                           bluelight_stim_type == 'bluelight' else None),
                                         colour=col_dict[control])
                    
                    ax = plot_timeseries(df=group_ts,
                                         feature=feature,
                                         error=True, 
                                         max_n_frames=video_length_seconds*fps, 
                                         smoothing=smoothing*fps, 
                                         ax=ax,
                                         bluelight_frames=(bluelight_frames if 
                                                           bluelight_stim_type == 'bluelight' else None),
                                         colour=col_dict[group])
                    
                    if ylim_minmax is not None:
                        assert isinstance(ylim_minmax, tuple)
                        plt.ylim(ylim_minmax[0], ylim_minmax[1])
                            
                    xticks = np.linspace(0, video_length_seconds*fps, int(video_length_seconds/60)+1)
                    ax.set_xticks(xticks)
                    ax.set_xticklabels([str(int(x/fps/60)) for x in xticks])   
                    ax.set_xlabel('Time (minutes)', fontsize=12, labelpad=10)
                    ylab = feature + " (um/sec)" if feature == 'speed' else feature
                    ax.set_ylabel(ylab, fontsize=12, labelpad=10)
                    ax.legend([control, group], fontsize=12, frameon=False, loc='best', handletextpad=1)
                    plt.subplots_adjust(left=0.1, top=0.95, bottom=0.1, right=0.95)
            
                    # save plot
                    print("Saving to: %s" % save_path)
                    plt.tight_layout()
                    plt.savefig(save_path)

#%%  Now calculate and then plot timeseries of data
        get_strain_timeseries(meta_df, 
                                  project_dir=Path(PROJECT_DIR), 
                                  strain=candidate_gene, 
                                  group_by='treatment',
                                  feature_list=['speed'],
                                  save_dir=saveto,
                                  n_wells=6,
                                  verbose=True,
                                  return_error_log=False)

        plot_timeseries_feature(meta_df,
                                    project_dir=Path(PROJECT_DIR),
                                    save_dir=Path(saveto) / 'timeseries-speed',
                                    group_by='treatment',
                                    control=control,
                                    groups_list=None,
                                    feature='speed',
                                    n_wells=6,
                                    bluelight_stim_type='bluelight',
                                    video_length_seconds=360,
                                    bluelight_timepoints_seconds=BLUELIGHT_TIMEPOINTS_SECONDS,
                                    smoothing=10,
                            fps=FPS,
                            ylim_minmax=(-20,330))  