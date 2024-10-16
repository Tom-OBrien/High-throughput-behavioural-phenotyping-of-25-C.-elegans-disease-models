#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 09 10:01:15 2023
@author: tobrien

This script imports all the datasets generated for the paper. It then makes
the full phenotyping figures and calculates pairwise stats for all features
(using permutation t-tests)and makes blue
light imaging/timeseries plots. 

To make feature plots: stats must either be calculated again or their location
set as the save location for figures. This save location can then be used
to make the number of significant feature plots shown in Figure 1.

As data was gathered in seprate experiments, these are collated into files
that are called and filtered as needed below.

**NOTE: This uses random label shuffling to calculate p-values for each feature
re-running the script will therefore result in slight variations of stats from
those saved in the strain stats files folder within the data repository.

"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from itertools import chain
from tierpsytools.analysis.significant_features import k_significant_feat
from tierpsytools.preprocessing.filter_data import filter_nan_inf
from tierpsytools.preprocessing.preprocess_features import impute_nan_inf
from tierpsytools.analysis.statistical_tests import (univariate_tests,
                                                     _multitest_correct)
sys.path.insert(0, '/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Code/Helper_Functions_and_Scripts')
from helper import (read_disease_data,
                    select_strains,
                    filter_features,
                    make_colormaps,
                    find_window,
                    BLUELIGHT_WINDOW_DICT,
                    STIMULI_ORDER, 
                    plot_colormap,
                    plot_cmap_text,
                    make_clustermaps,
                    clustered_barcodes,
                    feature_box_plots,
                    window_errorbar_plots,
                    CUSTOM_STYLE,
                    plot_frac_by_mode, 
                    MODECOLNAMES)
from Strain_cmap import strains_cmap as STRAIN_cmap

#%% The data was collected in two sets, therefore we load them both
#  Data 1
Data1_FEAT_FILE =  Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/DataSet_1/DataSet_1_featurematrix.csv') 
Data1_METADATA_FILE = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/DataSet_1/DataSet_1_metadata.csv')
# Data 2
Data2_FEAT_FILE = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/DataSet_2/DataSet_2_featurematrix.csv')
Data2_METADATA_FILE = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/DataSet_2/DataSet_2_metadata.csv')
# Data 3
Data3_FEAT_FILE = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/DataSet_3/DataSet_3_featurematrix.csv')
Data3_METADATA_FILE = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/DataSet_3/DataSet_3_metadata.csv')
# Data 4
Data4_FEAT_FILE = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/DataSet_4/DataSet_4_featurematrix.csv')
Data4_METADATA_FILE = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/DataSet_4/DataSet_4_metadata.csv')
# Data 5
Data5_FEAT_FILE = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/DataSet_5/DataSet_5_featurematrix.csv')
Data5_METADATA_FILE = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/DataSet_5/DataSet_5_metadata.csv')
# Data 6
Data6_FEAT_FILE = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/DataSet_6/DataSet_6_featurematrix.csv')
Data6_METADATA_FILE = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/DataSet_6/DataSet_6_metadata.csv')
# Data 7
Data7_FEAT_FILE = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/DataSet_7/DataSet_7_featurematrix.csv')
Data7_METADATA_FILE = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/DataSet_7/DataSet_7_metadata.csv')
# Data 8
Data8_FEAT_FILE = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/DataSet_8/DataSet_8_featurematrix.csv')
Data8_METADATA_FILE = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/DataSet_8/DataSet_8_metadata.csv')
# Path to timeseries files extracted by Tierpsy
STRAIN_TIME_SERIES = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/StrainTimeseries')
# Set the control strain
CONTROL_STRAIN = 'N2'  
# Choose what type of analysis to do

# All stim calcualtes stats and makes boxplots
# Blue light makes windowed feature plots
# Timeseries plots fraction of worms moving from raw video data
ANALYSIS_TYPE = ['all_stim','bluelight','timeseries'] #options:['all_stim','timeseries','bluelight']
# N2 analysis looks at N2 only data across all the screening days
N2_analysis=True
# Choose if to plot lineplots of motion modes
motion_modes=True
# Choose if to recalculate stats (several methods are avaliable using the in
# built Tierpsy tools modele, for paper we use permutation t-tests)
do_stats=False
# Choose to remove any strains from reanalysis (put name in list)
strains_done = []
#%%Setting plotting styles, filtering data & renaming strains
if __name__ == '__main__':
    # Set stats test to be performed and set save directory for output
    which_stat_test = 'permutation_ttest' 
    if which_stat_test == 'permutation_ttest':
        # Path to pre-calculated stats, and where data is saved
        saveto = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/Strain_stats_and_features_to_plot')
        figures_dir = saveto
    # Setting custom plotting style
    plt.style.use(CUSTOM_STYLE)
    sns.set_style('ticks')
    # %%Read in and filter first dataset
    Data1_featMat = pd.read_csv(Data1_FEAT_FILE, index_col=False)
    Data1_metadata = pd.read_csv(Data1_METADATA_FILE, index_col=False)
    Data1_featMat = filter_nan_inf(Data1_featMat, 0.5, axis=1, verbose=True)
    Data1_metadata = Data1_metadata.loc[Data1_featMat.index]
    Data1_featMat = filter_nan_inf(Data1_featMat, 0.05, axis=0, verbose=True)
    Data1_featMat = Data1_featMat.fillna(Data1_featMat.mean())
    Data1_metadata = Data1_metadata.loc[Data1_featMat.index]
    # filter features
    Data1_feat_df, Data1_meta_df, Data1_featsets = filter_features(Data1_featMat,
                                                                   Data1_metadata)
    # %% Do the same for the second set of data
    Data2_featMat = pd.read_csv(Data2_FEAT_FILE, index_col=False)
    Data2_metadata = pd.read_csv(Data2_METADATA_FILE, index_col=False)
    Data2_featMat = filter_nan_inf(Data2_featMat, 0.5, axis=1, verbose=True)
    Data2_metadata = Data2_metadata.loc[Data2_featMat.index]
    Data2_featMat = filter_nan_inf(Data2_featMat, 0.05, axis=0, verbose=True)
    Data2_featMat = Data2_featMat.fillna(Data2_featMat.mean())
    Data2_metadata = Data2_metadata.loc[Data2_featMat.index]
    # filter features
    Data2_feat_df, Data2_meta_df, Data2_featsets = filter_features(
                                                Data2_featMat, Data2_metadata)
    # %% Do the same for the third set of data
    Data3_featMat = pd.read_csv(Data3_FEAT_FILE, index_col=False)
    Data3_metadata = pd.read_csv(Data3_METADATA_FILE, index_col=False)
    Data3_featMat = filter_nan_inf(Data3_featMat, 0.5, axis=1, verbose=True)
    Data3_metadata = Data3_metadata.loc[Data3_featMat.index]
    Data3_featMat = filter_nan_inf(Data3_featMat, 0.05, axis=0, verbose=True)
    Data3_featMat = Data3_featMat.fillna(Data3_featMat.mean())
    Data3_metadata = Data3_metadata.loc[Data3_featMat.index]
    # filter features
    Data3_feat_df, Data3_meta_df, Data3_featsets = filter_features(
                                                Data3_featMat, Data3_metadata)
    # %% Do the same for the fourth set of data
    Data4_featMat = pd.read_csv(Data4_FEAT_FILE, index_col=False)
    Data4_metadata = pd.read_csv(Data4_METADATA_FILE, index_col=False)
    Data4_featMat = filter_nan_inf(Data4_featMat, 0.5, axis=1, verbose=True)
    Data4_metadata = Data4_metadata.loc[Data4_featMat.index]
    Data4_featMat = filter_nan_inf(Data4_featMat, 0.05, axis=0, verbose=True)
    Data4_featMat = Data4_featMat.fillna(Data4_featMat.mean())
    Data4_metadata = Data4_metadata.loc[Data4_featMat.index]
    # filter features
    Data4_feat_df, Data4_meta_df, Data4_featsets = filter_features(
                                                Data4_featMat, Data4_metadata)
    # %% Do the same for the fith set of data
    Data5_featMat = pd.read_csv(Data5_FEAT_FILE, index_col=False)
    Data5_metadata = pd.read_csv(Data5_METADATA_FILE, index_col=False)
    Data5_featMat = filter_nan_inf(Data5_featMat, 0.5, axis=1, verbose=True)
    Data5_metadata = Data5_metadata.loc[Data5_featMat.index]
    Data5_featMat = filter_nan_inf(Data5_featMat, 0.05, axis=0, verbose=True)
    Data5_featMat = Data5_featMat.fillna(Data5_featMat.mean())
    Data5_metadata = Data5_metadata.loc[Data5_featMat.index]
    # filter features
    Data5_feat_df, Data5_meta_df, Data5_featsets = filter_features(
                                                Data5_featMat, Data5_metadata)
    # %% Do the same for the sixth set of data
    Data6_featMat = pd.read_csv(Data6_FEAT_FILE, index_col=False)
    Data6_metadata = pd.read_csv(Data6_METADATA_FILE, index_col=False)
    Data6_featMat = filter_nan_inf(Data6_featMat, 0.5, axis=1, verbose=True)
    Data6_metadata = Data6_metadata.loc[Data6_featMat.index]
    Data6_featMat = filter_nan_inf(Data6_featMat, 0.05, axis=0, verbose=True)
    Data6_featMat = Data6_featMat.fillna(Data6_featMat.mean())
    Data6_metadata = Data6_metadata.loc[Data6_featMat.index]
    # filter features
    Data6_feat_df, Data6_meta_df, Data6_featsets = filter_features(
                                                Data6_featMat, Data6_metadata)
    # %% Do the same for the seventh set of data
    Data7_featMat = pd.read_csv(Data7_FEAT_FILE, index_col=False)
    Data7_metadata = pd.read_csv(Data7_METADATA_FILE, index_col=False)
    Data7_featMat = filter_nan_inf(Data7_featMat, 0.5, axis=1, verbose=True)
    Data7_metadata = Data7_metadata.loc[Data7_featMat.index]
    Data7_featMat = filter_nan_inf(Data7_featMat, 0.05, axis=0, verbose=True)
    Data7_featMat = Data7_featMat.fillna(Data7_featMat.mean())
    Data7_metadata = Data7_metadata.loc[Data7_featMat.index]
    # filter features
    Data7_feat_df, Data7_meta_df, Data7_featsets = filter_features(
                                                Data7_featMat, Data7_metadata)
    # %% Do the same for the eigth set of data
    Data8_featMat = pd.read_csv(Data8_FEAT_FILE, index_col=False)
    Data8_metadata = pd.read_csv(Data8_METADATA_FILE, index_col=False)
    Data8_featMat = filter_nan_inf(Data8_featMat, 0.5, axis=1, verbose=True)
    Data8_metadata = Data8_metadata.loc[Data8_featMat.index]
    Data8_featMat = filter_nan_inf(Data8_featMat, 0.05, axis=0, verbose=True)
    Data8_featMat = Data8_featMat.fillna(Data8_featMat.mean())
    Data8_metadata = Data8_metadata.loc[Data8_featMat.index]
    # filter features
    Data8_feat_df, Data8_meta_df, Data8_featsets = filter_features(
                                                Data8_featMat, Data8_metadata)
    #%% Concatenate the two datasets together
    append_feat_df = [Data1_feat_df, Data2_feat_df, Data3_feat_df, Data4_feat_df,
                      Data5_feat_df, Data6_feat_df, Data7_feat_df, Data8_feat_df,]
    append_meta_df = [Data1_meta_df, Data2_meta_df, Data3_meta_df, Data4_meta_df,
                      Data5_meta_df, Data6_meta_df, Data7_meta_df, Data8_meta_df]
    
    feat = pd.concat(append_feat_df,
                     axis=0,
                     ignore_index=True)

    meta = pd.concat(append_meta_df,
                     axis=0,
                     ignore_index=True)
    
    feat = pd.DataFrame(feat)
    meta = pd.DataFrame(meta)
    # Saving a copy of the dataframes to work from
    feat_df = feat 
    meta_df = meta 
    featsets = Data2_featsets
    #%% Remove wells annotated as bad using well annotator GUI
    n_samples = meta.shape[0]
    bad_well_cols = [col for col in meta.columns if 'is_bad' in col]
    bad = meta[bad_well_cols].any(axis=1)
    meta = meta.loc[~bad,:]
    #%% Find all the unique genes within the metadata
    genes = [g for g in meta.worm_gene.unique() if g != CONTROL_STRAIN]
    # Remove already analysed genes from the list
    genes = list(set(genes) - set(strains_done))
    genes.sort()
    strain_numbers = []
    # Duplcate date and imaging column to use with helper functions
    imaging_date_yyyymmdd = meta['date_yyyymmdd']
    imaging_date_yyyymmdd = pd.DataFrame(imaging_date_yyyymmdd)
    meta['imaging_date_yyyymmdd'] = imaging_date_yyyymmdd
    
    #%% N2 analysis only- makes cluster maps of N2 features
    # Function to select N2 only from the dataset
    if N2_analysis:
        feat_df, meta_df, idx, gene_list = select_strains(['N2'],
                                                          CONTROL_STRAIN,
                                                          feat_df=feat,
                                                          meta_df=meta)
        feat_df.drop_duplicates(inplace=True)
        meta_df.drop_duplicates(inplace=True)

        # Removes nan's, bad wells, bad days and selected tierpsy features
        feat_df, meta_df, featsets = filter_features(feat_df,
                                                     meta_df)
        # Make a stimuli colour map/ look up table with sns
        stim_cmap = sns.color_palette('Pastel1',3)
        stim_lut = dict(zip(STIMULI_ORDER.keys(), stim_cmap))
        feat_lut = {f:v for f in featsets['all'] for k,v in stim_lut.items() if k in f}
        # Impute nans from feature dataframe
        feat_nonan = impute_nan_inf(feat_df)
        # Calculate Z score of features
        featZ = pd.DataFrame(data=stats.zscore(feat_nonan[featsets['all']], 
                                               axis=0),
                             columns=featsets['all'],
                             index=feat_nonan.index)
        # Assert no nans within the data
        assert featZ.isna().sum().sum() == 0
        # Make a clustermap of the N2 only data
        N2clustered_features = make_clustermaps(featZ,
                                                meta_df,
                                                featsets,
                                                strain_lut={'N2': 
                                                            (0.6, 0.6, 0.6)},
                                                feat_lut=feat_lut,
                                                saveto=figures_dir)
        # Write order of clustered features into .txt file - this is used to
        # order future clustermaps within the paper
        for k, v in N2clustered_features.items():
            with open(figures_dir / 'N2_clustered_features_{}.txt'.format(k), 'w+') as fid:
                for line in v:
                    fid.write(line + '\n')
        # If not plotting heatmaps, read cluster features file and make dict
        # for plotting strain heatmaps etc later on in script
    else:
        N2clustered_features = {}
        for fset in STIMULI_ORDER.keys():
            N2clustered_features[fset] = []
            with open(figures_dir / 
                     'N2_clustered_features_{}.txt'.format(fset), 'r') as fid:
                N2clustered_features[fset] = [l.rstrip() 
                                              for l in fid.readlines()]
        with open(figures_dir / 'N2_clustered_features_{}.txt'.format('all'), 'r') as fid:
            N2clustered_features['all'] = [l.rstrip() for l in fid.readlines()]

    #%% Counting timer of individual gene selected for analysis
    for count, g in enumerate(genes):
        print('Analysing {} {}/{}'.format(g, count+1, len(genes)))
        candidate_gene = g
        saveto = figures_dir / candidate_gene
        saveto.mkdir(exist_ok=True)
        
        # Make a colour map for control and target strain- Here I call a
        # hardcoded strain cmap to keep all figures consistent for paper
        strain_lut = {}
        candidate_gene_colour = STRAIN_cmap[candidate_gene]
        
        if 'all_stim' in ANALYSIS_TYPE:
            print ('all stim plots for {}'.format(candidate_gene))

       # Function to select genes of interest for analysis
            feat_df, meta_df, idx, gene_list = select_strains([candidate_gene],
                                                              CONTROL_STRAIN,
                                                              feat_df=feat,
                                                              meta_df=meta)
       # Filter out unwanted features i.e dorsal feats
            feat_df, meta_df, featsets = filter_features(feat_df,
                                                         meta_df)

            strain_lut, stim_lut, feat_lut = make_colormaps(gene_list,
                                                            featlist=featsets['all'],
                                                            idx=idx,
                                                            candidate_gene=[candidate_gene],
                                                            )
        # Save colour maps as legends/figure keys for use in paper
            plot_colormap(strain_lut)
            plt.savefig(saveto / 'strain_cmap.png', bbox_inches='tight')
            plot_cmap_text(strain_lut)
            plt.savefig(saveto / 'strain_cmap_text.png', bbox_inches='tight')

            plot_colormap(stim_lut, orientation='horizontal')
            plt.savefig(saveto / 'stim_cmap.png', bbox_inches='tight')
            plot_cmap_text(stim_lut)
            plt.savefig(saveto / 'stim_cmap_text.png', bbox_inches='tight')

            plt.close('all')

            #%% Impute nan's and calculate Z scores of features for strains
            feat_nonan = impute_nan_inf(feat_df)

            featZ = pd.DataFrame(data=stats.zscore(feat_nonan[featsets['all']], 
                                                   axis=0),
                                 columns=featsets['all'],
                                 index=feat_nonan.index)

            assert featZ.isna().sum().sum() == 0
            #%% Make a nice clustermap of features for strain & N2
            # Plotting helper saves separate cluster maps for: prestim, postim, 
            # bluelight and all conditions
            (saveto / 'clustermaps').mkdir(exist_ok=True)

            clustered_features = make_clustermaps(featZ=featZ,
                                                  meta=meta_df,
                                                  featsets=featsets,
                                                  strain_lut=strain_lut,
                                                  feat_lut=feat_lut,
                                                  saveto=saveto / 'clustermaps')
            plt.close('all')
            
            # Make a copy of the cluster map for plotting pVals and selected
            # features later on in this script without overwriting plot
            N2clustered_features_copy = N2clustered_features.copy()
            # Make sure all save locations exist
            (saveto / 'heatmaps').mkdir(exist_ok=True)
            (saveto / 'heatmaps_N2ordered').mkdir(exist_ok=True)
            (saveto / 'boxplots').mkdir(exist_ok=True)
            # Calculate stats using permutation t-tests or LMM with Tierpsy
            # univariate stats modules
            if do_stats:
                    if which_stat_test == 'permutation_ttest':
                        _, unc_pvals, unc_reject = univariate_tests(
                            feat_nonan, y=meta_df['worm_gene'],
                            control='N2',
                            test='t-test',
                            comparison_type='binary_each_group',
                            multitest_correction=None,
                            n_permutation_test=100000,
                            perm_blocks=meta_df['imaging_date_yyyymmdd'],
                            )
                        reject, pvals = _multitest_correct(
                            unc_pvals, 'fdr_by', 0.05)
                        unc_pvals = unc_pvals.T
                        pvals = pvals.T
                        reject = reject.T
                    else:
                        raise ValueError((
                            f'Invalid value "{which_stat_test}"'
                            ' for which_stat_test'))
                    # massaging data to be in keeping with downstream analysis
                    assert pvals.shape[0] == 1, 'the output is supposed to be one line only I thought'
                    assert all(reject.columns == pvals.columns)
                    assert reject.shape == pvals.shape
                    # set the pvals over threshold to NaN - These are set to nan for convinence later on
                    bhP_values = pvals.copy(deep=True)
                    bhP_values.loc[:, ~reject.iloc[0, :]] = np.nan
                    bhP_values['worm_gene'] = candidate_gene
                    bhP_values.index = ['p<0.05']
                    # check the right amount of features was set to nan
                    assert reject.sum().sum() == bhP_values.notna().sum().sum()-1
                    # also save the corrected and uncorrected pvalues, without
                    # setting the rejected ones to nan, just keeping the same
                    # dataframe format as bhP_values
                    for p_df in [unc_pvals, pvals]:
                        p_df['worm_gene'] = candidate_gene
                        p_df.index = ['p value']
                    unc_pvals.to_csv(
                        saveto/f'{candidate_gene}_uncorrected_pvals.csv',
                        index=False)
                    pvals.to_csv(
                        saveto/f'{candidate_gene}_fdrby_pvals.csv',
                        index=False)
                    # Save total number of significant feats as .txt file
                    with open(saveto / 'sig_feats.txt', 'w+') as fid:
                        fid.write(str(bhP_values.notna().sum().sum()-1) + ' significant features out of \n')
                        fid.write(str(bhP_values.shape[1]-1))

                    bhP_values.to_csv(saveto / '{}_stats.csv'.format(candidate_gene),
                                      index=False)
                # If not recalculating stats, read the .csv file for plotting
            else:
                    bhP_values = pd.read_csv(saveto / '{}_stats.csv'.format(candidate_gene),
                                             index_col=False)
                    bhP_values.rename(mapper={0:'p<0.05'},
                                      inplace=True)
        #%%#I mport features to be plotted from a .txt file and make boxplots
            # Find .txt file (within save directory) and generate list of all feats to plot
            feat_to_plot_fname = list(saveto.rglob('feats_to_plot.txt'))[0]
            selected_feats = []
            with open(feat_to_plot_fname, 'r') as fid:
                    for l in fid.readlines():
                        selected_feats.append(l.rstrip().strip(','))

            all_stim_selected_feats=[]
            for s in selected_feats:
                    all_stim_selected_feats.extend([f for f in featsets['all'] if '_'.join(s.split('_')[:-1])=='_'.join(f.split('_')[:-1])])

            # Make a cluster map of strain vs N2
            clustered_barcodes(clustered_features, selected_feats,
                                    featZ,
                                    meta_df,
                                    bhP_values,
                                    saveto / 'heatmaps')

            # Use the copy of the N2 cluster map (made earlier) and plot
            # cluster map with pVals of all features alongside an asterix
            # denoting the selected features used to make boxplots
            clustered_barcodes(N2clustered_features_copy, selected_feats,
                                    featZ,
                                    meta_df,
                                    bhP_values,
                                    saveto / 'heatmaps_N2ordered')

            # Generate boxplots of selected features containing correct
            # pValues and formatted nicely
            for f in  all_stim_selected_feats:
                    feature_box_plots(f,
                                      feat_df,
                                      meta_df,
                                      strain_lut,
                                      show_raw_data='date',
                                      bhP_values_df=bhP_values
                                      )
                    plt.legend('',frameon=False)
                    plt.tight_layout()
                    plt.savefig(saveto / 'boxplots' / '{}_boxplot.png'.format(f),
                                bbox_inches='tight',
                                dpi=200)
                    plt.close('all')
                
        #%% Using window feature summaries to look at bluelight conditions
        if 'bluelight' in ANALYSIS_TYPE:
            # Set the path for the worm genes collected in the two different
            # datasets based upon what is in the list below
            data1_genes = ['ccpp-1', 'ncap-1', 'odr-8', 'rpy-1', 'tmem-222']
            data2_genes = ['blos-1', 'sam-4', 'blos-9', 'vps-50', 'smc-3[K115E]', 'sec-31', 'blos-8']
            data3_genes = ['irk-1', 'shl-1']
            data4_genes = ['pde-5', 'R10E11.6']
            data5_genes = ['pde-1', 'cpx-1', 'pmp-4', 'pacs-1[E205K]','Y47DA.1[R298W]']
            data6_genes = ['fnip-2', 'flcn-1']
            data7_genes = ['let-526']
            data8_genes = ['imb-2[D157N]']
            
            if candidate_gene in data1_genes:
                WINDOWS_FILES = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/DataSet_1/Window_Summaries/')
                WINDOWS_METADATA = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/DataSet_1/Window_Summaries/windows_metadata.csv')

            if candidate_gene in data2_genes:
                WINDOWS_FILES = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/DataSet_2/Window_Summaries/')
                WINDOWS_METADATA = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/DataSet_2/Window_Summaries/windows_metadata.csv')

            if candidate_gene in data3_genes:
                WINDOWS_FILES = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/DataSet_3/Window_Summaries/')
                WINDOWS_METADATA = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/DataSet_3/Window_Summaries/windows_metadata.csv')
                
            if candidate_gene in data4_genes:
                WINDOWS_FILES = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/DataSet_4/Window_Summaries/')
                WINDOWS_METADATA = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/DataSet_4/Window_Summaries/windows_metadata.csv')

            if candidate_gene in data5_genes:
                WINDOWS_FILES = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/DataSet_5/Window_Summaries/')
                WINDOWS_METADATA = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/DataSet_5/Window_Summaries/windows_metadata.csv')

            if candidate_gene in data6_genes:
                WINDOWS_FILES = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/DataSet_6/Window_Summaries/')
                WINDOWS_METADATA = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/DataSet_6/Window_Summaries/windows_metadata.csv')

            if candidate_gene in data7_genes:
                WINDOWS_FILES = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/DataSet_7/Window_Summaries/')
                WINDOWS_METADATA = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/DataSet_7/Window_Summaries/windows_metadata.csv')

            if candidate_gene in data8_genes:
                WINDOWS_FILES = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/DataSet_8/Window_Summaries/')
                WINDOWS_METADATA = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/DataSet_8/Window_Summaries/windows_metadata.csv')

            window_files = list(WINDOWS_FILES.rglob('*_window_*'))
            window_feat_files = [f for f in window_files if 'features' in str(f)]
            window_feat_files.sort(key=find_window)
            window_fname_files = [f for f in window_files if 'filenames' in str(f)]
            window_fname_files.sort(key=find_window)
        
            assert (find_window(f[0]) == find_window(f[1]) for f in list(zip(
                window_feat_files, window_fname_files)))
        
        # Read in window files and concatenate into one df
            feat_windows = []
            meta_windows = []
            for c,f in enumerate(list(zip(window_feat_files, window_fname_files))):
                _feat, _meta = read_disease_data(f[0],
                                                 f[1],
                                                 WINDOWS_METADATA,
                                                 drop_nans=True)
                _meta['window'] = find_window(f[0])
                
                meta_windows.append(_meta)
                feat_windows.append(_feat)
    
            meta_windows = pd.concat(meta_windows)
            meta_windows.reset_index(drop=True,
                                     inplace=True)
            
            feat_windows = pd.concat(feat_windows)
            feat_windows.reset_index(drop=True,
                                 inplace=True)
            # Update worm gene names for the paper
            print ('all window_plots for {}'.format(candidate_gene))
            meta_windows.worm_gene.replace({'odr-8_a':'odr-8',
                                            'H20J04.6':'tmem-222',
                                            'smc-3':'smc-3[K115E]',
                                            'pacs-1':'pacs-1[E205K]',
                                            'Y47DA.1':'Y47DA.1[R298W]',
                                            'imb-2':'imb-2[D157N]'},  
                                              inplace=True)    
            
            # Call dataframes window specific dataframes (made earlier)
            feat_windows_df, meta_windows_df, idx, gene_list = select_strains(
                                                          [candidate_gene],
                                                          CONTROL_STRAIN,
                                                          meta_windows,
                                                          feat_windows)

            # Filter out only the bluelight features
            bluelight_feats = [f for f in feat_windows_df.columns if 'bluelight' in f]
            feat_windows_df = feat_windows_df.loc[:,bluelight_feats]

            feat_windows_df, meta_windows_df, featsets = filter_features(feat_windows_df,
                                                                   meta_windows_df)
            bluelight_feats = list(feat_windows_df.columns)
            # Make a colour map for the blue light plotting data
            strain_lut_bluelight, stim_lut, feat_lut = make_colormaps(gene_list,
                                                            featlist=bluelight_feats,
                                                            idx=idx,
                                                            candidate_gene=[candidate_gene],
                                                            )

            #%% Impute nans with Tierpsy and z-normalise data
            feat_nonan = impute_nan_inf(feat_windows_df)
            featZ = pd.DataFrame(data=stats.zscore(feat_nonan[bluelight_feats], axis=0),
                                 columns=bluelight_feats,
                                 index=feat_nonan.index)
            assert featZ.isna().sum().sum() == 0

            #%% Find top significant feats that differentiate between prestim and bluelight
            #make save directory and set layout for plots using dictionary
            (saveto / 'windows_features').mkdir(exist_ok=True)
            meta_windows_df['light'] = [x[1] for x in meta_windows_df['window'].map(BLUELIGHT_WINDOW_DICT)]
            meta_windows_df['window_sec'] = [x[0] for x in meta_windows_df['window'].map(BLUELIGHT_WINDOW_DICT)]
            meta_windows_df['stim_number'] = [x[2] for x in meta_windows_df['window'].map(BLUELIGHT_WINDOW_DICT)]
            y_classes = ['{}, {}'.format(r.worm_gene, r.light) for i,r in meta_windows_df.iterrows()]
            # Using tierpsytools to find top 100 signifcant feats
            kfeats, scores, support = k_significant_feat(
                    feat_nonan,
                    y_classes,
                    k=100,
                    plot=False,
                    score_func='f_classif')
            # Grouping by stimulation number and line making plots for entire
            # experiment and each individual burst window
            stim_groups = meta_windows_df.groupby('stim_number').groups
            for f in kfeats[:50]:
                (saveto / 'windows_features' / f).mkdir(exist_ok=True)
                window_errorbar_plots(feature=f,
                                      feat=feat_windows_df,
                                      meta=meta_windows_df,
                                      cmap_lut=STRAIN_cmap)
                plt.savefig(saveto / 'windows_features' / f / 'allwindows_{}'.format(f), dpi=200)
                plt.close('all')
                for stim,locs in stim_groups.items():
                    window_errorbar_plots(feature=f,
                                          feat=feat_windows_df.loc[locs],
                                          meta=meta_windows_df.loc[locs],
                                          cmap_lut=STRAIN_cmap)
                    plt.savefig(saveto / 'windows_features' / f / 'window{}_{}'.format(stim,f),
                                dpi=200)
                    plt.close('all')

            #%% Calculating motion modes from bluelight features and making 
            # plots of these- saved in a sub-folder within bluelight analysis
            # if motion_modes:
            mm_feats = [f for f in bluelight_feats if 'motion_mode' in f]
            (saveto / 'windows_features' / 'motion_modes').mkdir(exist_ok=True)
            sns.set_style('ticks')
            for f in mm_feats:
                    window_errorbar_plots(feature=f,
                                          feat=feat_windows_df,
                                          meta=meta_windows_df,
                                          cmap_lut=strain_lut)
                    plt.savefig(saveto / 'windows_features' / 'motion_modes' / '{}'.format(f),
                                dpi=200)
                    plt.close('all')
                    for stim,locs in stim_groups.items():
                        window_errorbar_plots(feature=f,
                                              feat=feat_windows_df.loc[locs],
                                              meta=meta_windows_df.loc[locs],
                                              cmap_lut=strain_lut)
                        plt.savefig(saveto / 'windows_features' / 'motion_modes' / 'window{}_{}'.format(stim,f),
                                    dpi=200)
                        plt.close('all')

        #%% Make timerseries plots
        if 'timeseries' in ANALYSIS_TYPE:
             save_ts = saveto / 'timeseries'
             save_ts.mkdir(exist_ok=True)
             print ('timeseries plots for {}'.format(candidate_gene))  
             meta_ts = meta
             
             keep = [candidate_gene]
             keep.append('N2')
             mask = meta_ts['worm_gene'].isin(keep)
             meta_ts=meta_ts[mask]

             TS_STRAINS = {'plot':  [candidate_gene]}  
            # Make a list of strains with chain function (returns one iterable
            # from a list of several (not strictly necessary for this set of figs)
             ts_strain_list = list(chain(*TS_STRAINS.values()))
            
             # Find .hdf5 files of selected strains from root directory and read in
             # confidence intervals have already been calculated prior this results
             # in a list of 2 dataframes
             timeseries_df = []
             for g in ts_strain_list:
                _timeseries_fname = STRAIN_TIME_SERIES / '{}_timeseries.hdf5'.format(g)
                timeseries_df.append(pd.read_hdf(_timeseries_fname,
                                                  'frac_motion_mode_with_ci'))
                
             strain_lut = {candidate_gene:STRAIN_cmap[candidate_gene],
                          'N2':(0.6,0.6,0.6)}
          
             # Convert the list into one big dataframe and reset index
             timeseries_df = pd.concat(timeseries_df)
             timeseries_df.reset_index(drop=True, inplace=True)
             timeseries_df.worm_gene.replace({'odr-8_a':'odr-8',
                                            'H20J04.6':'tmem-222',
                                            'smc-3':'smc-3[K115E]',
                                            'pacs-1':'pacs-1[E205K]',
                                            'Y47DA.1':'Y47DA.1[R298W]',
                                            'imb-2':'imb-2[D157N]'},  
                                              inplace=True)    
            
            # Select all calculated faction modes for strains of interest and control
             frac_motion_modes = [timeseries_df.query('@ts_strain_list in worm_gene')]
             frac_motion_modes.append(timeseries_df.query('@CONTROL_STRAIN in worm_gene').groupby('timestamp').agg(np.mean))
             frac_motion_modes[1]['worm_gene'] = CONTROL_STRAIN
             frac_motion_modes = pd.concat(frac_motion_modes)
             frac_motion_modes.reset_index(drop=True,inplace=True)
            
            # Plot each of the fraction motion modes as separate plots
            # Modecolnames is just hardcoded list of 'fwd, bckwd and stationary' 
             for m in MODECOLNAMES:
                sns.set_style('ticks')
                plot_frac_by_mode(frac_motion_modes, strain_lut, modecolname=m)
                if m != 'frac_worms_st':
                    plt.ylim([0, 1.0])
                plt.savefig(save_ts / '{}_ts.png'.format(m), dpi=200)
