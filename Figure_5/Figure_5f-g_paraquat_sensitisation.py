#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 10:49:40 2022

This script calculates pairwise stats (using permutation t-tests) between N2
(wild-type) and imb-2 (mutant) worms treated with 1mM paraquat. 
It then plots the example figures shown in Fig.5F-G of the paper.

**Note: This script randomly shuffles data labels when calculating stats,
so will always result in slightly different p-values when re-running.

@author: tobrien
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from tierpsytools.preprocessing.filter_data import filter_nan_inf
from tierpsytools.preprocessing.preprocess_features import impute_nan_inf
from tierpsytools.analysis.statistical_tests import (univariate_tests,
                                                     _multitest_correct)
sys.path.insert(0, '/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Code/Helper_Functions_and_Scripts')
from helper import (select_strains,
                    filter_features_no_annotations,
                    plot_colormap,
                    plot_cmap_text,
                    feature_box_plots,
                    CUSTOM_STYLE)
#%% 'all_stim' calculates stats and plots boxplots shown in paper
ANALYSIS_TYPE = 'all_stim'
# Choose whether to recalculate stats
do_stats=False
# Choose stats test to use for this
which_stat_test = 'permutation_ttest'
# Path to data
FEAT_FILE =  Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/TNPO2_sensitisation/Paraquat_sensitisation/featurematrix.csv')
METADATA_FILE = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/TNPO2_sensitisation/Paraquat_sensitisation/metadata.csv')
# Path to save directory
save_dir= Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/TNPO2_sensitisation/Paraquat_sensitisation')
# What is the control
CONTROL_STRAIN = 'N2_Paraquat'  
# Choose what features to plot (provide these as a .txt file)
feat_plot = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/TNPO2_sensitisation/Paraquat_sensitisation/feat_to_plot')
strains_done = []
#%% Set custom plotting styles
if __name__ == '__main__':
    plt.style.use(CUSTOM_STYLE)
    sns.set_style('ticks')
    # Read in data
    feat = pd.read_csv(FEAT_FILE, index_col=False)
    meta = pd.read_csv(METADATA_FILE, index_col=False)
    # Make list of unique genes in metadata
    genes = [g for g in meta.worm_gene.unique() if g != CONTROL_STRAIN]
    # Remove any genes from reanalysis
    genes = list(set(genes) - set(strains_done))
    genes.sort()
    strain_numbers = []
    # Set date and imaging_date columns as the same- some downstream functions
    # rely on one or the other (easier if they're the same)
    imaging_date_yyyymmdd = meta['date_yyyymmdd']
    imaging_date_yyyymmdd = pd.DataFrame(imaging_date_yyyymmdd)
    meta['imaging_date_yyyymmdd'] = imaging_date_yyyymmdd                            
    
    #%% Filter nans with tierpsy tools function
    feat = filter_nan_inf(feat, 0.5, axis=1, verbose=True)
    meta = meta.loc[feat.index]
    feat = filter_nan_inf(feat, 0.05, axis=0, verbose=True)
    feat = feat.fillna(feat.mean())

#%%
    # Counting timer of individual gene selected for analysis
    for count, g in enumerate(genes):
        print('Analysing {} {}/{}'.format(g, count+1, len(genes)))
        candidate_gene = g
        # Set save path for stats and figures (make sure the exist)
        saveto = save_dir / candidate_gene
        saveto.mkdir(exist_ok=True)
        
        # Make a colour map for control and target strain- Here I use a
        # hardcoded strain cmap to keep all figures consistent for paper
        strain_lut = {}
        strain_lut = {CONTROL_STRAIN : (0.0, 0.4, 0.8),
                      candidate_gene: (0.8, 0.4, 0.0)}

        if 'all_stim' in ANALYSIS_TYPE:
            print ('all stim plots for {}'.format(candidate_gene))

       # Function to select strains to be analysed
            feat_df, meta_df, idx, gene_list = select_strains([candidate_gene],
                                                              CONTROL_STRAIN,
                                                              feat_df=feat,
                                                              meta_df=meta)
       # Double check filtering of data and features
            feat_df, meta_df, featsets = filter_features_no_annotations(feat_df,
                                                         meta_df)
        # Save colour maps as legends/figure keys for use in paper
            plot_colormap(strain_lut)
            plt.tight_layout()
            plt.savefig(saveto / 'strain_cmap.png', bbox_inches='tight')
            plot_cmap_text(strain_lut)
            plt.tight_layout()
            plt.savefig(saveto / 'strain_cmap_text.png', bbox_inches='tight')
            # #%% Impute nan's using Tierpsy tools
            feat_nonan = impute_nan_inf(feat_df)
            #%% Set save path to the data
            (saveto / 'boxplots').mkdir(exist_ok=True)
            # Calculate stats using permutation t-test function from Tierpsy
            if do_stats:
                    if which_stat_test == 'permutation_ttest':
                        _, unc_pvals, unc_reject = univariate_tests(
                            feat_nonan, y=meta_df['worm_gene'],
                            control=CONTROL_STRAIN,
                            test='t-test',
                            comparison_type='binary_each_group',
                            multitest_correction=None,
                            n_permutation_test=10000,
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
                # If not calculating stats, read the .csv file for plotting
            else:
                    bhP_values = pd.read_csv(saveto / '{}_stats.csv'.format(candidate_gene),
                                             index_col=False)
                    bhP_values.rename(mapper={0:'p<0.05'},
                                      inplace=True)
            #%% Import features to be plotted from a .txt file
            feat_to_plot_fname = list(feat_plot.rglob('feats_to_plot.txt'))[0]
            selected_feats = []
            with open(feat_to_plot_fname, 'r') as fid:
                    for l in fid.readlines():
                        selected_feats.append(l.rstrip().strip(','))
            # Append pre-stim, post-stim and blue light feats together
            all_stim_selected_feats=[]
            for s in selected_feats:
                    all_stim_selected_feats.extend([f for f in featsets['all'] if '_'.join(s.split('_')[:-1])=='_'.join(f.split('_')[:-1])])
            # Iterate over selected features and plot as boxplots
            for f in  all_stim_selected_feats:
                    feature_box_plots(f,
                                      feat_df,
                                      meta_df,
                                      strain_lut,
                                      show_raw_data='date',
                                      bhP_values_df=bhP_values)
                    plt.legend('',frameon=False)
                    plt.tight_layout()
                    plt.savefig(saveto / 'boxplots' / '{}_boxplot.png'.format(f),
                                bbox_inches='tight',
                                dpi=200)                
            plt.close('all')
