#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 16:33:31 2023

This script iterates over the different temperature sensitisation conditions 
shown in Fig.5D. For each condition it calculates stats of N2 vs imb-2
that have been exposed to the same condition and saves these within the
given save directory. 

These are the stats used to make the bar plot shown in Fig.5D using:
    'Figure_5d_temperature_sensitisation.py'
    
**Note: we use permutation t-tests to calculate the p-values and total stats.
This randomly shuffles data labels. Hence, rerunning the script will result in
slightly different stats every time

@author: tobrien
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from tierpsytools.preprocessing.filter_data import filter_nan_inf
from tierpsytools.preprocessing.preprocess_features import impute_nan_inf
from tierpsytools.analysis.statistical_tests import (univariate_tests,
                                                     _multitest_correct)
sys.path.insert(0, '/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Code/Helper_Functions_and_Scripts')
from helper import (select_strains,
                    filter_features)

#%% Set paths to data and save directory
FEAT_FILE =  Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/TNPO2_sensitisation/Temperature_sensitisation/featurematrix.csv') 
METADATA_FILE = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/TNPO2_sensitisation/Temperature_sensitisation/metadata.csv')
savedir = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Test')
# Select  analysis type (all_stim calculates stats)
ANALYSIS_TYPE = ['all_stim'] 
# Choose if to recalculate stats
do_stats=True
# Choose type of stats, we use permutation t-tests in paper
which_stat_test = 'permutation_ttest'
# Choose the control strain
CONTROL_STRAIN = 'N2'  
# You can choose to remove strains or treatment types from analysis here
treatments_done = []
strains_done = []
#%%Setting plotting styles, filtering data & renaming strains
if __name__ == '__main__':
    
    feat = pd.read_csv(FEAT_FILE, index_col=False)
    meta = pd.read_csv(METADATA_FILE, index_col=False)

    # Extract genes in metadata different from control strain and make a list
    # of the total number of straisn
    genes = [g for g in meta.worm_gene.unique() if g != CONTROL_STRAIN]
    
    # Remove strains done from gene list, so we're only analysing the strains
    # we want to
    genes = list(set(genes) - set(strains_done))
    genes.sort()
    strain_numbers = []

    imaging_date_yyyymmdd = meta['date_yyyymmdd']
    imaging_date_yyyymmdd = pd.DataFrame(imaging_date_yyyymmdd)
    meta['imaging_date_yyyymmdd'] = imaging_date_yyyymmdd                                      
    
    #%% Filter nans with tierpsy tools function
    feat = filter_nan_inf(feat, 0.5, axis=1, verbose=True)
    meta = meta.loc[feat.index]
    feat = filter_nan_inf(feat, 0.05, axis=0, verbose=True)
    feat = feat.fillna(feat.mean())

#%% Make a copy of the metadata df to work from
    meta_master = meta.copy(deep=True)
    feat_master = feat.copy(deep=True)
    # Find unique treatment conditions and iterate over
    treatments = [t for t in meta.analysis.unique()]
    treatments = list(set(treatments) - set(treatments_done))
    treatments.sort()
    for count, t in enumerate(treatments):
        print('Treatment = {}'.format(t))
        treat = [t]
        mask = meta_master['analysis'].isin(treat)
        meta = meta_master[mask]
        feat = feat_master[mask]

        # Counting timer of individual gene selected for analysis
        for count, g in enumerate(genes):
            print('Analysing N2 vs {} {}/{}'.format(g, count+1, len(genes)))
            candidate_gene = g
            
            # Set save path for figres
            saveto = savedir / t
            saveto.mkdir(exist_ok=True)
    
            if 'all_stim' in ANALYSIS_TYPE:
                print ('Calculating stats N2 vs {}'.format(candidate_gene))
    
               # Function to select only strain of interest (pairwise comparison)
                feat_df, meta_df, idx, gene_list = select_strains([candidate_gene],
                                                                  CONTROL_STRAIN,
                                                                  feat_df=feat,
                                                                  meta_df=meta)
                # Filter out featuresets
                feat_df_1, meta_df_1, featsets = filter_features(feat_df,
                                                              meta_df)

                #%% Impute nans with Tierpsy function
                feat_nonan = impute_nan_inf(feat_df)
                
                if do_stats:
                        if which_stat_test == 'permutation_ttest':
                            _, unc_pvals, unc_reject = univariate_tests(
                                feat_nonan, y=meta_df['worm_gene'],
                                control='N2',
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
                    