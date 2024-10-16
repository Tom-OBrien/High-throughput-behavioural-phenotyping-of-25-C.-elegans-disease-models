#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 16:33:31 2023

This calculates pairwise statistics of an untreated strain, vs strain treated
with a compound for each behavioural feature extracted by Tierpsy

*These are calculated with permutation t-tests that randomly shuffle data labels
hence when reunning the script, there will be slightly different p-values due
to this randomness

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
from tierpsytools.read_data.hydra_metadata import (align_bluelight_conditions,
                                                   read_hydra_metadata)
from tierpsytools.analysis.statistical_tests import (univariate_tests,
                                                     _multitest_correct)
sys.path.insert(0, '/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Code/Helper_Functions_and_Scripts')
from helper import (select_strains,
                    CUSTOM_STYLE)
#%% Set paths to data and a root directory for this analysis.
ROOT_DIR = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/Folliculin_mutant_drug_screen/Stats/Individual_stats_for_each_treatment')
FEAT_FILE =  Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/Folliculin_mutant_drug_screen/featurematrix.csv') 
FNAME_FILE = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/Folliculin_mutant_drug_screen/filenames.csv')
METADATA_FILE = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/Folliculin_mutant_drug_screen/metadata.csv')
# Choose to filter data based upon well annotations
filter_wells = True
# Calculate stats?
do_stats=True
# First we're going to look at the parental control
CONTROL_STRAIN = 'N2'  
# Remove the mutants from being analysed with N2 
strains_done = ['flcn-1',
                'flcn-1+3BDO',
                'flcn-1+BAY-3827',
                'flcn-1+BI-9774',
                'flcn-1+INK-128',
                'flcn-1+Rapamycin',
                'flcn-1+JR-AB-011',
                'fnip-2',
                'fnip-2+3BDO',
                'fnip-2+BAY-3827',
                'fnip-2+BI-9774',
                'fnip-2+INK-128',
                'fnip-2+Rapamycin',
                'fnip-2+JR-AB-011']
#%%
if __name__ == '__main__':
    # Set stats test to be performed and  save directory for output
    which_stat_test = 'permutation_ttest'  
    if which_stat_test == 'permutation_ttest':
        save_stats = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/Folliculin_mutant_drug_screen/Stats/Individual_stats_for_each_treatment')

    # CUSTOM_STYLE= mplt style card ensuring figures are consistent for papers
    plt.style.use(CUSTOM_STYLE)
    sns.set_style('ticks')
    
    # Read in data and align by bluelight with Tierpsy  functions
    feat, meta = read_hydra_metadata(
        FEAT_FILE,
        FNAME_FILE,
        METADATA_FILE)
    feat, meta = align_bluelight_conditions(feat, meta, how='inner')

    # Converting metadata date into nicer format when plotting
    meta['date_yyyymmdd'] = pd.to_datetime(
        meta['date_yyyymmdd'], format='%Y%m%d').dt.date
    # Filter out nan's within specified columns and print .csv of these    
    nan_worms = meta[meta.worm_gene.isna()][['featuresN_filename',
                                             'well_name',
                                             'imaging_plate_id',
                                             'instrument_name',
                                             'date_yyyymmdd']]
    nan_worms.to_csv(
        METADATA_FILE.parent / 'nan_worms.csv', index=False)
    print('{} nan worms'.format(nan_worms.shape[0]))
    feat = feat.drop(index=nan_worms.index)
    meta = meta.drop(index=nan_worms.index)            
    # Filter data based upon well annotation
    if filter_wells==True:
        mask = meta['well_label'].isin([1.0, 3.0])
    meta = meta[mask]    
    feat = feat[mask]
    # Combine information about strain and drug treatment
    meta['analysis'] = meta['worm_gene'] + '+' + meta['drug_type']
    # Rename controls for ease of use
    meta['analysis'].replace({'N2+DMSO':'N2',
                              'fnip-2+DMSO':'fnip-2',
                              'flcn-1+DMSO':'flcn-1'},
                              inplace=True)
    # Update worm gene column with new info to reuse existing functions
    meta['worm_gene'] = meta['analysis']
    # Print out number of features processed
    feat_filters = [line for line in open(FNAME_FILE) 
                     if line.startswith("#")]
    print ('Features summaries were processed \n {}'.format(
           feat_filters))
    # Make summary .txt file of feats
    with open(ROOT_DIR / 'feat_filters_applied.txt', 'w+') as fid:
        fid.writelines(feat_filters)
    # Extract genes in metadata different from control strain and make a list
    # of the total number of strains
    genes = [g for g in meta.worm_gene.unique() if g != CONTROL_STRAIN]
    # Remove strains done from gene list
    genes = list(set(genes) - set(strains_done))
    genes.sort()
    strain_numbers = []
    # Fix issues with imaging date column name
    imaging_date_yyyymmdd = meta['date_yyyymmdd']
    imaging_date_yyyymmdd = pd.DataFrame(imaging_date_yyyymmdd)
    meta['imaging_date_yyyymmdd'] = imaging_date_yyyymmdd
    
    #%% Filter nans with tierpsy tools function
    feat = filter_nan_inf(feat, 0.5, axis=1, verbose=True)
    meta = meta.loc[feat.index]
    feat = filter_nan_inf(feat, 0.05, axis=0, verbose=True)
    feat = feat.fillna(feat.mean())
    
#%% Counting timer of individual gene selected for analysis
    for count, g in enumerate(genes):
        print('Analysing {} {}/{}'.format(g, count+1, len(genes)))
        candidate_gene = g
        # Set save path for stats
        saveto = save_stats / candidate_gene
        saveto.mkdir(exist_ok=True)
       # Use helper function to select individual strains for analysis
        feat_df, meta_df, idx, gene_list = select_strains([candidate_gene],
                                                               CONTROL_STRAIN,
                                                               feat_df=feat,
                                                               meta_df=meta)
        #% Impute nan's using Tierpsy function
        feat_nonan = impute_nan_inf(feat_df)
        # Calculate stats using Tierpsy Univariate functions
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

# %% Now we change the control strain to flcn-1 and do the same analysis
CONTROL_STRAIN = 'flcn-1'  
# Now we remove out the other strains from analysis
strains_done = ['N2',
                'N2+3BDO',
                'N2+BAY-3827',
                'N2+BI-9774',
                'N2+INK-128',
                'N2+Rapamycin',
                'N2+JR-AB-011',
                'fnip-2',
                'fnip-2+3BDO',
                'fnip-2+BAY-3827',
                'fnip-2+BI-9774',
                'fnip-2+INK-128',
                'fnip-2+Rapamycin',
                'fnip-2+JR-AB-011']
#%%
if __name__ == '__main__':
    # Set stats test to be performed and  save directory for output
    which_stat_test = 'permutation_ttest'  
    if which_stat_test == 'permutation_ttest':
        save_stats = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/Folliculin_mutant_drug_screen/Stats/Individual_stats_for_each_treatment')

    # CUSTOM_STYLE= mplt style card ensuring figures are consistent for papers
    plt.style.use(CUSTOM_STYLE)
    sns.set_style('ticks')
    
    # Read in data and align by bluelight with Tierpsy  functions
    feat, meta = read_hydra_metadata(
        FEAT_FILE,
        FNAME_FILE,
        METADATA_FILE)
    feat, meta = align_bluelight_conditions(feat, meta, how='inner')

    # Converting metadata date into nicer format when plotting
    meta['date_yyyymmdd'] = pd.to_datetime(
        meta['date_yyyymmdd'], format='%Y%m%d').dt.date
    # Filter out nan's within specified columns and print .csv of these    
    nan_worms = meta[meta.worm_gene.isna()][['featuresN_filename',
                                             'well_name',
                                             'imaging_plate_id',
                                             'instrument_name',
                                             'date_yyyymmdd']]

    nan_worms.to_csv(
        METADATA_FILE.parent / 'nan_worms.csv', index=False)
    print('{} nan worms'.format(nan_worms.shape[0]))
    feat = feat.drop(index=nan_worms.index)
    meta = meta.drop(index=nan_worms.index)            
    # Filter data based upon well annotation
    if filter_wells==True:
        mask = meta['well_label'].isin([1.0, 3.0])
    meta = meta[mask]    
    feat = feat[mask]
    # Combine information about strain and drug treatment
    meta['analysis'] = meta['worm_gene'] + '+' + meta['drug_type']
    # Rename controls for ease of use
    meta['analysis'].replace({'N2+DMSO':'N2',
                              'fnip-2+DMSO':'fnip-2',
                              'flcn-1+DMSO':'flcn-1'},
                              inplace=True)
    # Update worm gene column with new info to reuse existing functions
    meta['worm_gene'] = meta['analysis']
    # Print out number of features processed
    feat_filters = [line for line in open(FNAME_FILE) 
                     if line.startswith("#")]
    print ('Features summaries were processed \n {}'.format(
           feat_filters))
    # Make summary .txt file of feats
    with open(ROOT_DIR / 'feat_filters_applied.txt', 'w+') as fid:
        fid.writelines(feat_filters)
    # Extract genes in metadata different from control strain and make a list
    # of the total number of strains
    genes = [g for g in meta.worm_gene.unique() if g != CONTROL_STRAIN]
    
    # Remove strains done from gene list
    genes = list(set(genes) - set(strains_done))
    genes.sort()
    strain_numbers = []
    # Fix issues with imaging date column name
    imaging_date_yyyymmdd = meta['date_yyyymmdd']
    imaging_date_yyyymmdd = pd.DataFrame(imaging_date_yyyymmdd)
    meta['imaging_date_yyyymmdd'] = imaging_date_yyyymmdd
    
    #%% Filter nans with tierpsy tools function
    feat = filter_nan_inf(feat, 0.5, axis=1, verbose=True)
    meta = meta.loc[feat.index]
    feat = filter_nan_inf(feat, 0.05, axis=0, verbose=True)
    feat = feat.fillna(feat.mean())
    
#%% Counting timer of individual gene selected for analysis
    for count, g in enumerate(genes):
        print('Analysing {} {}/{}'.format(g, count+1, len(genes)))
        candidate_gene = g
        # Set save path for stats
        saveto = save_stats / candidate_gene
        saveto.mkdir(exist_ok=True)
       # Use helper function to select individual strains for analysis
        feat_df, meta_df, idx, gene_list = select_strains([candidate_gene],
                                                               CONTROL_STRAIN,
                                                               feat_df=feat,
                                                               meta_df=meta)
        #% Impute nan's using Tierpsy function
        feat_nonan = impute_nan_inf(feat_df)
        # Calculate stats using Tierpsy Univariate functions
        if do_stats:
                    if which_stat_test == 'permutation_ttest':
                        _, unc_pvals, unc_reject = univariate_tests(
                            feat_nonan, y=meta_df['worm_gene'],
                            control='flcn-1',
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

# %% Now we change the control strain to fnip-2 and do the same analysis
CONTROL_STRAIN = 'fnip-2'  
# Now we remove out the other strains from analysis
strains_done = ['N2',
                'N2+3BDO',
                'N2+BAY-3827',
                'N2+BI-9774',
                'N2+INK-128',
                'N2+Rapamycin',
                'N2+JR-AB-011',
                'flcn-1',
                'flcn-1+3BDO',
                'flcn-1+BAY-3827',
                'flcn-1+BI-9774',
                'flcn-1+INK-128',
                'flcn-1+Rapamycin',
                'flcn-1+JR-AB-011']
#%%
if __name__ == '__main__':
    # Set stats test to be performed and  save directory for output
    which_stat_test = 'permutation_ttest'  
    if which_stat_test == 'permutation_ttest':
        save_stats = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/Folliculin_mutant_drug_screen/Stats/Individual_stats_for_each_treatment')

    # CUSTOM_STYLE= mplt style card ensuring figures are consistent for papers
    plt.style.use(CUSTOM_STYLE)
    sns.set_style('ticks')
    
    # Read in data and align by bluelight with Tierpsy  functions
    feat, meta = read_hydra_metadata(
        FEAT_FILE,
        FNAME_FILE,
        METADATA_FILE)
    feat, meta = align_bluelight_conditions(feat, meta, how='inner')

    # Converting metadata date into nicer format when plotting
    meta['date_yyyymmdd'] = pd.to_datetime(
        meta['date_yyyymmdd'], format='%Y%m%d').dt.date
    # Filter out nan's within specified columns and print .csv of these    
    nan_worms = meta[meta.worm_gene.isna()][['featuresN_filename',
                                             'well_name',
                                             'imaging_plate_id',
                                             'instrument_name',
                                             'date_yyyymmdd']]

    nan_worms.to_csv(
        METADATA_FILE.parent / 'nan_worms.csv', index=False)
    print('{} nan worms'.format(nan_worms.shape[0]))
    feat = feat.drop(index=nan_worms.index)
    meta = meta.drop(index=nan_worms.index)            
    # Filter data based upon well annotation
    if filter_wells==True:
        mask = meta['well_label'].isin([1.0, 3.0])
    meta = meta[mask]    
    feat = feat[mask]
    # Combine information about strain and drug treatment
    meta['analysis'] = meta['worm_gene'] + '+' + meta['drug_type']
    # Rename controls for ease of use
    meta['analysis'].replace({'N2+DMSO':'N2',
                              'fnip-2+DMSO':'fnip-2',
                              'flcn-1+DMSO':'flcn-1'},
                              inplace=True)
    # Update worm gene column with new info to reuse existing functions
    meta['worm_gene'] = meta['analysis']
    # Print out number of features processed
    feat_filters = [line for line in open(FNAME_FILE) 
                     if line.startswith("#")]
    print ('Features summaries were processed \n {}'.format(
           feat_filters))
    # Make summary .txt file of feats
    with open(ROOT_DIR / 'feat_filters_applied.txt', 'w+') as fid:
        fid.writelines(feat_filters)
    # Extract genes in metadata different from control strain and make a list
    # of the total number of strains
    genes = [g for g in meta.worm_gene.unique() if g != CONTROL_STRAIN]
    
    # Remove strains done from gene list
    genes = list(set(genes) - set(strains_done))
    genes.sort()
    strain_numbers = []
    # Fix issues with imaging date column name
    imaging_date_yyyymmdd = meta['date_yyyymmdd']
    imaging_date_yyyymmdd = pd.DataFrame(imaging_date_yyyymmdd)
    meta['imaging_date_yyyymmdd'] = imaging_date_yyyymmdd
    
    #%% Filter nans with tierpsy tools function
    feat = filter_nan_inf(feat, 0.5, axis=1, verbose=True)
    meta = meta.loc[feat.index]
    feat = filter_nan_inf(feat, 0.05, axis=0, verbose=True)
    feat = feat.fillna(feat.mean())
    
#%% Counting timer of individual gene selected for analysis
    for count, g in enumerate(genes):
        print('Analysing {} {}/{}'.format(g, count+1, len(genes)))
        candidate_gene = g
        # Set save path for stats
        saveto = save_stats / candidate_gene
        saveto.mkdir(exist_ok=True)
       # Use helper function to select individual strains for analysis
        feat_df, meta_df, idx, gene_list = select_strains([candidate_gene],
                                                               CONTROL_STRAIN,
                                                               feat_df=feat,
                                                               meta_df=meta)
        #% Impute nan's using Tierpsy function
        feat_nonan = impute_nan_inf(feat_df)
        # Calculate stats using Tierpsy Univariate functions
        if do_stats:
                    if which_stat_test == 'permutation_ttest':
                        _, unc_pvals, unc_reject = univariate_tests(
                            feat_nonan, y=meta_df['worm_gene'],
                            control='fnip-2',
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