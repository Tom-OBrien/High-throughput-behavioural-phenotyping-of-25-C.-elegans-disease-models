#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 13:43:30 2023

This script plots Fig.5H. As we are conducting an aldicarb assay, we are
measuring paralysis. The best Tierpsy feature to do this is forward motion.
However, any feature from the Tierpsy set could be plotted by updating the 
'feats_to_plot' variable at the top of the scrip.

@author: tobrien
"""

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from statannot import add_stat_annotation
from tierpsytools.preprocessing.filter_data import filter_nan_inf

#%% Set paths to the data
FEAT_FILE =  Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/TNPO2_sensitisation/Aldicarb_sensitisation/featurematrix.csv') 
METADATA_FILE = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/TNPO2_sensitisation/Aldicarb_sensitisation/metadata.csv')
# Set save path to a dedicated folder
saveto = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Test')
figures_dir = saveto / 'Aldicarb_sensitisation'
figures_dir.mkdir(exist_ok=True)
# Choose what feature to plot, here we're looking for paralysis i.e fwd motion
feats_to_plot = 'motion_mode_forward_fraction_bluelight'
# Chooose whether to filter data based on well annotations
keep_good_wells_only = True
strains_done = []
#%% Read in the data
if __name__ == '__main__':
    meta = pd.read_csv(METADATA_FILE)
    feat = pd.read_csv(FEAT_FILE)
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
    # Filter data using well annotations
    if keep_good_wells_only==True:
        mask = meta['well_label'].isin([1.0])
    meta = meta[mask]    
    feat = feat[mask]
    # Update metadata column name
    meta['Aldicarb conc (uM)'] = meta['imaging_plate_drug_concentration']
    # Extract genes in metadata different from control strain
    genes = [g for g in meta.worm_gene.unique()]
    drug_concs = [d for d in meta.imaging_plate_drug_concentration.unique()]

    #%% Filter nans with Tierpsy function
    feat = filter_nan_inf(feat, 0.5, axis=1, verbose=True)
    meta = meta.loc[feat.index]
    feat = filter_nan_inf(feat, 0.05, axis=0, verbose=True)
    feat = feat.fillna(feat.mean())
            
    # %% Keep only the feature(s) to plot defined as golbal variable
    selected_feats = [feats_to_plot]
    feat = feat[feat.columns.intersection(selected_feats)]
    # Concatenate the two dataframes
    data = pd.concat([feat, meta], axis=1)
            
# %% Choose strain colours
    strain_lut =  {'N2' : (0.0, 0.4, 0.8),
                   'imb-2': (0.8, 0.4, 0.0)}
    
    # Plot the aldicarb paralysis figure
    for f in selected_feats:
                plt.figure(figsize=(20,10))

                ax = sns.boxplot(data=data,
                    x='Aldicarb conc (uM)',
                    y=f,
                    # order=strain_lut.keys(),
                    palette=strain_lut,
                    hue='worm_gene',
                    showfliers=False)
                add_stat_annotation(
                                    ax=ax, 
                                    data=data, 
                                    x='Aldicarb conc (uM)', 
                                    y=f, 
                                    hue='worm_gene',
                                    box_pairs=[
                                    ((0.0, 'N2'), (0.0, 'imb-2')),
                                     ((500.0, 'N2'), (500.0, 'imb-2')),
                                     ((1000.0, 'N2'), (1000.0, 'imb-2')),
                                     ],
                                    test='t-test_welch', 
                                    comparisons_correction='bonferroni',
                                    text_format='full', 
                                    loc='outside', 
                                    verbose=2)
                ax = sns.stripplot(data=data,
                                  x='Aldicarb conc (uM)',
                                  y=f,
                                  hue='worm_gene',
                                  dodge=True,
                                  alpha=0.6,
                                  palette='dark',
                                  )
                plt.savefig(figures_dir/ '{}'.format(f),
                            dpi=600)
                plt.show()
                plt.close('all')
        