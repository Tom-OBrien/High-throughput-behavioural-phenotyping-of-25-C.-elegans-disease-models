#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 11:40:41 2024

This plots Fig5a-c from the paper. These are the key behavioural features of
the imb-2 mutant (strain specific gene card in paper) and N2 control upon
dsRNA silencing (RNAi) of the imb-1 and imb-3 genes

@author: tobrien
"""

import pandas as pd
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import chain

sys.path.insert(0, '/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Code/Helper_Functions_and_Scripts')
from helper import (select_strains,
                    filter_features,
                    make_colormaps,
                    feature_box_plots,
                    CUSTOM_STYLE)
# %% Set path to the data
FEAT_FILE = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/TNPO2_sensitisation/RNAi_sensitisation/featurematrix.csv')
METADATA_FILE = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/TNPO2_sensitisation/RNAi_sensitisation/metadata.csv')
# Choose which feature to plot (any extracted by Tierpsy)
RNAi_FEATURES = ['curvature_mean_hips_norm_abs_90th_bluelight',
                'd_angular_velocity_head_tip_abs_50th_prestim',
                'speed_tail_tip_w_paused_50th_bluelight']

# Note that I've kept 'STRAINS' as the global variable- we're actually looking
# at imb-2 reared on different food sources for this experiment though
STRAINS = {'bb': ['imb-2+imb-1 k/d',
                    'imb-2+imb-3 k/d',
                    'N2+imb-3 k/d']}
strain_list = list(chain(*STRAINS.values()))
CONTROL_STRAIN = 'N2+imb-1 k/d'

#%% Set plotting styles
if __name__ == '__main__':
    plt.style.use(CUSTOM_STYLE)
    sns.set_style('ticks')
    # Set the save path
    saveto = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Test')
    saveto.mkdir(exist_ok=True) 
    # Read in the data
    meta = pd.read_csv(METADATA_FILE)
    feat = pd.read_csv(FEAT_FILE)
    # Here I add information about RNAi treatment to worm gene info
    meta['analysis'] = meta['worm_gene']+meta['food_type']
    meta.analysis.replace({'N2imb-1_RNAi':'N2+imb-1 k/d', 
                            'N2imb-3_RNAi':'N2+imb-3 k/d',
                            'imb-2imb-1_RNAi':'imb-2+imb-1 k/d',
                            'imb-2imb-3_RNAi':'imb-2+imb-3 k/d'
                            }, inplace=True,)
    meta['worm_gene']=meta['analysis']
    # Set the date as a nicer format for plotting
    meta['date_yyyymmdd'] = pd.to_datetime(
    meta['date_yyyymmdd'], format='%Y%m%d').dt.date
    #%% Use helper function to select the strains of interest
    feat_df, meta_df, idx, gene_list = select_strains(strain_list,
                                                    CONTROL_STRAIN,
                                                    feat_df=feat,
                                                    meta_df=meta)
    # Filter features using Tierpsy
    feat_df, meta_df, featsets = filter_features(feat_df,
                                                 meta_df)
    # Make a soft coded colour map
    strain_lut, stim_lut, feat_lut = make_colormaps(gene_list,
                                                    featlist=featsets['all'],
                                                    idx=idx,
                                                    candidate_gene=strain_list
                                                    )
    # Harcoding colour map of strain/treatment conditions for paper
    strain_lut = {'N2+imb-1 k/d':'grey',
                  'imb-2+imb-1 k/d':'lightgreen',
                  'N2+imb-3 k/d':'black',
                  'imb-2+imb-3 k/d':'lightskyblue'}
    #%% Make sure the data saves in a dedicated folder
    saveto = saveto/ 'RNAi_boxplots'
    saveto.mkdir(exist_ok=True)
    # Iterate over the selected features and make boxplots of these
    for f in  RNAi_FEATURES:
        feature_box_plots(f,
                          feat_df,
                          meta_df,
                          strain_lut,
                          show_raw_data='date',
                          add_stats=True)
        plt.legend('')
        plt.savefig(saveto / '{}_boxplot.png'.format(f),
                    bbox_inches='tight',
                    dpi=200)
    plt.close('all')
