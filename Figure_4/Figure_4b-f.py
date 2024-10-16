#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 15:02:53 2021

This plots the drug screening boxplots shown in Fig.4. However, any feature
could be plotted with this.
- It calls the 'cleaned'' feature matrix/metadata files generated in the
'Stats_calculation_for_each_drug_treatment.py' script 

@author: tobrien
"""
import pandas as pd
import seaborn as sns
import sys
import matplotlib.pyplot as plt
from pathlib import Path
from tierpsytools.preprocessing.filter_data import (filter_nan_inf)

sys.path.insert(0, '/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Code/Helper_Functions_and_Scripts')
from helper import (select_strains,
                    filter_features,
                    feature_box_plots,
                    CUSTOM_STYLE)
#%% Set paths to the data
FEAT_FILE =  Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/Folliculin_mutant_drug_screen/cleaned_featurematrix.csv') 
METADATA_FILE = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/Folliculin_mutant_drug_screen/cleaned_metadata.csv')
# Set where to save data
Paper_Figure_Save_Dir = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Test')
saveto = Paper_Figure_Save_Dir 
saveto.mkdir(exist_ok=True)
# Choose control strain and to remove any previously analysed strains from data
CONTROL_STRAIN = 'N2'
strains_done =  []

#%%
if __name__ == '__main__':
    # Read in the data
    featMat = pd.read_csv(FEAT_FILE, index_col=False)
    meta = pd.read_csv(METADATA_FILE, index_col=False)
    # Filter features using Tierpsy function
    feat_df, meta_df, featsets = filter_features(featMat,
                                                 meta)
    feat_df = filter_nan_inf(feat_df, 0.5, axis=1, verbose=True)
    meta_df = meta_df.loc[feat_df.index]
    feat_df = filter_nan_inf(feat_df, 0.05, axis=0, verbose=True)
    meta_df = meta_df.loc[feat_df.index]
    feat_df = feat_df.fillna(feat_df.mean())
    meta_df = meta_df.loc[feat_df.index]
    
    #%% Set plotting style for figures
    plt.style.use(CUSTOM_STYLE)
    sns.set_style('ticks')
    # Find unique genes in metadata and remove already analysed genes
    genes = [g for g in meta.worm_gene.unique() if g != CONTROL_STRAIN]
    genes = list(set(genes) - set(strains_done))
    genes.sort()
    # Query the metadata to find genes of interest
    meta = meta.query('@genes in worm_gene or @CONTROL_STRAIN in worm_gene')
    feat = featMat.loc[meta.index,:]

    #%% First plot 3BDO figure
    # Harcode strain colours and order
    strain_lut = {'N2':'lightgrey',
                  'flcn-1':'lightgreen',
                  'fnip-2':'lightskyblue',
                  'N2+3BDO':'slategrey',
                  'flcn-1+3BDO':'darkgreen',
                  'fnip-2+3BDO':'mediumblue'}
    # Choose feature to plot and for which strains
    EXAMPLES = { 'speed_90th_bluelight': ['N2',
                                          'flcn-1',
                                          'fnip-2',
                                          'N2+3BDO',
                                          'flcn-1+3BDO',
                                          'fnip-2+3BDO']}
    # Now call the dictionary keys above to plot the boxplits and save
    for k,v in EXAMPLES.items():
        examples_feat_df, examples_meta_df, dx, gene_list = select_strains(v,
                                                                            CONTROL_STRAIN,
                                                                            feat_df=feat_df,
                                                                            meta_df=meta_df)
        feature_box_plots(k,
                          feat,
                          meta,
                          strain_lut,
                          show_raw_data='date',
                          add_stats=True,
                          )
        plt.legend('',frameon=False)
        plt.tight_layout()
        plt.savefig(saveto / '{}_3BDO_boxplot.png'.format(k), 
                    bbox_inches="tight",
                    dpi=400)
        plt.close('all')
    # %% Now plot BI-9774 figure
    # Harcode strain colours and order
    strain_lut = {'N2':'lightgrey',
                  'flcn-1':'lightgreen',
                  'fnip-2':'lightskyblue',
                  'N2+BI-9774':'slategrey',
                  'flcn-1+BI-9774':'darkgreen',
                  'fnip-2+BI-9774':'mediumblue'}
    # Choose feature to plot and for which strains
    EXAMPLES = { 'speed_90th_bluelight': ['N2',
                                          'flcn-1',
                                          'fnip-2',
                                          'N2+BI-9774',
                                          'flcn-1+BI-9774',
                                          'fnip-2+BI-9774']}
    # Now call the dictionary keys above to plot the boxplits and save
    for k,v in EXAMPLES.items():
        examples_feat_df, examples_meta_df, dx, gene_list = select_strains(v,
                                                                            CONTROL_STRAIN,
                                                                            feat_df=feat_df,
                                                                            meta_df=meta_df)
        feature_box_plots(k,
                          feat,
                          meta,
                          strain_lut,
                          show_raw_data='date',
                          add_stats=True,
                          )
        plt.legend('',frameon=False)
        plt.tight_layout()
        plt.savefig(saveto / '{}_BI-9774_boxplot.png'.format(k), 
                    bbox_inches="tight",
                    dpi=400)
        plt.close('all')
    # %% Now plot INK-128 figure
    # Harcode strain colours and order
    strain_lut = {'N2':'lightgrey',
                  'flcn-1':'lightgreen',
                  'fnip-2':'lightskyblue',
                  'N2+INK-128':'slategrey',
                  'flcn-1+INK-128':'darkgreen',
                  'fnip-2+INK-128':'mediumblue'}
    # Choose feature to plot and for which strains
    EXAMPLES = { 'speed_90th_bluelight': ['N2',
                                          'flcn-1',
                                          'fnip-2',
                                          'N2+INK-128',
                                          'flcn-1+INK-128',
                                          'fnip-2+INK-128']}
    # Now call the dictionary keys above to plot the boxplits and save
    for k,v in EXAMPLES.items():
        examples_feat_df, examples_meta_df, dx, gene_list = select_strains(v,
                                                                            CONTROL_STRAIN,
                                                                            feat_df=feat_df,
                                                                            meta_df=meta_df)
        feature_box_plots(k,
                          feat,
                          meta,
                          strain_lut,
                          show_raw_data='date',
                          add_stats=True,
                          )
        plt.legend('',frameon=False)
        plt.tight_layout()
        plt.savefig(saveto / '{}_INK-128_boxplot.png'.format(k), 
                    bbox_inches="tight",
                    dpi=400)
        plt.close('all')
    # %% Now plot BAY-3827 figure
    # Harcode strain colours and order
    strain_lut = {'N2':'lightgrey',
                  'flcn-1':'lightgreen',
                  'fnip-2':'lightskyblue',
                  'N2+BAY-3827':'slategrey',
                  'flcn-1+BAY-3827':'darkgreen',
                  'fnip-2+BAY-3827':'mediumblue'}
    # Choose feature to plot and for which strains
    EXAMPLES = { 'motion_mode_forward_fraction_poststim': ['N2',
                                          'flcn-1',
                                          'fnip-2',
                                          'N2+BAY-3827',
                                          'flcn-1+BAY-3827',
                                          'fnip-2+BAY-3827']}
    # Now call the dictionary keys above to plot the boxplits and save
    for k,v in EXAMPLES.items():
        examples_feat_df, examples_meta_df, dx, gene_list = select_strains(v,
                                                                            CONTROL_STRAIN,
                                                                            feat_df=feat_df,
                                                                            meta_df=meta_df)
        feature_box_plots(k,
                          feat,
                          meta,
                          strain_lut,
                          show_raw_data='date',
                          add_stats=True,
                          )
        plt.legend('',frameon=False)
        plt.tight_layout()
        plt.savefig(saveto / '{}_BAY-3827_boxplot.png'.format(k), 
                    bbox_inches="tight",
                    dpi=400)
        plt.close('all')
        
    # %% Now plot JR-AB-011 figure
    # Harcode strain colours and order
    strain_lut = {'N2':'lightgrey',
                  'flcn-1':'lightgreen',
                  'fnip-2':'lightskyblue',
                  'N2+JR-AB-011':'slategrey',
                  'flcn-1+JR-AB-011':'darkgreen',
                  'fnip-2+JR-AB-011':'mediumblue'}
    # Choose feature to plot and for which strains
    EXAMPLES = { 'angular_velocity_abs_50th_bluelight': ['N2',
                                          'flcn-1',
                                          'fnip-2',
                                          'N2+JR-AB-011',
                                          'flcn-1+JR-AB-011',
                                          'fnip-2+JR-AB-011']}
    # Now call the dictionary keys above to plot the boxplits and save
    for k,v in EXAMPLES.items():
        examples_feat_df, examples_meta_df, dx, gene_list = select_strains(v,
                                                                            CONTROL_STRAIN,
                                                                            feat_df=feat_df,
                                                                            meta_df=meta_df)
        feature_box_plots(k,
                          feat,
                          meta,
                          strain_lut,
                          show_raw_data='date',
                          add_stats=True,
                          )
        plt.legend('',frameon=False)
        plt.tight_layout()
        plt.savefig(saveto / '{}_JR-AB-011_boxplot.png'.format(k), 
                    bbox_inches="tight",
                    dpi=400)
        plt.close('all')
