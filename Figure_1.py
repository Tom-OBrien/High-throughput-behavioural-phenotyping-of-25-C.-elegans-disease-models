#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 13:46:20 2024

This script makes all of the figure pannels shown in Fig.1:
    - Clustermap of all behavioural features (z-normalised)
        -The script also makes an interactive clustermap (saved as a .html file)
    - Principal coordinate analtsis (z-normalised)
    - Collates all strain stats and makes a lineplot/clustermap of p-values
    for all behavioural features

**All the stats are calculated by the 'Individual_strain_phenotyping_stats_calculation.py' 
script. For ease of use, I have saved them within this repository as the 
'STRAIN_STATS_DIR' variable. However, you can re-run the stats calculation 
script and set the output folder as the input for this file

Likewise, the individual strain phenotyping script imports all the different
datasets. Here I simply call a compiled version of all of theses that are
called 'cleaned_' feature/metadata files in the repository.

@author: tobrien
"""

import pandas as pd
import seaborn as sns
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from pathlib import Path
from scipy import stats
from matplotlib.colors import LogNorm
import plotly.graph_objects as go
from plotly.offline import plot
from sklearn.decomposition import PCA
from tierpsytools.preprocessing.preprocess_features import impute_nan_inf
from tierpsytools.preprocessing.filter_data import (filter_nan_inf,
                                                    select_feat_set)
sys.path.insert(0, '/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Code/Helper_Functions_and_Scripts')
from helper import (filter_features,
                    make_colormaps,
                    STIMULI_ORDER, 
                    CUSTOM_STYLE)
# %%
FEAT_FILE = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/featurematrix_all_datasets_combined.csv')
METADATA_FILE = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/metadata_all_datasets_combined.csv')
STRAIN_STATS_DIR = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/Strain_stats_and_features_to_plot')
Save_Dir = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Test')
saveto = Save_Dir 
saveto.mkdir(exist_ok=True)
# Choose if to only analyse Tierpsy256 set
Tierpsy_256 = False
# Choose whether to make an interactive clustermap of the data
interactive_clustermap = True
CONTROL_STRAIN = 'N2'

# %% Read in the data
if __name__ == '__main__':
    featMat = pd.read_csv(FEAT_FILE, index_col=False)
    meta = pd.read_csv(METADATA_FILE, index_col=False)
    # Filter to keep only Tierpsy 256 set (if selected above) and save to a 
    # new directory
    if Tierpsy_256==True:
        featMat = select_feat_set(features=featMat, 
                                   tierpsy_set_name='tierpsy_256', 
                                   append_bluelight=True)
        saveto = saveto / 'Tierpsy256_only'
        saveto.mkdir(exist_ok=True)
        
    #%% filter features
    featMat = filter_nan_inf(featMat, 0.5, axis=1, verbose=True)
    meta = meta.loc[featMat.index]
    featMat = filter_nan_inf(featMat, 0.05, axis=0, verbose=True)
    featMat = featMat.fillna(featMat.mean())
    meta = meta.loc[featMat.index]
    featMat, meta, featsets = filter_features(featMat,
                                                 meta)
    # Impute nans using Tierpsy
    feat_nonan = impute_nan_inf(featMat)
    # Z-normalise data
    featZ = pd.DataFrame(data=stats.zscore(feat_nonan[featsets['all']], axis=0),
                         columns=featsets['all'],
                         index=feat_nonan.index)
    assert featZ.isna().sum().sum() == 0

    #%% Set style for all figures
    plt.style.use(CUSTOM_STYLE)
    sns.set_style('ticks')
    # Identify unique genes within the metadata
    genes = [g for g in meta.worm_gene.unique() if g != CONTROL_STRAIN]
    genes.sort()
    # Make colour maps for strains and stimuli
    strain_lut, stim_lut, feat_lut = make_colormaps(genes,
                                                    featlist=featsets['all'])
    #%% Look for strain_stats.csv files in defined directory
    strain_stats = [s for s in (STRAIN_STATS_DIR).glob('**/*_stats.csv')]
    print(('Collating pValues for {} worm strains').format(len(strain_stats)))
    # Combine all strain stats into one dataframe and reorder columns so worm
    # gene is the first one (easier to read/double check in variable explorer)
    combined_strains_stats = pd.concat([pd.read_csv(f) for f in strain_stats])
    heat_map_row_colors = combined_strains_stats['worm_gene'].map(strain_lut)
    first_column = combined_strains_stats.pop('worm_gene')
    combined_strains_stats.insert(0, 'worm_gene', first_column )
    # Set worm gene as index for dataframe- this removes this comlumn from df
    combined_strains_stats = combined_strains_stats.set_index(['worm_gene'])
    # Now count total features in df (total features) and save as variable
    total_feats = len(combined_strains_stats.columns)
    # Count nan's in df for each strain/row
    null_feats = combined_strains_stats.isnull().sum(axis=1)
    # Compute total number of significant feats for each strain
    sig_feats = total_feats - null_feats
    # Save as a dataframe (indexed by worm gene)
    sig_feats = pd.DataFrame(sig_feats)
    # Naming column containing number of significant feats
    sig_feats = sig_feats.rename(columns={0: 'Total_Significant_Features'}) 
    # Sorting dataframe from most -> fewest significant feats
    sig_feats = sig_feats.sort_values(by='Total_Significant_Features', axis=0, 
                                      ascending=False)
    # Resting index on ordered df for purposes of plotting later on
    sig_feats = sig_feats.reset_index()
    # Print a summary of the number of significant features
    print('Total number of features {}'.format(total_feats))
    print(sig_feats)
    
    #%% Make a line plot of total significant features ordered save as heatmap
    sns.set_style('ticks')
    l = sns.lineplot(data=sig_feats, 
                      x='worm_gene', 
                      y='Total_Significant_Features',
                      color='black')
    plt.xticks(rotation=90, fontsize=13)
    plt.yticks(rotation=45, fontsize=14)
    l.set_xlabel(' ', fontsize=18)
    l.set_ylabel('Number of Significant Features', fontsize=16, rotation=90)
    plt.yticks([0 ,1000, 2000, 3000, 4000, 5000, 6000, 7000], fontsize=14)
    l.axhline(y=0, color='black', linestyle=':', alpha=0.2)
    plt.savefig(saveto / 'sig_feats_lineplot.png', bbox_inches='tight',
                dpi=300)

    #%% Make heatmap of strains showing number of significant features
    
    # To make heatmap easy to interpret I set values to either 1 or 0
    # This means that it can be coloured as black-white for sig/non-sig feats
    combined_strain_stat_copy = combined_strains_stats
    heatmap_stats = combined_strain_stat_copy.fillna(value=1)
    # I then copy the indexing from the sig feats line plot
    sig_feats = sig_feats.set_index('worm_gene')
    heatmap_stats = heatmap_stats.reindex(sig_feats.index) 
    # Here I set colours for the heatmap 
    hm_colors = ((0.0, 0.0, 0.0), (0.95, 0.95, 0.9))
    hm_cmap = LinearSegmentedColormap.from_list('Custom', 
                                                hm_colors, 
                                                len(hm_colors))
    plt.subplots(figsize=[7.5,5])
    plt.gca().yaxis.tick_right()
    plt.yticks(fontsize=9)
    # Plot the heatmap
    ax=sns.heatmap(data=heatmap_stats,
                    vmin=0,
                    vmax=0.5,
                    xticklabels=False,
                    yticklabels=True,
                    cbar_kws = dict(use_gridspec=False,location="top"),
                    cmap=hm_cmap)
    # Add in the custom coour bar
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([0.1,  0.4])
    colorbar.set_ticklabels(['P < 0.05', 'P > 0.05'])
    ax.set_ylabel('')
    plt.savefig(saveto / 'ordered_formatted_heatmap.png', 
                bbox_inches='tight', dpi=300)
    
    #%% This does the same as above, but colours the p-val by significance using
    # a log(10) scale. First I reindex using the same method as above
    combined_strains_stats = combined_strains_stats.reindex(sig_feats.index)
    # Then we simply plot, I define the min/max sig values based on strain stats
    plt.subplots(figsize=[7.5,5])
    plt.gca().yaxis.tick_right()
    plt.yticks(fontsize=9)
    ax=sns.heatmap(data=combined_strains_stats,
                    # norm=LogNorm(vmin=1e-03, vmax=1e-01),
                    norm=LogNorm(vmin=1e-04, vmax=5e-02),
                    xticklabels=False,
                    yticklabels=True,
                    cbar_kws = dict(use_gridspec=False,
                                    location="top",
                                    # ticks=[1e-03 ,1e-02, 1e-01],
                                    ticks=[1e-04, 1e-03 ,1e-02, 5e-02],
                                    format='%.e'))
    # Add in the custom colour bar
    ax.collections[0].colorbar.set_label("P-value")
    # Save
    plt.savefig(saveto / 'sig_feat_heatmap_w_colours.png', 
                bbox_inches='tight', dpi=300)

# %% Group and concatenate the data by worm gene
    group_vars = ['worm_gene']
    featZ_grouped = pd.concat([featZ,meta],axis=1).groupby(['worm_gene']).mean() 
    featZ_grouped.reset_index(inplace=True)
    row_colors = featZ_grouped['worm_gene'].map(strain_lut)
    # Now make the clustermap
    clustered_features = {}
    plt.figsize=(24,60)
    for stim, fset in featsets.items():
        col_colors = featZ_grouped[fset].columns.map(feat_lut)   #This clustermap is coloured using the feature look up table
        plt.figure(figsize=[15,20])
        sns.set(font_scale=0.8)
        cg = sns.clustermap(featZ_grouped[fset],
                        row_colors=row_colors,
                        col_colors=col_colors,
                        vmin=-1.5,
                        vmax=1.5,
                        yticklabels=featZ_grouped['worm_gene'])
        cg.ax_heatmap.axes.set_xticklabels([])
        cg.savefig(Path(saveto) / '{}_clustermap.png'.format(stim), dpi=300)
    plt.show()
    plt.close('all')
    # %% This will make an interactive heatmap as a static .html file
    if interactive_clustermap:
            featZ_grouped = pd.concat([featZ,meta],axis=1).groupby(['worm_gene']).mean() 
            cg = sns.clustermap(featZ_grouped[featsets['all']], 
                                            vmin=-2,
                                            vmax=2
                                            )
                        
            plt.close('all')
            # get order of features and worm strains in clustermap
            row_order = cg.dendrogram_row.reordered_ind
            col_order = cg.dendrogram_col.reordered_ind     
            # re-order df to match clustering
            clustered_df_final = featZ_grouped.loc[featZ_grouped.index[row_order], featZ_grouped.columns[col_order]]
            # Define your heatmap
            intheatmap = ( go.Heatmap(x=clustered_df_final.columns, 
                            y=clustered_df_final.index, 
                            z=clustered_df_final.values,  
                            colorscale='Inferno', 
                            zmin=-2,
                            zmax=2,
                            showscale=True))
            
            intfig_cl = go.Figure(data=intheatmap)
            intfig_cl.update_xaxes(showticklabels=False)  
            intfig_cl.update_yaxes(showticklabels=False, autorange="reversed") 
            # Define your layout, adjusting colorbar size
            intfig_cl.update_layout({
                'width': 1200,
                'height': 550,
                'margin': dict(l=0,r=0,b=0,t=0),
                'showlegend': True,
                'hovermode': 'closest',
            })
            plot(intfig_cl, filename = str(saveto / "InteractiveClustermap_cosine.html"),
                  config={"displaylogo": False,
                          "displayModeBar": True,
                          "scale":10},
                  auto_open=False)

    #%% Now we're going to plot the principal coordinate analysis of
    # features as they move through the different periods of imaging
    # First, make long form feature matrix
    long_featmat = []
    for stim,fset in featsets.items():
        if stim != 'all':
            _featmat = pd.DataFrame(data=featMat.loc[:,fset].values,
                                    columns=['_'.join(s.split('_')[:-1])
                                                for s in fset],
                                    index=featMat.index)
            _featmat['bluelight'] = stim
            _featmat = pd.concat([_featmat,
                                  meta.loc[:,'worm_gene']],
                                  axis=1)
            long_featmat.append(_featmat)
    long_featmat = pd.concat(long_featmat,
                              axis=0)
    long_featmat.reset_index(drop=True,
                              inplace=True)
    full_fset = list(set(long_featmat.columns) - set(['worm_gene', 'bluelight']))
    long_feat_nonan = impute_nan_inf(long_featmat[full_fset])
    long_meta = long_featmat[['worm_gene', 'bluelight']]
    long_featmatZ = pd.DataFrame(data=stats.zscore(long_feat_nonan[full_fset], axis=0),
                                  columns=full_fset,
                                  index=long_feat_nonan.index)
    assert long_featmatZ.isna().sum().sum() == 0
    #%% Generate PCAs
    pca = PCA()
    X2=pca.fit_transform(long_featmatZ.loc[:,full_fset])
    # Explain PC variance using cumulative variance
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    thresh = cumvar <= 0.95 #set 95% variance threshold
    cut_off = int(np.argwhere(thresh)[-1])
    # Plot above as a figure
    plt.figure()
    plt.plot(range(0, len(cumvar)), cumvar*100)
    plt.plot([cut_off,cut_off], [0, 100], 'k')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('variance explained')
    plt.tight_layout()
    plt.savefig(saveto / 'long_df_variance_explained.png', dpi =300)
    # #now put the 1:cut_off PCs into a dataframe
    PCname = ['PC_%d' %(p+1) for p in range(0,cut_off+1)]
    PC_df = pd.DataFrame(data=X2[:,:cut_off+1],
                          columns=PCname,
                          index=long_featmatZ.index)
    PC_plotting = pd.concat([PC_df,
                              long_meta[['worm_gene',
                                            'bluelight']]],
                              axis=1)
    # Groupby worm gene to see the trajectory through PC space
    PC_plotting_grouped = PC_plotting.groupby(['worm_gene',
                                                'bluelight']).mean().reset_index()
    PC_plotting_grouped['stimuli_order'] = PC_plotting_grouped['bluelight'].map(STIMULI_ORDER)
    PC_plotting_grouped.sort_values(by=['worm_gene',
                                        'stimuli_order'],
                                    ascending=True,
                                    inplace=True)
    # Calculate standard error of mean of PC matrix computed above
    PC_plotting_sd = PC_plotting.groupby(['worm_gene',
                                          'bluelight']).sem().reset_index()
    # Map to stimuli order of PC_Grouped dataframe
    PC_plotting_sd['stimuli_order'] = PC_plotting_sd['bluelight'].map(STIMULI_ORDER)
    
#%% Make PC plots of all strains
    PC_cmap = {'N2':('royalblue'),
                'blos-1':('lightgrey'),
                'blos-8':('lightgrey'),
                'blos-9':('lightgrey'),
                'ccpp-1':('lightgrey'),
                'cpx-1':('lightgrey'),
                'flcn-1':('lightgrey'),
                'fnip-2':('red'),
                'imb-2[D157N]':('lightgrey'),
                'irk-1':('lightgrey'),
                'let-526':('lightgrey'),
                'ncap-1':('lightgrey'),
                'odr-8':('red'),
                'pacs-1[E205K]':('lightgrey'),
                'pde-1':('lightgrey'),
                'pde-5':('lightgrey'),
                'pmp-4':('lightgrey'),
                'R10E11.6':('lightgrey'),
                'rpy-1':('lightgrey'),
                'sam-4':('lightgrey'),
                'sec-31':('lightgrey'),
                'shl-1':('lightgrey'),
                'smc-3[K115E]':('lightgrey'),
                'tmem-222':('lightgrey'),
                'vps-50':('lightgrey'),
                'Y47DA.1[R298W]':('lightgrey')}

    plt.figure(figsize = [14,12])    
    s=sns.scatterplot(x='PC_1',
                    y='PC_2',
                    data=PC_plotting_grouped,
                    hue='worm_gene',
                    style='bluelight',
                    style_order=STIMULI_ORDER.keys(),
                    hue_order=PC_cmap.keys(),
                    palette=PC_cmap,
                    linewidth=0,
                    s=350)
    s.errorbar(
                x=PC_plotting_grouped['PC_1'],
                y=PC_plotting_grouped['PC_2'],
                xerr=PC_plotting_sd['PC_1'], 
                yerr=PC_plotting_sd['PC_2'],
                fmt='.',
                alpha=0.2,
                )
    ll=sns.lineplot(x='PC_1',
                y='PC_2',
                data=PC_plotting_grouped,
                hue='worm_gene',
                hue_order=PC_cmap.keys(),
                palette=PC_cmap,
                alpha=0.8,
                legend=False,
                sort=False)    
    plt.autoscale(enable=True, axis='both')
    # plt.axis('equal')
    plt.legend('',frameon=False)
    # plt.legend(loc='right', bbox_to_anchor=(0.8, 0.25, 0.5, 0.5), fontsize='large')
    plt.xlabel('PC_1 ({}%)'.format(np.round(pca.explained_variance_ratio_[2]*100,2)))
    plt.ylabel('PC_2 ({}%)'.format(np.round(pca.explained_variance_ratio_[3]*100,2)))                                 
    plt.tight_layout()
    plt.savefig(saveto / 'PC1PC2_trajectory_space.png', dpi=400)
    