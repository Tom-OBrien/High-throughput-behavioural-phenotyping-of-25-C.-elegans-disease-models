#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 16:32:09 2024

This figure calls a .csv file containing the combined stats for the pairwise
comparison of each drug treatment vs the same strain treated with DMSO only
for every behavioural feature extracted by Tierpsy.

These stats were calculated using the 
'Stats_calculation_for_each_drug_treatment.py' script and then
the number of significant features manually appended into the file used here.

@author: tobrien
"""

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

saveto = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Test')
data = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Data/Folliculin_mutant_drug_screen/Stats/Combined_stats/Stats_of_each_compound_compared_to_the_control.csv')
data = pd.read_csv(data)

ax = sns.barplot(data=data, x='drug', y='sig_feats', hue='worm_strain')

ax.set_ylabel('No. Significant Feats vs Control')
ax.set_xlabel('Drug')
plt.xticks(rotation=90)

plt.savefig(saveto/'Number_of_significant_features_vs_control.png',
           dpi=300, bbox_inches='tight')
plt.close('all')
