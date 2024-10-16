#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 14:52:37 2024

This script plots Figure.5D. The number of significant behaioural features
are calculated using permutation t-tests in a pairwise manner between N2
and imb-2 exposed/reared to the same temperature sensitisation.

These stats werecalculated using the 'Figure_5d_temperature_sensitisation_calculate_stats.py' 
script that that is within this repository (and contains paths to the raw data files)

@author: tobrien
"""
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

saveto = Path('/Users/tobrien/Documents/Zenodo/High-throughput behavioural phenotyping of 25 C. elegans disease models including patient-specific mutations /Test')
saveto = saveto / 'Temperature_sensitisation'
saveto.mkdir(exist_ok=True)

X = ['Reared 25C', 'Reared 27C', 'Reared 28C', 'Exposed 28C', 'Exposed 30C', 'Exposed 32C']
Y = [0, 524, 325, 854, 1088, 0]
c = ['red','lightgreen','lightskyblue','saddlebrown','violet','blue',]

ax= sns.barplot(X, 
                 Y,
                 palette=c)
plt.axhline(y=700, linewidth=2, color='red', ls='--')
plt.xticks(rotation=90)
plt.savefig(saveto / 'sig_feats_temp_stress.png',
            dpi=300, bbox_inches='tight')