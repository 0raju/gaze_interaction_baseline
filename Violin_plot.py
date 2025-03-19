# %%

"""
This code is part of research paper:
Title: Evaluating Eye Tracking Signal Quality with Real-time Gaze Interaction Simulation: A Study Using an Offline Dataset.
Authors: Mehedi Hasan Raju, Samantha Aziz, Michael J. Proulx, and Oleg V. Komogortsev
Published: 2025 Symposium on Eye Tracking Research and Applications (ETRA '25).
DOI: https://doi.org/10.1145/3715669.3723119

This work and its accompanying codebase are licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License 
(https://creativecommons.org/licenses/by-nc-sa/4.0/). 

For all other uses, please contact the Office for Commercialization and Industry Relations at Texas State University http://www.txstate.edu/ocir/

Property of Texas State University.

For inquiries and further information, please contact Mehedi Hasan Raju (m.raju@txstate.edu)

"""


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from helper_funcs import violin_plot

# Define experimental parameters, these are default values to generate results as per manuscript
buffer_periods = [1000]
datasets = ['gb'] ## Represents GazeBase dataset
dwell_times = [100]

# Base directory (modify as needed)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))

CSV_DIR = os.path.join(BASE_DIR, 'CSV')
FIGURE_DIR = os.path.join(BASE_DIR, 'Figures')

# Ensure output directory exists
os.makedirs(FIGURE_DIR, exist_ok=True)

for buffer_period in buffer_periods:
    for dataset in datasets:
        for dt in dwell_times:
            # Construct file paths dynamically
            file_prefix = os.path.join(CSV_DIR, f"EU_{dataset}_")
            EU_IDT_df = pd.read_csv(f"{file_prefix}IDT_CE{buffer_period}ms_DT{dt}ms.csv")
            EU_IVT_df = pd.read_csv(f"{file_prefix}IVT_CE{buffer_period}ms_DT{dt}ms.csv")
            EU_IKF_df = pd.read_csv(f"{file_prefix}IKF_CE{buffer_period}ms_DT{dt}ms.csv")

            # Extract relevant columns
            E50_IDT, E95_IDT = EU_IDT_df['E50'].values, EU_IDT_df['E95'].values
            E50_IVT, E95_IVT = EU_IVT_df['E50'].values, EU_IVT_df['E95'].values
            E50_IKF, E95_IKF = EU_IKF_df['E50'].values, EU_IKF_df['E95'].values

            # Determine the maximum length for padding
            max_length = max(len(E50_IDT), len(E50_IVT), len(E50_IKF))

            # Pad arrays to match the maximum length
            def pad_array(arr, max_length):
                return np.pad(arr, (0, max_length - len(arr)), 'constant', constant_values=np.nan)

            E50_IDT, E95_IDT = pad_array(E50_IDT, max_length), pad_array(E95_IDT, max_length)
            E50_IKF, E95_IKF = pad_array(E50_IKF, max_length), pad_array(E95_IKF, max_length)
            E50_IVT, E95_IVT = pad_array(E50_IVT, max_length), pad_array(E95_IVT, max_length)

            # Create dataframes for visualization
            data_E50 = pd.DataFrame({'IVT': E50_IVT, 'IDT': E50_IDT, 'IKF': E50_IKF})
            data_E95 = pd.DataFrame({'IVT': E95_IVT, 'IDT': E95_IDT, 'IKF': E95_IKF})
            
            # Convert to long format
            data_long_E50 = pd.melt(data_E50, var_name='Method', value_name='E50').dropna(subset=['E50'])
            data_long_E95 = pd.melt(data_E95, var_name='Method', value_name='E95').dropna(subset=['E95'])

            # Initialize figure
            fig, axes = plt.subplots(1, 2, figsize=(30, 12))

            # Plot E50 violin plot
            methods_E50 = data_long_E50['Method'].unique()
            for i, method in enumerate(methods_E50):
                values = data_long_E50[data_long_E50['Method'] == method]['E50'].values
                violin_plot(axes[0], values, i + 1, text_side='right', full_width=0.5, font_size=34)
            
            # Annotate success rates
            success = ['87.5', '99.4', '97.2']  ## From success_rate.py code
            for i, method in enumerate(methods_E50):
                axes[0].annotate(
                    f'{success[i]}%',
                    xy=(i + 1, 6.5),
                    xytext=(i + 1, 7.5),
                    fontsize=34, fontweight='bold', ha='center',
                    bbox=dict(facecolor='lightblue', edgecolor='black', boxstyle='round,pad=0.5')
                )

            # Format E50 plot
            axes[0].set_xticks(range(1, len(methods_E50) + 1))
            axes[0].set_xticklabels(methods_E50)
            axes[0].set_xlabel('E50 @ Classification Algorithms', fontsize=36)
            axes[0].set_ylabel('Angular Offset (dva)', fontsize=36)
            axes[0].tick_params(axis='both', which='major', labelsize=34)
            axes[0].set_ylim(0, 8.1)

            # Plot E95 violin plot
            methods_E95 = data_long_E95['Method'].unique()
            for i, method in enumerate(methods_E95):
                values = data_long_E95[data_long_E95['Method'] == method]['E95'].values
                violin_plot(axes[1], values, i + 1, text_side='right', full_width=0.5, font_size=34)
            
            # Format E95 plot
            axes[1].set_xticks(range(1, len(methods_E95) + 1))
            axes[1].set_xticklabels(methods_E95)
            axes[1].set_xlabel('E95 @ Classification Algorithms', fontsize=36)
            axes[1].set_ylabel('Angular Offset (dva)', fontsize=36)
            axes[1].tick_params(axis='both', which='major', labelsize=34)
            axes[1].set_ylim(0, 30)

            # Adjust layout and save figure
            plt.subplots_adjust(wspace=0.05)
            plt.tight_layout(pad=1.0)
            output_path = os.path.join(FIGURE_DIR, f'Violinplot_{dataset}_CE{buffer_period}ms_DT{dt}ms.png')
            plt.savefig(output_path, dpi=600, bbox_inches='tight')

# %%
