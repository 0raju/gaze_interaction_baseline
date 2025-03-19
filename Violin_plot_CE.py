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
from scipy.stats import gaussian_kde

def violin_plot(ax, values, xpos, text_side, full_width, font_size, min_label_sep=0.05, scatter_alpha=0.5, pctile_char="U"):
    if text_side == "left":
        sign = -1
        ha = "right"
    elif text_side == "right":
        sign = 1
        ha = "left"
    else:
        raise ValueError("text_side should be 'left' or 'right'")

    kde = gaussian_kde(values)
    density = kde(values)
    width = full_width / np.max(density)
    jitter_strength = density * width
    jitter = 2 * np.random.rand(*values.shape) - 1
    
    ax.violinplot(
        dataset=values,
        positions=[xpos],
        showmeans=False,
        showextrema=False,
        showmedians=False,
    )

    ax.scatter(
        x=xpos + jitter * jitter_strength,
        y=values,
        alpha=scatter_alpha,
        edgecolors="none",
    )

    ax.boxplot(
        values,
        positions=[xpos],
        widths=[full_width * 0.2],
        showfliers=False,
        patch_artist=True,
        boxprops={
            "facecolor": "gray",
            "color": "black",
        },
        medianprops={
            "color": "black",
        },
    )

    ax.scatter(
        x=xpos,
        y=np.mean(values),
        c="white",
        marker="o",
        edgecolors="black",
        zorder=3,
    )

    median = np.median(values)
    p75 = np.quantile(values, q=0.75)
    p95 = np.quantile(values, q=0.95)

    vmax = np.max(values)
    vmin = np.min(values)
    vrange = vmax - vmin

    median_y = median
    if (p75 - median_y) / vrange < min_label_sep:
        p75_y = median_y + min_label_sep * vrange
    else:
        p75_y = p75
    if (p95 - p75_y) / vrange < min_label_sep:
        p95_y = p75_y + min_label_sep * vrange
    else:
        p95_y = p95

    ax.annotate(
        f"{pctile_char.upper()}50={median:.2f}",
        xy=(xpos, median),
        xytext=(xpos - sign * full_width*0.69, median_y+0.5),
        arrowprops={
            "arrowstyle": "-",
            "facecolor": "black",
            "connectionstyle": "arc3",
            "lw": 2,
        },
        fontfamily="DejaVu Sans",
        fontsize=font_size,
        va="center",
        ha=ha,
    )
    ax.annotate(
        f"{pctile_char.upper()}75={p75:.2f}",
        xy=(xpos, p75),
        xytext=(xpos - sign * full_width*0.69, p75_y+1.5),
        arrowprops={
            "arrowstyle": "-",
            "facecolor": "black",
            "connectionstyle": "arc3",
            "lw": 2,
        },
        fontfamily="DejaVu Sans",
        fontsize=font_size,
        va="center",
        ha=ha,
    )
    ax.annotate(
        f"{pctile_char.upper()}95={p95:.2f}",
        xy=(xpos, p95),
        xytext=(xpos - sign * full_width*0.69, p95_y+2.5),
        arrowprops={
            "arrowstyle": "-",
            "facecolor": "black",
            "connectionstyle": "arc3",
            "lw": 2,
        },
        fontfamily="DejaVu Sans",
        fontsize=font_size,
        va="center",
        ha=ha,
        # bbox=dict(boxstyle="round,pad=0.2", edgecolor="none", facecolor="lightyellow")
    )


# Set base directory dynamically
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
CSV_DIR = os.path.join(BASE_DIR, 'CSV')
FIGURE_DIR = os.path.join(BASE_DIR, 'Figures')

# Ensure output directories exist
os.makedirs(FIGURE_DIR, exist_ok=True)

def read_and_prepare_data(base_path, dataset, algorithm, buffer_periods, dwell_time):

    data_E50, data_E95, methods_E50, methods_E95 = [], [], [], []

    for buffer_period in buffer_periods:
        file_path = os.path.join(base_path, f'EU_{dataset}_{algorithm}_CE{buffer_period}ms_DT{dwell_time}ms.csv')

        df = pd.read_csv(file_path)
        data_E50.append(df['E50'].values)
        data_E95.append(df['E95'].values)
        methods_E50.append(f'{buffer_period}ms')
        methods_E95.append(f'{buffer_period}ms')

    max_length = max(max(len(values) for values in data_E50), max(len(values) for values in data_E95))

    # Pad arrays to match the maximum length
    pad_array = lambda arr: np.pad(arr, (0, max_length - len(arr)), 'constant', constant_values=np.nan)
    padded_E50 = [pad_array(values) for values in data_E50]
    padded_E95 = [pad_array(values) for values in data_E95]

    # Convert to DataFrame and reshape for plotting
    data_E50_df = pd.DataFrame(dict(zip(methods_E50, padded_E50)))
    data_E95_df = pd.DataFrame(dict(zip(methods_E95, padded_E95)))
    data_long_E50 = pd.melt(data_E50_df, var_name='Method', value_name='E50').dropna()
    data_long_E95 = pd.melt(data_E95_df, var_name='Method', value_name='E95').dropna()
    
    return data_long_E50, data_long_E95, methods_E50, methods_E95

def create_violin_plots(data_long_E50, data_long_E95, methods_E50, methods_E95, dataset, algorithm, dwell_time, success):
    """Generates violin plots for E50 and E95 metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(34, 12))

    for i, method in enumerate(methods_E50):
        values = data_long_E50[data_long_E50['Method'] == method]['E50'].values
        violin_plot(ax=axes[0], values=values, xpos=i + 1, text_side='right', full_width=0.5, font_size=38)
        axes[0].annotate(f'{success[i]}%', xy=(i + 1, 17.4), xytext=(i + 1, 18.4), fontsize=36, fontweight='bold', ha='center',
                         bbox=dict(facecolor='lightblue', edgecolor='black', boxstyle='round,pad=0.5'))
    
    axes[0].set_xticks(range(1, len(methods_E50) + 1))
    axes[0].set_xticklabels(methods_E50)
    axes[0].set_xlabel('E50 @ Buffer-period', fontsize=38)
    axes[0].set_ylabel('Angular Offset (dva)', fontsize=38)
    axes[0].tick_params(axis='both', which='major', labelsize=36)
    axes[0].set_ylim(0, 20)
    axes[0].set_yticks(np.arange(0, 21, 5))

    for i, method in enumerate(methods_E95):
        values = data_long_E95[data_long_E95['Method'] == method]['E95'].values
        violin_plot(ax=axes[1], values=values, xpos=i + 1, text_side='right', full_width=0.5, font_size=38)
    
    axes[1].set_xticks(range(1, len(methods_E95) + 1))
    axes[1].set_xticklabels(methods_E95)
    axes[1].set_xlabel('E95 @ Buffer-period', fontsize=38)
    axes[1].set_ylabel('Angular Offset (dva)', fontsize=38)
    axes[1].tick_params(axis='both', which='major', labelsize=36)
    axes[1].set_ylim(0, 33)

    plt.subplots_adjust(wspace=0.2)
    plt.tight_layout(pad=1.0)
    output_path = os.path.join(FIGURE_DIR, f'Violinplot_{dataset}_CE_DT{dwell_time}ms_{algorithm}.png')
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.show()

# Experiment settings
dataset = 'gb'
dwell_time = 100
buffer_periods = [1000, 800, 600, 400]

success_IVT = ['87.5', '86.2', '83.7', '79.1']
success_IDT = ['99.4', '99.1', '97.9', '94.9']
success_IKF = ['97.2', '96.8', '95.6', '91.7']

# Plot for IVT
data_long_E50, data_long_E95, methods_E50, methods_E95 = read_and_prepare_data(CSV_DIR, dataset, 'IVT', buffer_periods, dwell_time)
create_violin_plots(data_long_E50, data_long_E95, methods_E50, methods_E95, dataset, 'IVT', dwell_time, success_IVT)

# Plot for IDT
data_long_E50, data_long_E95, methods_E50, methods_E95 = read_and_prepare_data(CSV_DIR, dataset, 'IDT', buffer_periods, dwell_time)
create_violin_plots(data_long_E50, data_long_E95, methods_E50, methods_E95, dataset, 'IDT', dwell_time, success_IDT)

# Plot for IKF
data_long_E50, data_long_E95, methods_E50, methods_E95 = read_and_prepare_data(CSV_DIR, dataset, 'IKF', buffer_periods, dwell_time)
create_violin_plots(data_long_E50, data_long_E95, methods_E50, methods_E95, dataset, 'IKF', dwell_time, success_IKF)

# %%
