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


## Replicate Table-1 & Supllementary Table-2

import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))

CSV_DIR = os.path.join(BASE_DIR, 'CSV')

total_recordings = 644 # GazeBase dataset has 322 subjects, 2 session per subject

datasets = ["gb"]
methods = ["IVT", "IDT", "IKF"]
buffer_periods = [1000, 900, 800, 700, 600, 500, 400]
dwell_times = [100, 150, 200, 250, 300]

for data in datasets:
    for buffer_period in buffer_periods:
        for method in methods:
            for dt  in dwell_times:
                big_list_df = pd.read_csv(os.path.join(CSV_DIR, f'EU_gb_{method}_CE{buffer_period}ms_DT{dt}ms.csv'))
                success_rate = big_list_df['Rank-1_Fix'].sum()/total_recordings
                print(method, buffer_period, dt, f'{np.round(success_rate, 1)} ({total_recordings-len(big_list_df)})')
            print("------------")


# %%