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
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

## Our helper functions
from helper_funcs import *


def process_file(file_path, buffer_period, method, sampling_rate, dwell_time):
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    subject, session, _ = base_filename.split("_")[1:4]

    df = pd.read_csv(file_path)
    x, y, xT, yT = extract_gaze(df, nan_free=True)

    if method == 'IDT':
        rank1_fixations = rank1_fixation_selection_IDT(
                            x, y, xT, yT, 
                            buffer_period=buffer_period, 
                            dispersion_threshold = 0.5, 
                            dwell_time = dwell_time
                        )

    elif method == 'IKF':
        K_predicted, K_measured = kalman_filter(x, y, sampling_rate=sampling_rate)
        fixations = classify_gaze_points(
                        x, y,
                        K_predicted, K_measured, 
                        window_size=5, 
                        chi_threshold=3.75, 
                        deviation=1000
                    )
        rank1_fixations = rank1_fixation_selection_IKF(
                            x, y,
                            xT, yT, fixations, 
                            dwell_time=dwell_time, 
                            buffer_period=buffer_period
                        )
    
    elif method == 'IVT':
        rank1_fixations = rank1_fixation_selection_IVT(
                            x, y, xT, yT, 
                            buffer_period=buffer_period, 
                            sampling_rate=sampling_rate, 
                            velocity_threshold=30, 
                            dwell_time =dwell_time
                        )
    
    angular_offsets = [get_angular_offset(x[start:end], xT[start:end], y[start:end], yT[start:end]) for start, end in rank1_fixations]


    if angular_offsets:
        E50, E75, E95 = np.percentile(angular_offsets, [50, 75, 95])
    else:
        E50 = E75 = E95 = None

    return subject, session, E50, E75, E95, rank1_fixations

def main():

    datasets = ["gb"]
    methods = ["IVT", "IDT", "IKF"]
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
    CSV_DIR = os.path.join(BASE_DIR, 'CSV')

    for data in datasets:
        if data == 'gb':
            gb = True
            data_directory = os.path.join(BASE_DIR, 'Data')
            csv_files = sorted(glob.glob(os.path.join(data_directory, '*_RAN.csv')))
            csv_files = [f for f in csv_files if 'S_1' in os.path.basename(f)]
        
            sampling_rate = 1000
            dwell_times = [100, 150, 200, 250, 300]
            buffer_periods = [1000, 800, 600, 400]


        for method in methods:
            for dwell_time in dwell_times:
                for buffer_period in buffer_periods:
                    with ProcessPoolExecutor() as executor:
                        results = list(tqdm(executor.map(process_file, csv_files, 
                                    [buffer_period]*len(csv_files), 
                                    [method]*len(csv_files), 
                                    [sampling_rate]*len(csv_files), 
                                    [dwell_time]*len(csv_files)), 
                                    total=len(csv_files), desc=f"Processing files: {method}, DT: {dwell_time}, Buffer-time: {buffer_period} samples"))

                    results = [value for value in results if value is not None]
                    big_list = []
                    for result in results:
                        subject, session, E50, E75, E95, rank1_fixations = result
                        big_list.append([subject, session, E50, E75, E95, len(rank1_fixations)])

                    big_list_df = pd.DataFrame(big_list, columns=['Subject', 'Session', 'E50', 'E75', 'E95', 'Rank-1_Fix']).round(4)
                    big_list_df = big_list_df.dropna()
                    big_list_df.to_csv(os.path.join(CSV_DIR, f'EU_{data}_{method}_CE{buffer_period}ms_DT{dwell_time}ms.csv'), index=False)


if __name__ == '__main__':
    main()

# %%
