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


import numpy as np
import pandas as pd
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
        xytext=(xpos + sign * full_width * 0.69, median_y+0.5),
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
        xytext=(xpos + sign * full_width * 0.69, p75_y+1),
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
        xytext=(xpos + sign * full_width * 0.4, p95_y+1.5),
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


def find_stationary_periods(xT, yT, buffer_period):
    stationary_periods = []
    start = 0
    while start < len(xT) - 1:
        end = start + 1

        while end < len(xT) and np.abs(xT[end] - xT[start]) == 0 and np.abs(yT[end] - yT[start]) == 0:
            end += 1

        if end - start > 1:
            stationary_periods.append([start, start+buffer_period])
        start = end
    return stationary_periods

def get_angular_offset(xpos, xtrg, ypos, ytrg):
    centroid_gaze = [np.mean(xpos), np.mean(ypos)]
    centroid_target = [np.mean(xtrg), np.mean(ytrg)]

    v = np.array([np.tan(np.radians(centroid_gaze[0])), np.tan(np.radians(centroid_gaze[1])), 1])
    v = v / np.linalg.norm(v)
    vT = np.array([np.tan(np.radians(centroid_target[0])), np.tan(np.radians(centroid_target[1])), 1])
    vT = vT / np.linalg.norm(vT)

    d = np.degrees(np.arccos(np.clip(np.dot(v, vT), -1.0, 1.0)))

    return d

def calculate_velocity(x, y, dt):
    x_velocity = np.diff(x) / dt
    x_velocity = np.insert(x_velocity, 0, 0)
    y_velocity = np.diff(y) / dt
    y_velocity = np.insert(y_velocity, 0, 0)
    velocity = np.sqrt(x_velocity**2 + y_velocity**2)

    return x_velocity, y_velocity, velocity

def rank1_fixation_selection_IVT(x, y, xT, yT, buffer_period, sampling_rate, velocity_threshold, dwell_time):
    stationary_periods = find_stationary_periods(xT, yT, buffer_period=buffer_period)

    rank1_fixations = []

    for (start, end) in stationary_periods:

        x_window = x[start:end]
        y_window = y[start:end]
        _, _, velocity = calculate_velocity(x_window, y_window, dt=1/sampling_rate)

        period_fixations = [start + idx for idx, val in enumerate(velocity) if val < velocity_threshold]

        if len(period_fixations) < dwell_time:
            continue

        min_distance = float('inf')
        rank1_fixation = None

        for i in range(len(period_fixations) - dwell_time + 1):
            if period_fixations[i + dwell_time - 1] - period_fixations[i] == dwell_time - 1:
                start_, end_ = period_fixations[i], period_fixations[i + dwell_time - 1] + 1
                fixation_x = x[start_:end_]
                fixation_y = y[start_:end_]
                fixation_xT = xT[start_:end_]
                fixation_yT = yT[start_:end_]
                
                distance = np.sqrt((fixation_x - fixation_xT) ** 2 + (fixation_y - fixation_yT) ** 2)
                mean_distance = np.mean(distance)

                if mean_distance < min_distance:
                    min_distance = mean_distance
                    rank1_fixation = [start_, end_]
        
        if rank1_fixation is not None:
            rank1_fixations.append(rank1_fixation)

    return rank1_fixations


def get_window_dispersion(window_x, window_y):
    if np.any(np.isnan(window_x)) or np.any(np.isnan(window_y)):
        return np.nan
    dispersion = np.abs(window_x.max() - window_x.min()) + np.abs(window_y.max() - window_y.min())
    return dispersion

def rank1_fixation_selection_IDT(x, y, xT, yT, buffer_period, dispersion_threshold, dwell_time):
    
    stationary_periods = find_stationary_periods(xT, yT, buffer_period=buffer_period)
    rank1_fixations = []
    temporal_dispersion = 30

    for period in stationary_periods:
        start = period[0]
        end = min(start + temporal_dispersion, period[1])
        minimum_distance = float('inf')
        rank1_fixation = None
    
        while start < period[1]:
            IDT_dispersion = get_window_dispersion(x[start:end], y[start:end])

            if IDT_dispersion <= dispersion_threshold:
                while (IDT_dispersion < dispersion_threshold) and (end < period[1]):
                    end += 1
                    if end < period[1]:
                        IDT_dispersion = get_window_dispersion(x[start:end], y[start:end])

                window_size = end - start -1
                if window_size >= dwell_time:
                    minimum_distance = float('inf')
                    rank1_fixation = None

                    for i in range(start, end - dwell_time + 1):
                        fixation = [i, i + dwell_time]
                        fixation_x = x[i:i + dwell_time]
                        fixation_y = y[i:i + dwell_time]
                        fixation_xT = xT[i:i + dwell_time]
                        fixation_yT = yT[i:i + dwell_time]
                        distance = np.sqrt(np.sum((fixation_x - fixation_xT) ** 2 + (fixation_y - fixation_yT) ** 2))
                        mean_distance = np.mean(distance)

                        if mean_distance < minimum_distance:
                            minimum_distance = mean_distance
                            rank1_fixation = fixation

                start, end = end, min(end + temporal_dispersion, period[1])
            else:
                start += 1
                end = min(start + temporal_dispersion, period[1])
        if rank1_fixation is not None:
            rank1_fixations.append(rank1_fixation)
    
    return rank1_fixations

def initialize_kalman_filter(dt):
    KF_K = 0
    KF_x = np.array([0, 0])
    KF_y = np.array([0, 0])
    KF_P = np.array([[1, 0], [0, 1]])
    KF_A = np.array([[1, dt], [0, 1]])
    KF_H = np.array([1, 0])
    KF_Q = np.array([[0.5, 0], [0, 0.5]])
    KF_R = 0.5

    return KF_K, KF_x, KF_y, KF_P, KF_A, KF_H, KF_Q, KF_R

def predict_kalman_filter(KF_x, KF_y, KF_P, KF_A, KF_Q):
    KF_x = np.dot(KF_A, KF_x)
    KF_y = np.dot(KF_A, KF_y)
    KF_P = np.dot(np.dot(KF_A, KF_P), KF_A.T) + KF_Q
    return KF_x, KF_y, KF_P

def calculate_kalman_gain(KF_P, KF_H, KF_R):
    KF_K = np.dot(KF_P, KF_H.T) / (np.dot(np.dot(KF_H, KF_P), KF_H.T) + KF_R)
    return KF_K

def update_kalman_filter(KF_x, KF_y, KF_P, KF_H, KF_K, measurement_x, measurement_y):
    KF_x += np.dot(KF_K, (measurement_x - np.dot(KF_H, KF_x)))
    KF_y += np.dot(KF_K, (measurement_y - np.dot(KF_H, KF_y)))
    KF_P -= np.dot(np.dot(KF_K, KF_H), KF_P)
    return KF_x, KF_y, KF_P

def kalman_filter(x, y, sampling_rate):
    dt = 1 / sampling_rate
    x_velocity, y_velocity, velocity = calculate_velocity(x, y, dt)
    
    KF_K, KF_x, KF_y, KF_P, KF_A, KF_H, KF_Q, KF_R = initialize_kalman_filter(dt)

    K_predicted = np.zeros(len(x))
    K_measured = np.zeros(len(x))
    noise = []

    K_predicted_x = np.zeros(len(x))
    K_predicted_y = np.zeros(len(y))
    updated_x = np.zeros(len(x))
    updated_y = np.zeros(len(y))
    
    for i in range(len(x)):
        if np.isnan(velocity[i]):
            noise.append(i)
        else:
            KF_x, KF_y, KF_P = predict_kalman_filter(KF_x, KF_y, KF_P, KF_A, KF_Q)

            ## Storing predicted
            K_predicted_x[i] = KF_x[0]
            K_predicted_y[i] = KF_y[1]


            predicted_speed_x = KF_x[1]
            predicted_speed_y = KF_y[1]

            K_predicted[i] = np.sqrt(predicted_speed_x**2 + predicted_speed_y**2)
            K_measured[i] = velocity[i]

            KF_K = calculate_kalman_gain(KF_P, KF_H, KF_R)

            measurement_x = np.array([x[i], x_velocity[i]])
            measurement_y = np.array([y[i], y_velocity[i]])

            KF_x, KF_y, KF_P = update_kalman_filter(KF_x, KF_y, KF_P, KF_H, KF_K, measurement_x, measurement_y)

            ## Storing updated
            updated_x[i] = KF_x[0]
            updated_y[i] = KF_y[1]

            
    return K_predicted, K_measured

def classify_gaze_points(length_x, K_predicted, K_measured, window_size, chi_threshold, deviation):
    chi_values = np.cumsum((K_measured - K_predicted) ** 2 / deviation)
    chi_values = np.insert(chi_values, 0, 0)
    chi_values_diff = chi_values[window_size:] - chi_values[:-window_size]

    fixations, saccades = [], []

    for i in range(length_x - window_size + 1):
        KF_chi = chi_values_diff[i]
        if abs(KF_chi) < chi_threshold:
            fixations.append(i)
        else:
            saccades.append(i)

    return fixations

def rank1_fixation_selection_IKF(x, y, xT, yT, fixations, dwell_time, buffer_period):

    stationary_periods = find_stationary_periods(xT, yT, buffer_period=buffer_period)
    rank1_fixations = []

    for (start, end) in stationary_periods:
        period_fixations = [fix for fix in fixations if start <= fix < end]
        if len(period_fixations) < dwell_time:
            continue

        min_distance = float('inf')
        rank1_fixation = None

        for i in range(len(period_fixations) - dwell_time + 1):
            if period_fixations[i + dwell_time - 1] - period_fixations[i] == dwell_time - 1:
                sub_start = period_fixations[i]
                sub_end = period_fixations[i + dwell_time - 1] + 1  # +1 to include the last index

                # Get the corresponding gaze coordinates
                sub_x = x[sub_start:sub_end]
                sub_y = y[sub_start:sub_end]
                sub_xT = xT[sub_start:sub_end]
                sub_yT = yT[sub_start:sub_end]

                # Calculate distances and mean distance
                distance = np.sqrt((np.array(sub_x) - np.array(sub_xT)) ** 2 + 
                                    (np.array(sub_y) - np.array(sub_yT)) ** 2)
                mean_distance = np.mean(distance)

                if mean_distance < min_distance:
                    min_distance = mean_distance
                    rank1_fixation = [sub_start, sub_end]
        
        if rank1_fixation is not None:
            rank1_fixations.append(rank1_fixation)

    return rank1_fixations


def carry_forward_nan_filling(data):
    if np.isnan(data[0]):
        first_valid = 0
        data[0] = first_valid
    for i in range(1, len(data)):
        if np.isnan(data[i]):
            data[i] = data[i - 1]
    return data

def extract_gaze(df, nan_free=True):
    x = df[['x']].values.flatten()
    y = df[['y']].values.flatten()
    xT = df[['xT']].values.flatten()
    yT = df[['yT']].values.flatten()

    # Filling-up nans only
    if nan_free:
        x = carry_forward_nan_filling(x)
        y = carry_forward_nan_filling(y)

    return x, y, xT, yT       
