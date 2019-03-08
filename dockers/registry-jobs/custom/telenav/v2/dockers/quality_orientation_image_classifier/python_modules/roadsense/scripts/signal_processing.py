import pandas as pd
from tqdm import tqdm

tqdm.pandas()

import scipy.signal
from scipy import signal
from scipy.signal import butter, lfilter
import numpy as np

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def add_projection_features(df, min_nr_obs_between_peeks):
    
    acc_projection_col_name = 'acc_prj_on_g'
    df[acc_projection_col_name] = df['accX'] * df['gX'] + df['accY'] * df['gY'] + df['accZ'] * df['gZ']
    
    proj_features_col_names = [acc_projection_col_name]
    for feature_col_name in [acc_projection_col_name]:
        
        indexes_high, _  = scipy.signal.find_peaks(np.array(df[feature_col_name]), 
                                                   height=0.3, 
                                                   distance=min_nr_obs_between_peeks)
        indexes_high = np.array(indexes_high)
        prominences_high = scipy.signal.peak_prominences(df[feature_col_name], indexes_high)[0]
        
        indexes_low, _  = scipy.signal.find_peaks(df[feature_col_name] * -1, 
                                                  height=0.3, 
                                                  distance=min_nr_obs_between_peeks)
        indexes_low = np.array(indexes_low)
        prominences_low = scipy.signal.peak_prominences(df[feature_col_name] * -1, indexes_low)[0]
        
        peaks = np.zeros(len(df))
        for idx in indexes_high:
            peaks[idx] = prominences_high[np.where(indexes_high==idx)]
            
        for idx in indexes_low:
            peaks[idx] += prominences_low[np.where(indexes_low==idx)]
            
        peaks_col_name = f'{feature_col_name}_peaks'
        rolling_peaks_col_name = f'{feature_col_name}_rolling_peaks'
        
        df[peaks_col_name] = pd.Series(peaks, index=df.index)
        df[rolling_peaks_col_name] = df[peaks_col_name].rolling('2s').sum()
    
        proj_features_col_names += [peaks_col_name,rolling_peaks_col_name]
        
    return df,proj_features_col_names



# High & low pass filters
def __butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients so we can check its frequency response.
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def __butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = __butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def __butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def __butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = __butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def low_high_pass_filter(data, Fs, high_threshold_hz=0.5, low_threshold_hz=4):
    filtered_signal = __butter_lowpass_filter(data, low_threshold_hz, Fs)
    filtered_signal = __butter_highpass_filter(filtered_signal, high_threshold_hz, Fs)
    return filtered_signal


def get_filtered_spectogram_bins(data, time_arr, Fs, high_threshold_hz, low_threshold_hz, bins_count = 5):
    # filter signal
    filtered_signal = low_high_pass_filter(data, Fs, high_threshold_hz, low_threshold_hz)
    f, t, Sxx = signal.spectrogram(filtered_signal, Fs, nperseg=256, noverlap=246)
    selected_frequencies = (f>=high_threshold_hz) & (f<=low_threshold_hz)
    filtered_frequencies = f[selected_frequencies]
    # grouping signal in frequeny bins
    Sxx_filtered = Sxx[selected_frequencies, :]
    bin_ids = (np.arange(len(filtered_frequencies)) // (len(filtered_frequencies) / bins_count)).astype(int)
    frequency_bins_amplitude = np.apply_along_axis(lambda x: np.bincount(bin_ids, weights=x), axis=0, arr=Sxx_filtered)
    frequency_bins = np.unique(bin_ids)
        
    return frequency_bins, t, frequency_bins_amplitude


def get_with_freq_bin_features(df, t, Sxx, sampling_interval_ms,freq_bins): 
    datatimes = df.index
    
    freq_features_names = ["f1","f2","f3","f4","f5"]

    freq_features = [{'datetime': datatimes[0], 'f1': np.nan, 'f2': np.nan, 'f3': np.nan, 'f4': np.nan, 'f5': np.nan}]
    dt = int(((datatimes[-1] - datatimes[0]) / len(t)) / np.timedelta64(1, 'ns')) # delta t in ns

    for idx_t in range(len(t)-1):
        freq_feat = {'datetime': datatimes[0] + np.timedelta64((idx_t + 1) * dt, 'ns')}
        for i in range(freq_bins):
            freq_feat['f' + str(i+1)] = Sxx[i, idx_t]
        freq_features.append(freq_feat)
    freq_features.append({'datetime': datatimes[-1], 'f1': np.nan, 'f2': np.nan, 'f3': np.nan, 'f4': np.nan, 'f5': np.nan})
    df_freq_features = pd.DataFrame.from_dict(freq_features)
    df_freq_features.set_index('datetime', inplace=True)
    df_freq_features.fillna(method='ffill', inplace=True)
    df_freq_features.fillna(method='bfill', inplace=True)
    how_resample = {col_name:"mean" for col_name in freq_features_names}
    resampled = df_freq_features.resample(f"{sampling_interval_ms}L", kind='timestamp', how=how_resample)
    interpolated_df = resampled.interpolate(method='time', order=2)

    all_features_df = df.join(interpolated_df)
    return all_features_df, freq_features_names

def add_freq_bin_features(sensor_df, config):
    T = config[cp.FREQUENCY] * 10**-3 # sampling interval (X ms)
    Fs = 1/T # sample rate, Hz
    high_threshold_hz = 0.5
    low_threshold_hz = 4.9
    freq_bins = 5

    time_arr = sensor_df.index.values # time axes
    y = sensor_df['acc_prj_on_g'].values

    f, t, Sxx = get_filtered_spectogram_bins(y, time_arr, Fs, high_threshold_hz, low_threshold_hz,bins_count = freq_bins)
    sensor_df,freq_features_names = get_with_freq_bin_features(sensor_df, t, Sxx, config[cp.FREQUENCY],freq_bins)
    
    
    return sensor_df,freq_features_names