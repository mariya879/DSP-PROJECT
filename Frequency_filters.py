import numpy as np
import scipy.signal
import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['time'] = pd.to_datetime(df['time'])
    return df

def low_pass_filter_fft(df, column_name, cutoff_frequency):
    data = df[column_name].values
    fft_result = np.fft.fft(data)
    frequencies = np.fft.fftfreq(len(data))
    fft_result[np.abs(frequencies) > cutoff_frequency] = 0
    filtered_data = np.fft.ifft(fft_result)
    return filtered_data

def high_pass_filter_fft(df, column_name, cutoff_frequency):
    data = df[column_name].values
    fft_result = np.fft.fft(data)
    frequencies = np.fft.fftfreq(len(data))
    fft_result[np.abs(frequencies) < cutoff_frequency] = 0
    filtered_data = np.fft.ifft(fft_result).real
    return filtered_data

def band_pass_filter_fft(df, column_name, lower_cutoff, upper_cutoff):
    data = df[column_name].values
    fft_result = np.fft.fft(data)
    frequencies = np.fft.fftfreq(len(data))
    fft_result[(np.abs(frequencies) < lower_cutoff) | (np.abs(frequencies) > upper_cutoff)] = 0
    filtered_data = np.fft.ifft(fft_result).real
    return filtered_data

def plot_data_and_filtered(df, time_column, data_column, filtered_data, title):
    plt.figure(figsize=(12, 6))
   
    plt.plot(df[time_column].values, df[data_column].values, label='Original Data')
    
    plt.plot(df[time_column].values, np.abs(filtered_data), 'r', label='Filtered Data')
    plt.xlabel('Time')
    plt.ylabel('Data Value')
    plt.grid()
    plt.legend()
    plt.title(title)
    plt.show()
    
    
def main():
    file_path = 'sv02mdpl_4_2_0_0.csv'
    df = load_data(file_path)
   
    
    
    cutoff_freq_lp_fft = 0.2
    filtered_data_lp_fft = low_pass_filter_fft(df, '536', cutoff_freq_lp_fft)
    plot_data_and_filtered(df, 'time', '536', filtered_data_lp_fft, 'Low-Pass Filtered Data (FFT)')
    
    cutoff_freq_hp_fft = 0.25
    filtered_data_hp_fft = high_pass_filter_fft(df, '536', cutoff_freq_hp_fft)
    plot_data_and_filtered(df, 'time', '536', filtered_data_hp_fft, 'High-Pass Filtered Data (FFT)')
   
    lower_cutoff_bp_fft = 0.05
    upper_cutoff_bp_fft = 0.15
    filtered_data_bp_fft = band_pass_filter_fft(df, '536', lower_cutoff_bp_fft, upper_cutoff_bp_fft)
    plot_data_and_filtered(df, 'time', '536', filtered_data_bp_fft, 'Band-Pass Filtered Data (FFT)')

    
    
if __name__ == "__main__":
    main()