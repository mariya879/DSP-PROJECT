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

def adaptive_filter_gaussian(data, window_length, std_dev):
    gaussian_window = scipy.signal.windows.gaussian(window_length, std=std_dev)
    filtered_data = scipy.signal.convolve(data, gaussian_window, mode='same') / sum(gaussian_window)
    return filtered_data

def butterworth_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    filtered_data = scipy.signal.lfilter(b, a, data)
    return filtered_data

def kalman_filter(data, process_variance, measurement_variance):
    n = len(data)
    x_hat = np.zeros(n)  # Initial estimate
    P = np.zeros(n)  # Initial estimate error covariance
    x_hat_minus = np.zeros(n)  # Predicted estimate
    P_minus = np.zeros(n)  # Predicted estimate error covariance
    K = np.zeros(n)  # Kalman Gain
    for k in range(1, n):
        # Predict
        x_hat_minus[k] = x_hat[k - 1]
        P_minus[k] = P[k - 1] + process_variance
        # Update
        K[k] = P_minus[k] / (P_minus[k] + measurement_variance)
        x_hat[k] = x_hat_minus[k] + K[k] * (data[k] - x_hat_minus[k])
        P[k] = (1 - K[k]) * P_minus[k]
    return x_hat

def total_variation_filter(data, lambda_tv):
    n = len(data)
    u = data.copy()
    for _ in range(100):  # Perform TV denoising iteratively
        du = np.diff(u)
        u[:-1] += lambda_tv * du
        u[1:] -= lambda_tv * du
    return u

def weiner_filter(data, mysize):
    filtered_data = scipy.signal.wiener(data, mysize=mysize)
    return filtered_data

def moving_average_filter(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')

def butterworth_high_pass_filter(data, b, a):
    filtered_data = scipy.signal.lfilter(b, a, data)
    return filtered_data


def plot_data_and_filtered(ax, df, time_column, data_column, filtered_data, title):
    ax.plot(df[time_column].values, df[data_column].values, label='Original Data')
    ax.plot(df[time_column].values, np.abs(filtered_data), 'r', label='Filtered Data')
    ax.set_xlabel('Time')
    ax.set_ylabel('Data Value')
    ax.grid()
    ax.legend()
    ax.set_title(title)
    
    
def main():
    file_path = 'sv02mdpl_4_2_0_0.csv'
    df = load_data(file_path)
    fig, axs = plt.subplots(5, 2, figsize=(12, 12))
    fig.suptitle("Filtered Data Comparison")
    
    # Low-Pass Filter (FFT)
    cutoff_freq_lp_fft = 0.2
    filtered_data_lp_fft = low_pass_filter_fft(df, '542', cutoff_freq_lp_fft)
    plot_data_and_filtered(axs[0, 0], df, 'time', '542', filtered_data_lp_fft, 'Low-Pass Filtered Data (FFT)')
    
    # High-Pass Filter (FFT)
    cutoff_freq_hp_fft = 0.25
    filtered_data_hp_fft = high_pass_filter_fft(df, '542', cutoff_freq_hp_fft)
    plot_data_and_filtered(axs[0, 1], df, 'time', '542', filtered_data_hp_fft, 'High-Pass Filtered Data (FFT)')
    
    # Band-Pass Filter (FFT)
    lower_cutoff_bp_fft = 0.05
    upper_cutoff_bp_fft = 0.15
    filtered_data_bp_fft = band_pass_filter_fft(df, '542', lower_cutoff_bp_fft, upper_cutoff_bp_fft)
    plot_data_and_filtered(axs[1, 0], df, 'time', '542', filtered_data_bp_fft, 'Band-Pass Filtered Data (FFT)')
    
    # Gaussian Filter
    window_length = 3
    std_dev = 3
    filtered_data_gaussian = adaptive_filter_gaussian(df['542'].values, window_length, std_dev)
    plot_data_and_filtered(axs[1, 1], df, 'time', '542', filtered_data_gaussian, 'Gaussian Filtered Data')
    
    # Butterworth Filter
    lowcut = 0.05
    highcut = 0.25
    fs = 1.0
    filtered_data_butterworth = butterworth_filter(df['542'].values, lowcut, highcut, fs)
    plot_data_and_filtered(axs[2, 0], df, 'time', '542', filtered_data_butterworth, 'Butterworth Filtered Data')
    
    # Kalman Filter
    process_variance = 0.1
    measurement_variance = 0.2
    filtered_data_kalman = kalman_filter(df['542'].values, process_variance, measurement_variance)
    plot_data_and_filtered(axs[2,1], df, 'time', '542', filtered_data_kalman, 'Kalman Filtered Data')
    
    # Total Variation Filter
    lambda_tv = 0.1
    filtered_data_tv = total_variation_filter(df['542'].values, lambda_tv)
    plot_data_and_filtered(axs[3, 0], df, 'time', '542', filtered_data_tv, 'Total Variation Filtered Data')
    
    # Wiener Filter
    mysize = 10
    filtered_data_wiener = weiner_filter(df['542'].values, mysize)
    plot_data_and_filtered(axs[3, 1], df, 'time', '542', filtered_data_wiener, 'Wiener Filtered Data')
    
    # Moving Average Filter
    window_size = 10
    filtered_data_ma = moving_average_filter(df['542'].values, window_size)
    plot_data_and_filtered(axs[4, 0], df, 'time', '542', filtered_data_ma, 'Moving Average Filtered Data')#plt.grid()

    # Butterworth High-Pass Filter
    cutoff_frequency = 0.5
    filter_order = 4
    b, a = scipy.signal.butter(filter_order, cutoff_frequency, btype='high', analog=False)
    filtered_data_bthpf = butterworth_high_pass_filter(df['542'].values, b, a)
    plot_data_and_filtered(axs[4, 1], df, 'time', '542', filtered_data_bthpf, 'Butterworth High-Pass Filtered Data')
    plt.xlabel('Time')
    
    plt.ylabel('Data Value')
    plt.legend()
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()
    
if __name__ == "__main__":
    main()
