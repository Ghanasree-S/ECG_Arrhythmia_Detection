import wfdb
import matplotlib.pyplot as plt

# Load ECG signals from the MIT-BIH Arrhythmia Database
record_name = '100'  # Example record name
record_path = wfdb.io.get_record_list('mitdb')[0]
signals, fields = wfdb.rdsamp(record_path)

# Extract signal data and sampling frequency
ecg_signal = signals[:, 0]  # Use the first channel of the signal
sampling_frequency = fields['fs']

# Plot ECG signal
plt.figure(figsize=(10, 6))
plt.plot(ecg_signal, color='blue')
plt.title('ECG Signal (Record {})'.format(record_name))
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()
