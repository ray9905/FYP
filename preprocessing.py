import mne
import numpy as np

#loads the eeg data from edf file
eeg_data = mne.io.read_raw_edf('0000003.edf', preload=True)

#displays info about the eeg data
print(eeg_data.info)

#filters relevant data between 1 and 50 Hz
eeg_data.filter(1, 50)

#applies average reference to the data
eeg_data.set_eeg_reference('average')

#converts data from vaults to microvolts
eeg_array = eeg_data.get_data() *1000000

#swaps axes to samples, channels
eeg_array = eeg_array.transpose(1,0)

print(eeg_array)
print(f"Minimum: {np.min(eeg_array)} , Maximum: {np.max(eeg_array)}")

#seconds
window_duration = 1

#sampling frequency
frequency = eeg_data.info['sfreq'] 

#number of samples in a window
window_samples = int(window_duration * frequency)

#creates segments of the data
eeg_segments = np.array([eeg_array[i:i+window_samples] for i in range(0, eeg_array.shape[0], window_samples)])



