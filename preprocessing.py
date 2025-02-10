import mne

#loads the eeg data from edf file
eeg_data = mne.io.read_raw_edf('', preload=True)

#displays info about the eeg data
print(eeg_data.info)

#filters relevant data between 1 and 50 Hz
eeg_data.filter(1, 50)

