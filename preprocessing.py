import mne

#Loads the eeg data from edf file
eeg_data = mne.io.read_raw_edf('', preload=True)

#displays info about the eeg data
print(eeg_data.info)
