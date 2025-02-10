import mne

#loads the eeg data from edf file
eeg_data = mne.io.read_raw_edf('0000003.edf', preload=True)

#displays info about the eeg data
print(eeg_data.info)

#filters relevant data between 1 and 50 Hz
eeg_data.filter(1, 50)

#applies average reference to the data
eeg_data.set_eeg_reference('average')

