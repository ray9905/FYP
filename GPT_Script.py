import mne
import os
import numpy as np
import matplotlib.pyplot as plt

#Loads EEG file path
file_path = "/content/drive/MyDrive/FYP/gpt/normal_files/0000681.edf"
raw_eeg = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

#Montage assignment and channel renaming to match standard EEG system
standard_montage = mne.channels.make_standard_montage('standard_1020')
channel_renaming = {'T3':'T7','T4':'T8','T5':'P7','T6':'P8','FP1':'Fp1','FP2':'Fp2','FZ':'Fz','CZ':'Cz','PZ':'Pz'}
raw_eeg.rename_channels(channel_renaming)

#Assigns positions and warns if any channels are missing
raw_eeg.set_montage(standard_montage, on_missing='warn')

#Bandpass filtering for EEG data
raw_eeg.filter(1., 40., fir_design='firwin')

#Detects and interpolates bad channels based on variance
channel_variances = np.var(raw_eeg.get_data(), axis=1)
variance_threshold = 4 * np.median(channel_variances)
bad_channels = [raw_eeg.ch_names[i] for i, var in enumerate(channel_variances) if var > variance_threshold]
raw_eeg.info['bads'] = bad_channels
if raw_eeg.info['bads']:
    raw_eeg.interpolate_bads()

#ICA artifact removal using Picard method
ica = mne.preprocessing.ICA(n_components=15, random_state=97, method='picard', max_iter='auto')
ica.fit(raw_eeg)
eog_indices, _ = ica.find_bads_eog(raw_eeg, ch_name=['Fp1', 'Fp2'], threshold=3.0)
ica.exclude = eog_indices
ica.apply(raw_eeg)

#Creates 30 second epochs and drops bad epochs
reject_criteria = dict(eeg=80e-6)
epochs = mne.make_fixed_length_epochs(raw_eeg, duration=30, preload=True)
epochs.drop_bad(reject=reject_criteria)

save_path = '/content/drive/MyDrive/FYP/gpt/681_n_v3'
os.makedirs(save_path, exist_ok=True)

for i, epoch in enumerate(epochs):
    fig = epochs[i].plot(
        scalings='auto',
        show=False,
        n_channels=len(raw_eeg.ch_names),
        block=False,
        butterfly=False
    )
    fig.set_size_inches(24, 14)  #Sets larger figure size to reduce channel overlap
    fig.savefig(f'{save_path}/epoch_{i+1}.png', dpi=300)  #Sets higher resolution of epochs for clearer review
    plt.close(fig)

#Shows which channels are causing epoch rejection
epochs.plot_drop_log()