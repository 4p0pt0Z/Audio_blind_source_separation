import os

import torch
import torch.utils.data as torchdata

import numpy as np
import pandas as pd
import scipy

import librosa


class DCASE2013_remixed_dataset(torchdata.Dataset):
    """
    
    """

    @classmethod
    def default_config(cls):
        config = {
            # Mix files parameters
            "mix_length_s": 10,
            "sampling_rate": 16000,

            # Feature extraction parameters (log Mel spectrogram computation)
            "STFT_frame_width_ms": 64,
            "STFFT_frame_shift_ms": 32,
            "STFT_window_function": "hamming",
            "n_Mel_filters": 64,
            "Mel_min_freq": 20,
            "Mel_max_freq": 8000,

            # Train - dev - test split proportion
            "train_pct": 0.8,
            "dev_pct": 0.1,
            "test_pct": 0.1,

            # Path to the mix files folder (also include the label file)
            "data_folder": "../Datadir/remixed_DCASE2013"
        }
        return config

    @classmethod
    def split(cls, config):
        labels = pd.read_csv(os.path.join(config["data_folder"], "weak_labels.csv"))
        train_labels, dev_labels, test_labels = np.split(labels.sample(frac=1),  # random shuffle
                                                         [int(config["train_pct"] * len(labels)),
                                                          int((config["train_pct"] + config["dev_pct"]) * len(labels))])

        return cls(train_labels, config), cls(dev_labels, config), cls(test_labels, config)

    def __init__(self, files_df, config):
        self.config = config
        self.mel_filterbank = librosa.filters.mel(self.config["sampling_rate"],
                                                  n_fft=int(np.floor(self.config["STFT_frame_width_ms"]
                                                                     * self.config["sampling_rate"])),
                                                  n_mels=self.config["n_Mel_filters"],
                                                  fmin=self.config["Mel_min_freq"],
                                                  fmax=self.config["Mel_max_freq"])
        self.features = torch.from_numpy(np.asarray([self.extract_features(file)
                                                     for file in os.path.join(self.config["data_folder"],
                                                                              files_df["filename"])]))
        self.labels = torch.from_numpy(files_df.drop("filename", axis=1).values)

    def extract_features(self, filename):
        audio = librosa.core.load(filename, sr=self.config["sampling_rate"], mono=True)
        spectrogram = scipy.signal.spectrogram(audio,
                                               window=self.config["STFT_window_function"],
                                               nperseg=int(np.floor(self.config["STFT_frame_width_ms"]
                                                                    * self.config["sampling_rate"])),
                                               noverlap=int(np.floor(self.config["STFT_window_function"]
                                                                     * self.config["sampling_rate"])),
                                               detrend=False)
        mel_spectrogram = self.mel_filterbank @ spectrogram
        features = 10.0 * np.log10(mel_spectrogram)
        return np.expand_dims(features, 0)  # introduce a dimension for the 'image' channel

    def move_data_to(self, device):
        self.features.to(device)
        self.labels.to(device)

    def __getitem__(self, index):
        return self.features[index]

    def __len__(self):
        return self.features.shape[0]
