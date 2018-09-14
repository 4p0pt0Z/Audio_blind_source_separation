import os

import torch
import torch.utils.data as torchdata

import numpy as np
import pandas as pd
import scipy

import librosa


def find_data_set_class(data_set_type):
    """
        Get the class of a model from a string identifier
    Args:
        data_set_type (str):

    Returns:
        Class implementing the desired model.
    """
    if data_set_type == "DCASE2013_remixed_data_set":
        return DCASE2013_remixed_data_set
    else:
        raise NotImplementedError("Data set type " + data_set_type + " is not available.")


class DCASE2013_remixed_data_set(torchdata.Dataset):
    """
        This class implements the audio processing to apply on the audio files remixed from the DCASE2013 data set.
        For speed, all the audio processing is done during the initialization method, then the features and labels
        are available in memory for fast access during training (and can be moved to gpu with the 'to' method).
    """

    @classmethod
    def default_config(cls):
        config = {
            # Mix files parameters
            "mix_length_s": 10,
            "sampling_rate": 16000,

            # Feature extraction parameters (log Mel spectrogram computation)
            "STFT_frame_width_ms": 64,
            "STFT_frame_shift_ms": 32,
            "STFT_window_function": "hamming",
            "n_Mel_filters": 64,
            "Mel_min_freq": 20,
            "Mel_max_freq": 8000,

            "standardize": False,

            # Path to the mix files folder (also include the label file)
            "data_folder": "Datadir/remixed_DCASE2013"  # to this will be appended the set folder (train-dev-val)
        }
        return config

    @classmethod
    def split(cls, config):
        """
            This method instantiates 3 DCASE2013_remixed_dataset classes, for training, development and test set
            respectively. The script generating the data should take care of splitting it into 3 disjoint sets.

            The data folder for each set is updated accordingly in the passed down 'config' dict to
            point directly to the folder holding the audio data.
        Args:
            config (dict): Configuration dictionary for the data set, containing parameters for the audio processing.

        Returns:
            A tuple of 3 DCASE2013_remixed_dataset: train_set, dev_set, test_set
        """
        # Update data folder to point to the train, dev or test set
        tr_config, dev_config, test_config = dict(config), dict(config), dict(config)
        tr_config["data_folder"] = os.path.join(config["data_folder"], "training")
        dev_config["data_folder"] = os.path.join(config["data_folder"], "development")
        test_config["data_folder"] = os.path.join(config["data_folder"], "validation")

        tr_labels = pd.read_csv(os.path.join(tr_config["data_folder"], "weak_labels.csv"))
        dev_labels = pd.read_csv(os.path.join(dev_config["data_folder"], "weak_labels.csv"))
        test_labels = pd.read_csv(os.path.join(test_config["data_folder"], "weak_labels.csv"))

        return cls(tr_labels, tr_config), cls(dev_labels, dev_config), cls(test_labels, test_config)

    def __init__(self, files_df, config):
        """
            Initiates the data set: - build the mel filter bank for audio processing
                                    - Load all files from disk and extract the features (Mel spectrogram)
                                    - Convert features and labels to torch.Tensor to have everything ready in memory.
        Args:
            files_df (pd.Dataframe): Dataframe obtained from reading the '.csv' file describing the labels associated
                                     with each audio file
            config (dict): Configuration dictionary containing parameters for audio features extraction.
        """
        self.config = config
        self.mel_filterbank = librosa.filters.mel(self.config["sampling_rate"],
                                                  n_fft=int(np.floor(self.config["STFT_frame_width_ms"]
                                                                     * self.config["sampling_rate"] // 1000)),
                                                  n_mels=self.config["n_Mel_filters"],
                                                  fmin=self.config["Mel_min_freq"],
                                                  fmax=self.config["Mel_max_freq"]).astype(np.float32)
        self.features = torch.from_numpy(np.asarray([self.extract_features(os.path.join(self.config["data_folder"],
                                                                                        file))
                                                     for file in files_df["filename"]]))
        if config["standardize"]:
            self.standardize()
        self.labels = torch.from_numpy(files_df.drop("filename", axis=1).values.astype(np.float32))

    def standardize(self):
        for i in range(self.features.shape[1]):  # average per-channel
            self.features[:, i, :, :] = (self.features[:, i, :, :] - self.features[:, i, :, :].mean()) \
                                        / self.features[:, i, :, :].std()

    def extract_features(self, filename):
        """
            Extract features from an audio file: compute the log Mel spectrogram.
        Args:
            filename (str): Path to the audio file to process

        Returns:
            np.ndarray with shape (1, F, T) (1 is for channel dimension in image processing) with the log - Mel scaled
            spectrogram values.
        """
        audio, _ = librosa.core.load(filename, sr=self.config["sampling_rate"], mono=True)
        _, _, spectrogram = scipy.signal.spectrogram(audio,
                                                     window=self.config["STFT_window_function"],
                                                     nperseg=int(self.config["STFT_frame_width_ms"]
                                                                 * self.config["sampling_rate"] // 1000),  # sr is per s
                                                     noverlap=int(self.config["STFT_frame_shift_ms"]
                                                                  * self.config["sampling_rate"] // 1000),
                                                     detrend=False)
        mel_spectrogram = self.mel_filterbank @ spectrogram
        with np.errstate(divide='ignore'):  # take only log of positive values, but log is computed for entire array
            features = np.where(mel_spectrogram > 0, 10.0 * np.log10(mel_spectrogram), mel_spectrogram)
        return np.expand_dims(features, 0)  # introduce a dimension for the 'image' channel

    def features_shape(self):
        return self.features[0].shape

    def to(self, device):
        """
            After this method is called, the data set should only provide batches of tensors on 'device',
            therefore in this case we move the features and label to the corresponding device.
        Args:
            device (torch.device):

        """
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return self.features.shape[0]
