import os

import concurrent.futures

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
            "feature_type": "log-mel",
            "STFT_frame_width_ms": 64,
            "STFT_frame_shift_ms": 32,
            "STFT_window_function": "hamming",
            "n_Mel_filters": 64,
            "Mel_min_freq": 20,
            "Mel_max_freq": 8000,

            # Path to the mix files folder (also include the label file)
            "data_folder": "Datadir/remixed_DCASE2013",  # to this will be appended the set folder (train-dev-val)

            "thread_max_worker": 3,
        }
        return config

    @classmethod
    def split(cls, config, which="all"):
        """
            This method instantiates 3 DCASE2013_remixed_dataset classes, for training, development and test set
            respectively. The script generating the data should take care of splitting it into 3 disjoint sets.

            The data folder for each set is updated accordingly in the passed down 'config' dict to
            point directly to the folder holding the audio data.
        Args:
            config (dict): Configuration dictionary for the data set, containing parameters for the audio processing.
            which (str): Identifier

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

        if which == "all":
            return cls(tr_labels, tr_config), cls(dev_labels, dev_config), cls(test_labels, test_config)
        elif which == "train":
            return cls(tr_labels, tr_config)
        elif which == "dev":
            return cls(dev_labels, dev_config)
        elif which == "test":
            return cls(test_labels, test_config)
        raise ValueError("ID " + which + " is not valid.")

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
        self.inverse_mel_filterbank = np.linalg.pinv(self.mel_filterbank)  # "inverse" matrix
        with concurrent.futures.ThreadPoolExecutor(max_workers=config["thread_max_worker"]) as executor:
            audios = executor.map(lambda file: self.load_audio(os.path.join(self.config["data_folder"], file)),
                                  files_df["filename"])
        self.magnitudes, self.phases = tuple(map(lambda x: np.asarray(list(x)),
                                                 zip(*[self.separated_stft(audio) for audio in audios])))
        self.features = torch.Tensor([np.expand_dims(self.stft_magnitude_to_features(stft), 0)
                                      for stft in self.magnitudes])

        self.labels = torch.from_numpy(files_df.drop("filename", axis=1).values.astype(np.float32))
        self.classes = list(files_df.columns)
        self.classes.remove("filename")
        self.filenames = files_df["filename"].tolist()

    def stft_magnitude_to_features(self, magnitude):
        mel_spectrogram = self.mel_filterbank @ magnitude
        if self.config["feature_type"] == "mel":
            return mel_spectrogram
        elif self.config["feature_type"] == "log-mel":
            with np.errstate(divide='ignore'):  # take only log of positive values, but log is computed for entire array
                log_mel_spectrogram = np.where(mel_spectrogram > 0, 10.0 * np.log10(mel_spectrogram), mel_spectrogram)
            return log_mel_spectrogram

    def separated_stft(self, audio):
        _, _, stft = scipy.signal.stft(audio,
                                       window=self.config["STFT_window_function"],
                                       nperseg=int(self.config["STFT_frame_width_ms"]
                                                   * self.config["sampling_rate"] // 1000),  # sr is per second
                                       noverlap=int(self.config["STFT_frame_shift_ms"]
                                                    * self.config["sampling_rate"] // 1000),
                                       detrend=False,
                                       boundary=None,
                                       padded=False)
        magnitude = np.abs(stft)
        phase = stft / magnitude
        return magnitude, phase

    def istft(self, ftst):
        _, istft = scipy.signal.istft(ftst,
                                      window=self.config["STFT_window_function"],
                                      nperseg=int(self.config["STFT_frame_width_ms"]
                                                  * self.config["sampling_rate"] // 1000),  # sr is per second
                                      noverlap=int(self.config["STFT_frame_shift_ms"]
                                                   * self.config["sampling_rate"] // 1000),
                                      input_onesided=True,
                                      boundary=None)
        return istft

    def load_audio(self, filename):
        audio, _ = librosa.core.load(filename, sr=self.config["sampling_rate"], mono=True)
        return audio

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

    def compute_shift_and_scaling(self):
        n_channels = self.features.shape[1]
        channel_means = [np.nan] * n_channels
        channel_std = [np.nan] * n_channels
        for i in range(self.features.shape[1]):  # average per-channel
            channel_means[i] = self.features[:, i, :, :].mean()
            channel_std[i] = self.features[:, i, :, :].std()
        return channel_means, channel_std

    def shift_and_scale(self, center, scaling):
        for i in range(self.features.shape[1]):  # average per-channel
            self.features[:, i, :, :] = (self.features[:, i, :, :] - center[i]) / scaling[i]

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return self.features.shape[0]
