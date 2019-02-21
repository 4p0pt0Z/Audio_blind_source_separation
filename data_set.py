import os
from abc import abstractmethod
import h5py
import yaml
import json

import concurrent.futures

import torch
import torch.utils.data as torchdata

import numpy as np
import pandas as pd
import scipy

import librosa
from pcen import no_arti_pcen


def find_data_set_class(data_set_type):
    r"""Get the class of a data set from a string identifier

        The same training framework can be used for training models using different Data set. The audio features
        extraction is implemented in an abstract class - AudioDataset - as these methods are common to all audio
        processing. This class inherits from torch.data.Dataset that provides convenient functions for iterating over
        the set during training. The handling of the audio data format, label format... and any data set specific
        processing is implemented in specialized sub-class of AudioDataset. This method provides a way for the user
        to get this class from a command line argument.

    Args:
        data_set_type (str): Identifier of the Dataset class

    Returns:
        Class implementing the desired model.
    Examples:
        >>> config = find_data_set_class("AudiosetSegments").default_config()
    """

    if data_set_type == "DCASE2013RemixedDataSet":
        return DCASE2013RemixedDataSet
    elif data_set_type == "ICASSP2018JointSeparationClassificationDataSet":
        return ICASSP2018JointSeparationClassificationDataSet
    elif data_set_type == "AudiosetSegments":
        return AudiosetSegments
    else:
        raise NotImplementedError("Data set type " + data_set_type + " is not available.")


class AudioDataSet(torchdata.Dataset):
    r"""This class implements the common audio processing functions that are used to load an audio .wav file and
        extract features from it for the training.

        This class implements two functionality:
            - common interface for audio features extraction from audio data
              Also include inverse processing: from audio features back to audio waveform
            - common interface for data loading during training. (inherits torch.data.Dataset)
    """

    @classmethod
    def default_config(cls):
        r"""Get the available audio features parameters

        Returns:
            Dictionary with audio feature extraction parameters
        """

        config = {
            # Audio mixture files parameters
            "sampling_rate": 0,

            # Audio features: "spectrogram", "mel", "log-mel", "log-mel_no_shift", "pcen". see
            # stft_magnitude_to_features
            "feature_type": "log-mel",

            # spectrogram parameters
            "STFT_frame_width_ms": 0,  # Amount of audio to include in the fft during the stft in milli-seconds.
            "STFT_frame_shift_ms": 0,  # Amount of time shift between 2 fft during the stft in milli-seconds
            "STFT_window_function": "hamming",  # Window function to use in the stft

            # Mel scaling parameters
            "n_Mel_filters": 0,  # Number of mel filters
            "Mel_min_freq": 0,  # Minimal frequency of the Mel scale
            "Mel_max_freq": 0,  # Maximal frequency of the Mel scale (should not be superior to sampling_rate / 2)

            # PCEN can be performed by the torch.Dataset with fixed parameters or it can be used as a network layer (
            # with trainable parameters) in which case it is a part of the separation_model and "mel" features should
            # be used.
            # pcen parameters
            # (default values from Yuxian Wang et al. "Trainable Frontend For Robust and Far-Field Keyword Spotting")
            "pcen_s": 0.04,  # low-pass filter parameter
            "pcen_eps": 1e-6,  # epsilon (used for numerical stability of normalization)
            "pcen_alpha": 0.98,  # alpha (power of normalization term)
            "pcen_delta": 2.0,  # delta (shift before power transform)
            "pcen_r": 0.5,  # power of the power function (dynamic range compression)

            "data_folder": "Datadir/",  # Path to the folder containing the audio data. Better to pass absolute path.

            # type of feature normalization: "min-max scaling", "standardization" see shift_and_scale()
            # Used for scaling and centering the audio features using data set statistics during training
            "scaling_type": "standardization"
        }
        return config

    def __init__(self, config):
        r"""Defines API for audio variables for the Dataset classes.

        Args:
            config (dict): Dictionary with audio features parameters. Saved as class variable.
        """

        super(AudioDataSet, self).__init__()

        self.config = config
        # For all feature types except "spectrogram", we will use Mel scaling (and inverse for reconstruction ?)
        if config['feature_type'] != "spectrogram":
            self.mel_filterbank = librosa.filters.mel(self.config["sampling_rate"],
                                                      n_fft=int(np.floor(self.config["STFT_frame_width_ms"]
                                                                         * self.config["sampling_rate"] // 1000)),
                                                      n_mels=self.config["n_Mel_filters"],
                                                      fmin=self.config["Mel_min_freq"],
                                                      fmax=self.config["Mel_max_freq"]).astype(np.float32)
            self.inverse_mel_filterbank = np.linalg.pinv(self.mel_filterbank)

        # Define general names for audio features and labels. Feature extraction is implemented in this class,
        # but data loading from disk and label formating is specific to the sub-classes, therefore these variables
        # are populated by the sub-classes.
        self.features = None
        self.labels = None
        self.magnitudes = None
        self.phases = None

    @classmethod
    @abstractmethod
    def split(cls, config, which="all"):
        r"""Build training, testing and validation sets.

            Implemented in sub-classes.

        Args:
            config (dict): dictionary with audio feature parameters required for building the data sets.
            which (str): optional: to build only 1 of the 3 set partitions (training, testing or validation)

        Returns:
            (training_set, testing_set, validation_set): tuple of 3 data set.

        Examples:
            >>> train_set, test_set, val_set = find_data_set_class("data_set_type").split(config)
            >>> val_set = find_data_set_class("data_set_type").split(config, which='val')
        """

        pass

    def features_shape(self):
        r"""Get the pytorch shape of a training example features: [Channels, Frequency, Time]

        Returns:
            Shape of a training example.
        """

        return tuple(self.features[0].shape)

    def n_classes(self):
        r"""Get the number of classes in the data set.

        Returns:
            Number of labelled classes in the data set.
        """

        return self.labels.shape[1]  # labels.shape: [number of training samples, number of classes]

    def to(self, device):
        r"""Moves the Dataset to a pytorch device (cpu or gpu)

            After this method is called, the Dataset __getitem__ method, which is used to iterate over the data set,
            is expected to return torch.tensors that are on 'device'.

        Args:
            device (torch.device): Move the data set to this device
        """

        # Data sets used so far fit in gpu memory. We can move all the features and labels for speed.
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)

    def compute_shift_and_scaling(self):
        r"""Calculate data set statistics require for features centering and scaling .

            3 modes are supported:
                - standardization: center at 0, scale to have unit variance
                - min-max: scale to [0, 1] using min and max values of data set.
                - none: leave the data set unchanged.

        Returns:
            (Shift, scaling): values of shift and scaling to use for centering and scaling the data set.
        """

        n_channels = self.features.shape[1]
        channel_shift = [np.nan] * n_channels
        channel_scaling = [np.nan] * n_channels
        for i in range(self.features.shape[1]):
            if self.config["scaling_type"] == "standardization":
                channel_shift[i] = self.features[:, i, :, :].mean()
                channel_scaling[i] = self.features[:, i, :, :].std()
            elif self.config["scaling_type"] == "min-max":
                channel_shift[i] = self.features[:, i, :, :].min()
                channel_scaling[i] = self.features[:, i, :, :].max() - channel_shift[i]  # max - min
            elif self.config["scaling_type"].lower() == "none":
                print("[WARNING] No normalization procedure is used !")
                channel_shift[i] = 0.0
                channel_scaling[i] = 1.0

        return channel_shift, channel_scaling

    def shift_and_scale(self, shift, scaling):
        r"""Shift and scale the Dataset audio features with the provided values. Inplace.

        Args:
            shift (list): shift to use for shifting to [0, 1] or centering, for each channel of the features
            scaling (list): scaling factor, for each channel of the features.
        """

        for i in range(self.features.shape[1]):  # per channel
            self.features[:, i, :, :] = (self.features[:, i, :, :] - shift[i]) / scaling[i]

    def un_shift_and_scale(self, shift, scaling):
        r"""Invert the shift and scaling. Inplace.

            Dataset.shift_and_scale(shift, scaling).un_shift_and_scale(shift, scaling)
            has the same features than Dataset.

        Args:
            shift (list): shift value to revert, for each feature channel
            scaling (list): scaling factor to inverse, for each feature channel.

        """

        for i in range(self.features.shape[1]):
            self.features[:, i, :, :] = (self.features[:, i, :, :] * scaling[i]) + shift[i]

    def rescale_to_initial(self, features, shift, scaling):
        r"""Invert the scaling and shifting of the input 'features'. Inplace.

            In spirit, this is features.un_shift_and_scale(shift, scaling), but un_shift_and_scale is operating on
            entire Dataset while this operates on a batch of features.

        Args:
            features (torch.tensor): audio features to un-scale and un-shift. shape: [Batch, Channel, Frequency, Time]
            shift (list): shift value to revert, for each feature channel
            scaling (list): scaling factor to inverse, for each feature channel.

        """

        for i in range(features.shape[1]):
            features[:, i, :, :] = (features[:, i, :, :] * scaling[i].to(features[:, i, :, :].device)) \
                                   + shift[i].to(features[:, i, :, :].device)

    def stft_magnitude_to_features(self, magnitude=None, mel_spectrogram=None):
        r"""Calculate audio features from input.

            To process audio data using convolutional neural network, it is common to work in Time x Frequency
            representation. This is accomplished by computing the magnitude of the short-time Fourier transform of
            the audio waveform.
            However, it is also common to process further the magnitude for several reasons:
                - Replicate human perception of frequencies. Human perception of frequency is not linear,
                therefore the frequency axis of the stft output (which is linear in frequency) can be re-scaled to
                the Mel scale. https://en.wikipedia.org/wiki/Mel_scale
                - dynamic range compression (DRC). Optimization algorithms such as gradient descent work better when the
                input range is in the same order of magnitude. This is often performed by taking the log of the features
                - stationary noise cancellation. The 2 above operation have been combined with a filtering operation
                that removes stationary background noise in the Per-Channel Energy Normalization (PCEN) processing.

            This function supports 4 features types:
                - "spectrogram": Use as features the un-processed magnitude of the STFT.
                - "mel": Use as features the Mel-scaled spectrogram.
                - "log-mel": Use as features log(Mel-spectrogram + 1.0)
                             +1.0 is used to make sure that the log transform is contracting. (-> DRC)
                - "log-mel_no_shift": Use as features log(Mel-spectrogram)
                             Same as "log-mel", but with-out "+1.0". Still uses a small shift to avoid log(0.0)
                - "pcen": PCEN processing with fixed parameters. It is applied on the Mel-scaled spectrogram.

            To increase loading speed, for some data sets the mel scale spectrogram have already been computed,
            therefore we can skip this step in this function.

        Args:
            magnitude (np.ndarray): Magnitude of the stft of an audio example. shape: [Frequency, Time]
            mel_spectrogram (np.ndarray): Mel scaled magnitude of the stft of an audio example.
                                          shape: [Mel Frequency, Time]

        Returns:
            Processed features from the input magnitude or mel-spectrogram.
        """

        if self.config["feature_type"].lower() == "spectrogram":
            return magnitude

        # If we are not provided the mel-scaled spectrogram, compute it.
        if mel_spectrogram is None:
            mel_spectrogram = self.mel_filterbank @ magnitude  # Apply the mel filters on the input spectrogram

        if self.config["feature_type"].lower() == "mel":
            return mel_spectrogram

        elif self.config["feature_type"].lower() == "log-mel":
            log_mel_spectrogram = 10.0 * np.log1p(mel_spectrogram) / np.log(10.0)  # +1 to make sure log is contracting
            return log_mel_spectrogram

        elif self.config["feature_type"].lower() == "log-mel_no_shift":
            return np.log(mel_spectrogram + 1e-15)  # avoir log(0.0)

        elif self.config["feature_type"].lower() == "pcen":
            # Use the PCEN with forward-backward filtering to avoid having artifacts due to the filter initialization
            return no_arti_pcen(mel_spectrogram, sr=self.config["sampling_rate"],
                                hop_length=int(np.floor(self.config["STFT_frame_shift_ms"]
                                                        * self.config["sampling_rate"] // 1000)),
                                gain=self.config["pcen_alpha"], bias=self.config["pcen_delta"],
                                power=self.config["pcen_r"], b=self.config["pcen_s"], eps=self.config["pcen_eps"])

    def features_to_stft_magnitudes(self, features, features_idx):
        r"""Computes the STFT magnitude corresponding to the input batch of features

            For audio re-synthesis after the processing of an audio example by a neural network, it might be required
            to apply the inverse processing than what was used to map STFT magnitudes to features, in order this time
            to go from features to STFT magnitudes.

            "features_to_stft_magnitudes(stft_magnitude_to_features(magnitude)) = magnitude"

        Args:
            features (np.ndarray): batch of features. shape: [Batch, Channel, (Mel-)Frequency, Time]
            features_idx (int): index in the Dataset of the processed features. Used to get the initial magnitude
            corresponding to the processed input features, if the conversion to STFT magnitude requires it.

        Returns:
            STFT magnitude representation corresponding to the input features.
        """

        if self.config["feature_type"] == "spectrogram":
            return features
        if self.config["feature_type"] == "log-mel":
            features = np.power(10.0 * np.ones(features.shape), (features / 10.0)) - 1.0
            return self.inverse_mel_filterbank[np.newaxis, np.newaxis, ...] @ features
        elif self.config["feature_type"] == "mel":  # Use pseudo-inverse of Mel filterbank matrix
            return self.inverse_mel_filterbank[np.newaxis, np.newaxis, ...] @ features
        elif self.config["feature_type"] == "pcen":
            # Inversing the PCEN is tricky (and probably not advisable)
            # WARNING: This implementation is mostly un-tested.
            # To do so, we have to compute the filter M used for stationary background removal, and then invert it.
            # Get the filter
            M = scipy.signal.filtfilt([self.config["pcen_s"]], [1, self.config["pcen_s"] - 1],
                                      self.mel_filterbank @ self.magnitudes[features_idx], axis=-1, padtype=None)
            M = np.exp(-self.config["pcen_alpha"]
                       * (np.log(self.config["pcen_eps"]) + np.log1p(M / self.config["pcen_eps"])))
            # Inverse PCEN normalization and dynamic range compression
            return (np.power(features + self.config["pcen_delta"] ** self.config["pcen_r"], 1.0 / self.config["pcen_r"])
                    - self.config["pcen_delta"]) / M

    def separated_stft(self, audio):
        r"""Compute the short-time Fourier transform of the audio waveform, and separate magnitude from phase.

            The short-time Fourier transform has mainly 3 parameters to tune that define the resolution in frequency
            and time:
                - Number of audio samples in a fft frame: nperseg
                - Number of samples in a fft segment: nfft (nfft = nperseg + zero-padding)
                  This is the actual number of samples in a fft frame, containing audio samples and padding samples.
                - Number of overlapping samples between 2 consecutive fft frames: noverlap
                  In order to be able to perform inverse STFT, we need noverlap >= nperseg // 2

            In librosa implementation of the STFT, nperseg and nfft are forced to have the same value (ie no padding
            is done on a fft frame). The same convention is used here.
            noverlap is related to librosa's hop_length by: noverlap = nfft - hop_length

            It is easier for the user to define the length of the fft frame in milli-seconds rather than in number of
            samples (the two are related by the sampling rate). The conversion is done here.
        Args:
            audio (np.ndarray): audio waveform. shape: [number of samples]

        Returns:
            (magnitude, phase): magnitude and phase component of the STFT of audio.
        """

        _, _, stft = scipy.signal.stft(audio,
                                       window=self.config["STFT_window_function"],
                                       nperseg=int(self.config["STFT_frame_width_ms"]
                                                   * self.config["sampling_rate"] // 1000),  # sr is per second
                                       noverlap=int(self.config["STFT_frame_shift_ms"]
                                                    * self.config["sampling_rate"] // 1000),
                                       detrend=False,
                                       boundary=None,  # No padding at boundary of 'audio' to center the first fft frame
                                       padded=False)  # The last incomplete fft frame is discarded
        magnitude = np.abs(stft)  # Energy spectrogram: magnitude of the stft.
        phase = np.exp(1.j * np.angle(stft))
        return magnitude, phase

    def istft(self, stft):
        r"""Compute inverse short-time Fourier transform: reverse operation that the STFT computed in separated_stft()

        Args:
            stft (np.ndarray): STFT array (complex). shape: [Frequency, Time]

        Returns:
            np.ndarray. Audio waveform
        """

        _, istft = scipy.signal.istft(stft,
                                      window=self.config["STFT_window_function"],
                                      nperseg=int(self.config["STFT_frame_width_ms"]
                                                  * self.config["sampling_rate"] // 1000),  # sr is per second
                                      noverlap=int(self.config["STFT_frame_shift_ms"]
                                                   * self.config["sampling_rate"] // 1000),
                                      input_onesided=True,
                                      boundary=None)
        return istft

    def load_audio(self, filename):
        r"""Load an audio waveform from .wav file into memory.

            Re-sampling is performed if the sampling rate of the file differs from the Dataset sampling rate.
            Stereo file are converted to mono-channel.

        Args:
            filename (str): Path to the .wav file.

        Returns:
            np.ndarray. Aduio waveform
        """

        audio, _ = librosa.core.load(filename, sr=self.config["sampling_rate"], mono=True)
        return audio

    @abstractmethod
    def audio_full_filename(self, filename_basename):
        r"""Get the path to an audio file, given its relative path in the data_folder.

        Args:
            filename_basename (str): Path of the audio file in the data folder.

        Returns:
            Path to an audio file.
        """

        pass


class DCASE2013RemixedDataSet(AudioDataSet):
    r"""This class implements the audio processing to apply on the audio files remixed from the DCASE2013 data set.
        (Subtask 1 of Event Detection problem at: http://c4dm.eecs.qmul.ac.uk/sceneseventschallenge/description.html)

        The data set consists of 16 classes of office noises:
        alert clearthroat cough doorslam drawer keyboard keys knock laughter mouse pageturn pendrop phone printer speech
        switch

        These events have been mixed together with white noise background to produce training mixtures. The mixing
        step is performed by the generate_weakly_labelled_audio_mixtures_from_DCASE2013.py script

        For speed, all the audio processing is done during the initialization method, then the features and labels
        are available in memory for fast access during training (and can be moved to gpu with the 'to' method).

        Once the set has been build, it can be saved to .hdf5 using save_to_hdf5(). Then it is possible to load it
        more quickly using the 'data_set_load_folder_path' argument (used in build_from_hdf5())
    """

    @classmethod
    def default_config(cls):
        r"""Update the parameters for audio processing with adequate values for this data set.

        Returns:
            Configuration dictionary with adequate audio processing values for this data set.
        """

        config = super(DCASE2013RemixedDataSet, cls).default_config()
        config.update({
            # Mix files parameters
            "sampling_rate": 16000,

            # Feature extraction parameters (log Mel spectrogram computation)
            "feature_type": "log-mel",
            "STFT_frame_width_ms": 64,
            "STFT_frame_shift_ms": 32,
            "STFT_window_function": "hamming",
            "n_Mel_filters": 64,
            "Mel_min_freq": 0,
            "Mel_max_freq": 8000,

            "scaling_type": "standardization",  # type of feature normalization: "min-max scaling", "standardization"

            # Path to the audio mixture files folder (also include the label file) (needed if building from audio files)
            "data_folder": "Datadir/remixed_DCASE2013",  # to this will be appended the set folder (train-test-val)

            # Categories for regrouping the classes together: list of categories, each category is indicated as a string
            # of the classes in the category separated by dots.
            # eg: --class_categories speech.laughter clearthroat.cough doorslam.drawer.keys.knock.pendrop.switch \
            # keyboard.mouse phone.alert pageturn printer
            "class_categories": ['all_separated'],

            "data_set_save_folder_path": "",  # Path to .hdf5 file to save the computed features.
            "data_set_load_folder_path": "",  # Path to .hdf5 file to load the pre-computed features.
            "thread_max_worker": 3,  # Number of thread for loading the audio data (if building from audio files)
        })
        return config

    @classmethod
    def split(cls, config, which="all"):
        r"""This method instantiates 3 or 1 DCASE2013_remixed_dataset, for training, testing and validation set
            respectively. The script generating the audio mixtures should take care of splitting it into 3 disjoint sets
            see generate_weakly_labelled_audio_mixtures_from_DCASE2013.py

            The data folder for each set is updated accordingly in the passed down 'config' dict to
            point directly to the folder holding the audio data. (ie: to data_folder is appended 'train' or 'test'...)
        Args:
            config (dict): Configuration dictionary for the data set, containing parameters for the audio processing.
            which (str): to chose which set to return: either "all" (default), 'train', 'test' or 'val'

        Returns:
            Either one or a tuple of 3 DCASE2013RemixedDataSet.
        """

        # Update data folder to point to the train, dev or test set
        tr_config, dev_config, test_config = dict(config), dict(config), dict(config)
        tr_config["data_folder"] = os.path.join(config["data_folder"], "training")
        dev_config["data_folder"] = os.path.join(config["data_folder"], "development")
        test_config["data_folder"] = os.path.join(config["data_folder"], "validation")

        if which == "all":
            return cls(tr_config), cls(dev_config), cls(test_config)
        elif which == "train":
            return cls(tr_config)
        elif which == "test":
            return cls(dev_config)
        elif which == "val":
            return cls(test_config)
        raise ValueError("ID " + which + " is not valid.")

    def __init__(self, config):
        r"""Constructor. Build the dataset either from the audio mixtures files (and labels) or by loading
            pre-computed features from .hdf5 file.

            First: try to load a pre-computed data set from .hdf5 file (build_from_hdf5())
            Note: if the dataset is loaded from hdf5, the new audio processing parameters are not used !
            If this failed: compute the audio features from the audio mixture files directly (build_from_audio_files())

            If requested: save the computed features to .hdf5

        Args:
            config (dict): Configuration dictionary containing parameters for audio features extraction.
        """

        super(DCASE2013RemixedDataSet, self).__init__(config)

        try:  # Try to load features from hdf5
            self.magnitudes, self.phases, self.features, self.labels, self.classes, self.filenames = \
                self.build_from_hdf5()
        except ValueError as e:  # if failed: compute the features from the audio mixtures files.
            print(e)
            print("Building data set from audio files !")
            # Read the .csv file with the labels.
            # csv shape: filename, class1, class2, class3, class4 ...
            files_df = pd.read_csv(os.path.join(config["data_folder"], "weak_labels.csv"))
            files_df = files_df.reindex(sorted(files_df.columns), axis=1)  # sort columns
            # build the audio mixtures
            self.magnitudes, self.phases, self.features, self.labels, self.classes, self.filenames = \
                self.build_from_audio_files(files_df)

        # save to hdf5
        if self.config["data_set_save_folder_path"]:
            self.save_to_hdf5()

    def build_from_audio_files(self, files_df):
        r"""Compute audio features from audio files.

            Loading the audio files is slow (especially with HDD), therefore a parallel execution is implemented in
            threads to try and speed things up.

            The magnitude, phase and features are computed using the methods of the super class.

            If requested by user - argument "class_categories" - the classes can be grouped together in categories.
            The labels for these categories will be merged.

            The features and labels are converted to pytorch tensors. Magnitude and phase are left as numpy arrays.

        Args:
            files_df (pd.DataFrame): DataFrame build from the label csv file. Contains the mixture filenames and
                                     associated labels.
                                     DataFrame structure: filename | class1 | class2 | class3 | ...
                                                          1234.wav     0         0        1     ...
        Returns:
            np.ndarray, np.ndarray, torch.tensor, torch.tensor, list, list
            Computed Magnitude, phases, features, labels for the audio mixtures in the data folder.
            Also return the classes corresponding to the labels.
        """

        # Load all audio mixture to RAM
        # audios is a list of np.ndarray containing the audio data.
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config["thread_max_worker"]) as executor:
            audios = executor.map(lambda file: self.load_audio(os.path.join(self.config["data_folder"], file)),
                                  files_df["filename"])
        # Compute STFT for all mixtures
        # magnitudes and phase are np.ndarray with shape [number mixtures, Frequency, Time]
        magnitudes, phases = tuple(map(lambda x: np.asarray(list(x)),
                                       zip(*[self.separated_stft(audio) for audio in audios])))
        # Compute audio features from the STFT magnitudes.
        # Also add a dimension for the "channel"
        # features is a torch.tensor with shape [Number mixtures, Channel=1, Frequency, Time]
        features = torch.Tensor([np.expand_dims(self.stft_magnitude_to_features(mag), 0)
                                 for mag in magnitudes])

        # Group the labels per categories
        if self.config["class_categories"] != 'all_separated':
            category_df = pd.DataFrame()
            # Assign label 1 to a category if any of the labels of the class in the category is present
            for category in self.config["class_categories"]:
                category_df[category] = files_df.drop("filename", axis=1).apply(
                    lambda row: any([row[event] for event in category.split('.')]), axis=1)
            labels = torch.from_numpy(category_df.values.astype(np.float32))
            classes = category_df.columns  # The classes associated to the labels are now the categories
        # if no categories is asked: simply load the labels.
        else:
            labels = torch.from_numpy(files_df.drop("filename", axis=1).values.astype(np.float32))
            classes = list(files_df.columns)
            classes.remove("filename")

        filenames = files_df["filename"].tolist()
        return magnitudes, phases, features, labels, classes, filenames

    def build_from_hdf5(self):
        r"""Load pre-comupted features from .hdf5

            To avoid re-computing the audio features for each training, they have been saved to .hdf5.
            This method implements how to read them back.

            WARNING: The parameters used when computing the features are not saved ! Be careful to use the same
            parameters for the audio processing than when the set has been computed and saved !

            If the .hdf5 could not be parsed (either corrupted file  or the path to the file is incorrect) then the
            method raises an Exception.

        Returns:
            np.ndarray, np.ndarray, torch.tensor, torch.tensor, list, list
            Computed Magnitude, phases, features, labels for the audio mixtures in the data folder.
            Also return the classes corresponding to the labels.
        """

        # Add 'train', 'test' or 'val' to data_set_load_folder_path to use the right set.
        path = os.path.join(self.config["data_set_load_folder_path"],
                            os.path.basename(self.config["data_folder"])) + '.h5'
        try:
            with h5py.File(path, 'r') as hf:
                magnitudes = np.array(hf.get('magnitudes'))
                phases = np.array(hf.get('phases'))
                features = torch.from_numpy(np.array(hf.get('features')))
                labels = torch.from_numpy(np.array(hf.get('labels')))
                classes = [s.decode('utf-8') for s in hf.get('classes')]
                filenames = [s.decode('utf-8') for s in hf.get('filenames')]
                return magnitudes, phases, features, labels, classes, filenames
        except OSError:
            raise ValueError("Can not load data set from file " + path)

    def save_to_hdf5(self):
        r"""Save the computed audio features to .hdf5

            Creates a folder at data_set_save_folder_path. Then save the features as 'train.h5', 'test.h5' or
            'val.h5' according to the type of data set.
        """

        # Create folder if needed
        if not os.path.exists(self.config["data_set_save_folder_path"]):
            os.makedirs(self.config["data_set_save_folder_path"])
        # Add 'train', 'test' or 'val' to filename.
        path = os.path.join(self.config["data_set_save_folder_path"],
                            os.path.basename(self.config["data_folder"])) + '.h5'
        # Write features
        with h5py.File(path, 'w') as hf:
            hf.create_dataset('magnitudes', data=self.magnitudes)
            hf.create_dataset('phases', data=self.phases)
            hf.create_dataset('features', data=self.features.numpy())
            hf.create_dataset('labels', data=self.labels.numpy())
            hf.create_dataset('classes', data=np.array(self.classes, dtype='S'))
            hf.create_dataset('filenames', data=np.array(self.filenames, dtype='S'))

    def audio_full_filename(self, filename):
        r"""Get the full path to an audio mixture.

        Args:
            filename (str): Path to an audio mixture, relatively to the data_folder

        Returns:
            Path to an audio mixture
        """

        return os.path.join(self.config["data_folder"], filename)

    def load_audio_source_files(self, idx):
        r"""Load the sources used to create the audio mixture with index 'idx' of the data set.

            This data set is composed of audio mixtures artificially created. The sources used to create the mixtures
            are saved along with the mixtures, so that the separation performances of a model can be measured.
            This method implements the loading of these sources, given the index of a mixture in the set.

        Args:
            idx (int): Index of an audio mixture in the set.

        Returns:
            np.ndarray: shape: [Number of classes (or categories), audio length]
                               Each element of the first dimension is a source used to create the audio mixture 'idx'.
                               Note: Most of the sources will be empty (all-zero)
        """

        # Path to folder containing the sources
        source_files_dir = os.path.join(self.config["data_folder"],
                                        os.path.splitext(self.filenames[idx])[0])
        # Load the sources in alphabetical order
        source_files = sorted(os.listdir(source_files_dir))
        reference_sources = np.asarray([self.load_audio(os.path.join(source_files_dir, filename))
                                        for filename in source_files])
        # Group the sources into category if required
        if self.config["class_categories"] != self.default_config()["class_categories"]:
            reference_category_sources = []
            for category in self.config["class_categories"]:
                source_rows = [idx for idx, source in enumerate(source_files)
                               if any(event in source for event in category.split('.'))]
                reference_category_sources.append(np.sum(reference_sources[source_rows, :], axis=0))
            return np.asarray(reference_category_sources)
        else:
            return reference_sources

    def __getitem__(self, index):
        r"""Torch.data.Dataset method for getting an item in the data set.

            This method needs to be implemented to be able to iterate over the dataset

        Args:
            index (int): Index of an audio example

        Returns:
            (audio features, labels) of this example
        """

        return self.features[index], self.labels[index]

    def __len__(self):
        r"""
        Returns:
            Length of the Dataset (number of audio examples in the data set).
        """

        return self.features.shape[0]


class ICASSP2018JointSeparationClassificationDataSet(AudioDataSet):
    r"""This class implements the audio processing to apply on the audio mixtures build from the TUT Rare sound
        events 2017

        The data set consists of 3 classes of rare events:
        babycry glassbreak gunshot

        These events have been mixed together with background scene to create audio mixtures. In order to compare the
        results of a model with the paper
        Qiuqiang Kong et al. "A joint separation-classification model for sound event detection of weakly labelled
        data". In: CoRR abs/1711.03037 (2017). arXiv:1711.03037. URL: http://arxiv.org/abs/1711.03037
        the features computed by the authors have been used. The code of the authors is available at
        https://github.com/qiuqiangkong/ICASSP2018_joint_separation_classification
        Running the runme.sh script generates 2 data sets: a training and a testing set. The audio mixtures of each
        set, as well as the sources used in each mixtures, are saved. The audio features and labels are also computed
        and saved to .hdf5 file.
        This implementations loads the audio sources to be able to compare the separation results of a model with the
        initial sources. The audio features are re-computed to make sure the inversion step is handle properly.

        The feature computation (from STFT magnitude) and reverse process is re-implemented in this class (not using
        the super class methods) in order to replicate as much as possible the processing made by the reference paper.
    """

    @classmethod
    def default_config(cls):
        r"""Update the parameters for audio processing with adequate values for this data set.

        Returns:
            Configuration dictionary with adequate audio processing values for this data set.
        """

        config = super(ICASSP2018JointSeparationClassificationDataSet, cls).default_config()
        config.update({
            # Mix files parameters
            "sampling_rate": 16000,

            # Feature extraction parameters (log Mel spectrogram computation)
            "feature_type": "log-mel",
            "STFT_frame_width_ms": 64,
            "STFT_frame_shift_ms": 32,
            "STFT_window_function": "hamming",
            "n_Mel_filters": 64,
            "Mel_min_freq": 0,
            "Mel_max_freq": 8000,

            # pcen parameters  # PCEN can be performed by the torch.Dataset with fixed parameters
            "pcen_s": 0.04,  # or it can be used as a network layer (with trainable parameters)
            "pcen_eps": 1e-6,
            "pcen_alpha": 0.98,
            "pcen_delta": 2.0,
            "pcen_r": 0.5,

            # Path to the folder containing the features (hdf5 files)
            "data_folder": "../ICASSP2018_joint_separation_classification/packed_features/logmel/",

            # Path to the folder containing the audio mixes and groundtruths
            "audio_folder": "../ICASSP2018_joint_separation_classification/mixed_audio",

            # Path to yaml file describing the mixes composition
            "yaml_file": "../ICASSP2018_joint_separation_classification/mixed_yaml",  # training/testing is appended

            "thread_max_worker": 3,

            "scaling_type": "standardization"  # type of feature normalization: "min-max scaling", "standardization"
        })
        return config

    def __init__(self, config):
        r"""Constructor. Build the dataset by loading the audio mixture and sources from audio files,
        and the labels and filenames from .hdf5 file.

        Args:
            config (dict): Configuration dictionary containing parameters for audio features extraction.
        """

        super(ICASSP2018JointSeparationClassificationDataSet, self).__init__(config)

        self.config = config

        # Load the labels and filenames.
        with h5py.File(config["data_file"], 'r') as hf:
            self.filenames = [file.decode() for file in list(hf.get('na_list'))]
            # self.features = torch.from_numpy(np.array(hf.get('x'))).unsqueeze(1).permute(0, 1, 3, 2)
            self.labels = torch.from_numpy(np.array(hf.get('y')))

        # Get the indices of mixtures, events and background .wav files.
        self.mix_idx = [idx for idx, filename in enumerate(self.filenames) if 'mix' in filename]
        self.event_idx = [idx for idx, filename in enumerate(self.filenames) if 'event' in filename]
        self.background_idx = [idx for idx, filename in enumerate(self.filenames) if 'bg' in filename]
        with open(config["yaml_file"], 'r') as yaml_stream:
            self.yaml = yaml.safe_load(yaml_stream)

        # Load the audio waveform
        with concurrent.futures.ThreadPoolExecutor(max_workers=config["thread_max_worker"]) as executor:
            audios = executor.map(lambda file: self.load_audio(os.path.join(self.config["audio_folder"], file)),
                                  self.filenames)
        self.magnitudes, self.phases = tuple(map(lambda x: np.asarray(list(x)),
                                                 zip(*[self.separated_stft(audio) for audio in audios])))
        self.features = torch.Tensor([np.expand_dims(self.stft_magnitude_to_features(mag), 0)
                                      for mag in self.magnitudes])
        self.classes = ['babycry', 'glassbreak', 'gunshot', 'background']

    @classmethod
    def split(cls, config, which="all"):
        r"""This method instantiates 3 or 1 ICASSP2018JointSeparationClassificationDataSet for training and
            validation respectively.

            The mixtures from different authors are used, and they only splitted the data into training and
            validation set. Therefore the testing and validation set will be the same for this set.

            The values in 'config' will be updated for each set to point the the right audio folders and feature file.

        Args:
            config (dict): Configuration dictionary for the data set, containing parameters for the audio processing.
            which (str): To chose which set to return: either "all" (default), 'train', 'test' or 'val'

        Returns:
            3 or 1 ICASSP2018JointSeparationClassificationDataSet.
        """

        tr_config, test_config = dict(config), dict(config)
        tr_config["data_file"] = os.path.join(config["data_folder"], "training.h5")
        tr_config["audio_folder"] = os.path.join(config["audio_folder"], "training")
        tr_config["yaml_file"] = os.path.join(config["yaml_file"], "training.csv")
        test_config["data_file"] = os.path.join(config["data_folder"], "testing.h5")
        test_config["audio_folder"] = os.path.join(config["audio_folder"], "testing")
        test_config["yaml_file"] = os.path.join(config["yaml_file"], "testing.csv")

        if which == "all":
            return cls(tr_config), cls(test_config), cls(test_config)
        elif which == "train":
            return cls(tr_config)
        elif which == "test" or which == "val":
            print("WARNING: Testing and Validation set are the same for this set !")
            return cls(test_config)
        else:
            raise ValueError("Set identifier " + which + " is not available.")

    def stft_magnitude_to_features(self, magnitude):
        r"""Transform from STFT magnitude to audio features used in reference paper (log Mel spectrogram)

        Args:
            magnitude (np.ndarray): magnitude of STFT of audio file

        Returns:
            Log Mel spectrogram
        """

        return np.log(self.mel_filterbank @ magnitude + 1e-8)

    def features_to_stft_magnitudes(self, features, _):
        r"""Transform a log Mel spectrogram back to a spectrogram.

        Args:
            features (np.ndarray): Batch of Log Mel spectrogram. shape: [Batch, Channel, Frequency, Time]
            _ (int): Feature index. Not used here

        Returns:
            Energy spectrogram corresponding to the input batch of Log Mel spectrogram.
            shape: [Batch, Channel, Frequency, Time]
        """

        return self.inverse_mel_filterbank[np.newaxis, np.newaxis, ...] @ (np.exp(features) - 1e-8)

    def audio_full_filename(self, filename):
        r"""Get the full path to an audio mixture.

        Args:
            filename (str): Path to an audio mixture, relatively to the data_folder

        Returns:
            Path to an audio mixture
        """

        return os.path.join(self.config["audio_folder"], filename)

    def load_audio(self, filename):
        r"""Special audio loading function to make sum (and not average) of stereo files

        Args:
            filename (str): path to audio file

        Returns:
            Re-sampled and converted to mono (sum) audio waveform.
        """

        audio, _ = librosa.core.load(filename, sr=self.config["sampling_rate"], mono=False)
        if audio.ndim > 1:
            audio = np.sum(audio, axis=0)
        return audio

    def load_audio_source_files(self, idx):
        r"""Load the audio sources used to generate the mixture that has the index 'idx'

            The data set contains examples of audio events mixed with background noises, and audio events alone. This
            function makes sure to always return a source for each event class (even if not present in the mixture)
            and for the background, even if there is no event or no background in the mixture.

        Args:
            idx (int): Index of the audio mixture in the data set.

        Returns:
            np.ndarray: shape: [Number of classes (or categories), audio length]
                               Each element of the first dimension is a source used to create the audio mixture 'idx'.
                               Note: Most of the sources will be empty (all-zero)
        """

        if 'event' in self.filenames[idx]:  # event only
            event = self.load_audio(self.audio_full_filename(self.filenames[idx]))
            background = np.zeros(event.shape)
        elif 'bg' in self.filenames[idx]:  # background only
            background = self.load_audio(self.audio_full_filename(self.filenames[idx]))
            event = np.zeros(background.shape)
        elif 'mix' in self.filenames[idx]:  # event and background
            # events are saved in a channel of the mix, background in the other
            audio, _ = librosa.core.load(self.audio_full_filename(self.filenames[idx]), sr=16000, mono=False)
            background = audio[0]
            event = audio[1]

        # Get the empty sources for the events that are not present at the right places.
        class_dict = {class_name: class_idx for class_idx, class_name in enumerate(self.classes)}

        sources = np.zeros((len(self.classes), event.shape[0]))
        sources[class_dict[self.yaml[int(self.filenames[idx].split('.')[0])]['event_type']]] = event
        sources[class_dict['background']] = background

        return sources

    def __getitem__(self, index):
        r"""Torch.data.Dataset method for getting an item in the data set.

            This method needs to be implemented to be able to iterate over the dataset

        Args:
            index (int): Index of an audio example

        Returns:
            (audio features, labels) of this example
        """

        return self.features[index], self.labels[index]

    def __len__(self):
        r"""
        Returns:
            Length of the Dataset (number of audio examples in the data set).
        """

        return self.features.shape[0]


class AudiosetSegments(AudioDataSet):
    r"""This class implements a data set class from audio data taken from Audioset.

        Audioset is a huge data set of weakly labelled videos from youtube. https://research.google.com/audioset/
        From this huge set, 9 categories have been selected to represent possible noise sources. The precise labels
        of the audio events in the recordings, as well as the presence of speech, has been labelled by CloudFactory.
        Using this information, the audio files, initially 10 seconds long, have been split into smaller segments and
        divided into training, testing and validation set.
        The audio mixtures have been saved to a folder. To avoid re-computing the features, the audio features have
        also been computed and saved to hdf5 files.

        We do not have access to the sources that compose the mixtures for this set, so the inversion of the features
        is not required. This class simply implements the reading of the features and the audio from disk and
        accessibility throught the torch.data.Dataset API.

        Note: This set is quite big. It takes 20~25 GB of RAM to load everything into memory during training.

    """

    @classmethod
    def default_config(cls):
        r"""Get audio features parameters

            For this set, the audio features have been computed before hand. Therefore it is not possible to change
            the audio feature parameters. Instead, the parameters will be loaded from a json file (see __init__).

        Returns:
            Dictionary containing audio features parameters.
        """

        config = super(AudiosetSegments, cls).default_config()
        config["multi_loudness"] = False  # allow random change in loudness of examples (see getitem)
        config["categories"] = ["none"]  # allow to group classes together and train to classify categories (see init)
        return config

    @classmethod
    def split(cls, config, which="all"):
        r"""Instantiates 3 AudiosetSegments for training, testing and validation respectively ; or instantiate 1 of
            the 3 according to user input.

            The features have already been computed, this method simply point each set to the correct data file.

        Args:
            config (dict): Configuration dictionary for the data set, containing parameters for the audio processing.
            which (): To chose which set to return: either "all" (default), 'train', 'test' or 'val'

        Returns:
            3 or 1 AudiosetSegments
        """

        config_json_path = os.path.join(config["data_folder"], 'config.json')
        tr_hdf5_path = os.path.join(config["data_folder"], 'train_data.hdf5')
        dev_hdf5_path = os.path.join(config["data_folder"], 'dev_data.hdf5')
        test_hgf5_path = os.path.join(config["data_folder"], 'test_data.hdf5')

        if which == "all":
            return cls(config, config_json_path, tr_hdf5_path), \
                   cls(config, config_json_path, dev_hdf5_path), \
                   cls(config, config_json_path, test_hgf5_path)
        elif which == "train":
            return cls(config, config_json_path, tr_hdf5_path)
        elif which == "test":
            return cls(config, config_json_path, dev_hdf5_path)
        elif which == "val":
            return cls(config, config_json_path, test_hgf5_path)
        else:
            raise ValueError('Set ID ' + which + ' is not available.')

    def __init__(self, config, config_json_path, data_hdf5_path):
        r"""Constructor. Instantiate the data set by loading the features from file.

        Args:
            config (dict): Configuration dictionary
            config_json_path (str): Path to the json file containing the audio processing parameters.
            data_hdf5_path (str): Path to the data file (.hdf5) containing the audio features.
        """

        # First read feature extraction configuration file
        with open(config_json_path, 'r') as config_file:
            data_config = json.load(config_file)
        config.update(data_config)

        super(AudiosetSegments, self).__init__(config)

        # Read the data file.
        with h5py.File(data_hdf5_path, 'r') as data_file:
            self.magnitudes = np.array(data_file.get('stft_magnitudes'))
            self.phases = np.array(data_file.get('stft_phases'))
            self.features = np.array(data_file.get('mel_spectrograms'))
            self.labels = torch.from_numpy(np.array(data_file.get('labels')))
            self.filenames = [os.path.basename(file.decode()) for file in list(data_file.get('filenames'))]

        # Convert features to torch.tensor and add dimension for "channel" shape: [N_examples, Channel, Frequency, Time]
        self.features = torch.Tensor([np.expand_dims(self.stft_magnitude_to_features(self.magnitudes[idx],
                                                                                     self.features[idx]), 0)
                                      for idx in range(self.features.shape[0])])
        self.classes = self.config["classes"]

        # If user provided categories for grouping the classes, we need to merge the labels
        if config["categories"][0] != "none":
            # Get the indices of the classes in each category for each category
            # Example: classes = ['speech', 'glassbreak', 'baby_cry', 'smoke_alarm']
            #          categories = ['speech.baby_cry', 'glassbreak.smoke_alarm']
            #          indices = [[0, 2], [1, 3]]
            indices = [[idx
                        for idx, a_class in enumerate(self.classes)
                        for category_class in category.split('.')
                        if a_class == category_class]
                       for category in config["categories"]]
            new_labels = np.empty((self.labels.shape[0], len(config["categories"])))
            self.labels = self.labels.numpy()
            for category_idx, category_class_indices in enumerate(indices):
                new_labels[:, category_idx] = self.labels[:, category_class_indices].max(axis=1)
            self.labels = torch.from_numpy(new_labels.astype(np.float32))
            self.classes = config["categories"]

    def audio_full_filename(self, filename):
        r"""Get the full path to an audio mixture.

        Args:
            filename (str): Path to an audio mixture, relatively to the data_folder

        Returns:
            Path to an audio mixture
        """
        return os.path.join(self.config["data_folder"], filename)

    def __getitem__(self, index):
        r"""Torch.data.Dataset method for getting an item in the data set.

            This method needs to be implemented to be able to iterate over the dataset.
            In order to train multi-pcen, it is advised in
            Yuxian Wang et al. "Trainable Frontend For Robust and Far-Field Keyword Spotting" (2016)
            to use different mixture volume to help the network learn the dynamic range compression parameters.
            This is done here by randomly changing the intensity of the features during the training.

        Args:
            index (int): Index of an audio example

        Returns:
            (audio features, labels) of this example
        """

        if self.config["multi_loudness"]:
            return (np.random.random() * 0.4 + 0.6) * self.features[index], self.labels[index]
        return self.features[index], self.labels[index]

    def __len__(self):
        r"""
        Returns:
            Length of the Dataset (number of audio examples in the data set).
        """

        return self.features.shape[0]
