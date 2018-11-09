import os
from abc import abstractmethod
import h5py
import yaml

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
    if data_set_type == "DCASE2013RemixedDataSet":
        return DCASE2013RemixedDataSet
    elif data_set_type == "ICASSP2018JointSeparationClassificationDataSet":
        return ICASSP2018JointSeparationClassificationDataSet
    else:
        raise NotImplementedError("Data set type " + data_set_type + " is not available.")


class AudioDataSet(torchdata.Dataset):
    """
        This class implements the common audio processing functions that are used to load an audio .wav file and
        extract features from it.
    """

    @classmethod
    def default_config(cls):
        config = {
            # Mix files parameters
            "sampling_rate": 0,

            # Feature extraction parameters (log Mel spectrogram computation)
            "feature_type": "log-mel",
            "STFT_frame_width_ms": 0,
            "STFT_frame_shift_ms": 0,
            "STFT_window_function": "hamming",
            "n_Mel_filters": 0,
            "Mel_min_freq": 0,
            "Mel_max_freq": 0,

            "data_folder": "Datadir/",

            "scaling_type": "standardization"  # type of feature normalization: "min-max scaling", "standardization"
        }
        return config

    def __init__(self, config):
        super(AudioDataSet, self).__init__()

        self.config = config
        self.mel_filterbank = librosa.filters.mel(self.config["sampling_rate"],
                                                  n_fft=int(np.floor(self.config["STFT_frame_width_ms"]
                                                                     * self.config["sampling_rate"] // 1000)),
                                                  n_mels=self.config["n_Mel_filters"],
                                                  fmin=self.config["Mel_min_freq"],
                                                  fmax=self.config["Mel_max_freq"]).astype(np.float32)
        self.inverse_mel_filterbank = np.linalg.pinv(self.mel_filterbank)

    @classmethod
    @abstractmethod
    def split(cls, config, which="all"):
        pass

    @abstractmethod
    def features_shape(self):
        pass

    @abstractmethod
    def n_classes(self):
        pass

    @abstractmethod
    def to(self, device):
        pass

    @abstractmethod
    def compute_shift_and_scaling(self):
        pass

    @abstractmethod
    def shift_and_scale(self, shift, scaling):
        pass

    @abstractmethod
    def un_shift_and_scale(self, shift, scaling):
        pass

    @abstractmethod
    def rescale_to_initial(self, features, shift, scaling):
        pass

    def stft_magnitude_to_features(self, magnitude):
        mel_spectrogram = self.mel_filterbank @ magnitude
        if self.config["feature_type"] == "mel":
            return mel_spectrogram
        elif self.config["feature_type"] == "log-mel":
            log_mel_spectrogram = 10.0 * np.log10(mel_spectrogram + 1.0)  # +1 to make sure log is contracting
            return log_mel_spectrogram

    def features_to_stft_magnitudes(self, features):
        if self.config["feature_type"] == "log-mel":
            features = np.power(10.0*np.ones(features.shape), (features/10.0)) - 1.0
        return self.inverse_mel_filterbank[np.newaxis, np.newaxis, ...] @ features

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
        phase = stft / (magnitude + 1e-15)
        return magnitude, phase

    def istft(self, stft):
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
        audio, _ = librosa.core.load(filename, sr=self.config["sampling_rate"], mono=True)
        return audio

    @abstractmethod
    def audio_full_filename(self, idx):
        pass


class DCASE2013RemixedDataSet(AudioDataSet):
    """
        This class implements the audio processing to apply on the audio files remixed from the DCASE2013 data set.
        For speed, all the audio processing is done during the initialization method, then the features and labels
        are available in memory for fast access during training (and can be moved to gpu with the 'to' method).
    """

    @classmethod
    def default_config(cls):
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

            # Path to the mix files folder (also include the label file) (needed if building from audio files)
            "data_folder": "Datadir/remixed_DCASE2013",  # to this will be appended the set folder (train-dev-val)

            # Categories for regrouping the classes together: list of categories, each category is indicated as a string
            # of the classes in the category separated by dots.for
            # eg: --class_categories speech.laughter clearthroat.cough doorslam.drawer.keys.knock.pendrop.switch \
            # keyboard.mouse phone.alert pageturn printer
            "class_categories": ['all_separated'],

            "data_set_save_folder_path": "",
            "data_set_load_folder_path": "",  # (needed if building from a pre-saved data set)
            "thread_max_worker": 3,  # Number of thread for loading the audio data (if build from audio files)

            "scaling_type": "standardization"  # type of feature normalization: "min-max scaling", "standardization"
        })
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

        if which == "all":
            return cls(tr_config), cls(dev_config), cls(test_config)
        elif which == "train":
            return cls(tr_config)
        elif which == "dev":
            return cls(dev_config)
        elif which == "test":
            return cls(test_config)
        raise ValueError("ID " + which + " is not valid.")

    def __init__(self, config):
        """
            Initiates the data set: - build the mel filter bank for audio processing
                                    - Load all files from disk and extract the features (Mel spectrogram)
                                    - Convert features and labels to torch.Tensor to have everything ready in memory.
        Args:
            config (dict): Configuration dictionary containing parameters for audio features extraction.
        """
        super(DCASE2013RemixedDataSet, self).__init__(config)

        try:
            self.magnitudes, self.phases, self.features, self.labels, self.classes, self.filenames = \
                self.build_from_file()
        except ValueError as e:
            print(e)
            print("Building data set from audio files !")
            files_df = pd.read_csv(os.path.join(config["data_folder"], "weak_labels.csv"))
            files_df = files_df.reindex(sorted(files_df.columns), axis=1)  # sort columns
            self.magnitudes, self.phases, self.features, self.labels, self.classes, self.filenames = \
                self.build_from_audio_files(files_df)

        if self.config["data_set_save_folder_path"]:
            self.save_to_file()

    def build_from_audio_files(self, files_df):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config["thread_max_worker"]) as executor:
            audios = executor.map(lambda file: self.load_audio(os.path.join(self.config["data_folder"], file)),
                                  files_df["filename"])
        magnitudes, phases = tuple(map(lambda x: np.asarray(list(x)),
                                       zip(*[self.separated_stft(audio) for audio in audios])))
        features = torch.Tensor([np.expand_dims(self.stft_magnitude_to_features(stft), 0)
                                 for stft in magnitudes])

        if self.config["class_categories"] != self.default_config()["class_categories"]:
            category_df = pd.DataFrame()
            for category in self.config["class_categories"]:
                category_df[category] = files_df.drop("filename", axis=1).apply(
                    lambda row: any([row[event] for event in category.split('.')]), axis=1)
            labels = torch.from_numpy(category_df.values.astype(np.float32))
            classes = category_df.columns
        else:
            labels = torch.from_numpy(files_df.drop("filename", axis=1).values.astype(np.float32))
            classes = list(files_df.columns)
            classes.remove("filename")

        filenames = files_df["filename"].tolist()
        return magnitudes, phases, features, labels, classes, filenames

    def build_from_file(self):
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

    def save_to_file(self):
        if not os.path.exists(self.config["data_set_save_folder_path"]):
            os.makedirs(self.config["data_set_save_folder_path"])
        path = os.path.join(self.config["data_set_save_folder_path"],
                            os.path.basename(self.config["data_folder"])) + '.h5'
        with h5py.File(path, 'w') as hf:
            # save parameters of the default config as a python string
            hf.create_dataset('magnitudes', data=self.magnitudes)
            hf.create_dataset('phases', data=self.phases)
            hf.create_dataset('features', data=self.features.numpy())
            hf.create_dataset('labels', data=self.labels.numpy())
            hf.create_dataset('classes', data=np.array(self.classes, dtype='S'))
            hf.create_dataset('filenames', data=np.array(self.filenames, dtype='S'))

    def features_shape(self):
        return tuple(self.features[0].shape)

    def n_classes(self):
        return self.labels.shape[1]

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
        channel_shift = [np.nan] * n_channels
        channel_scaling = [np.nan] * n_channels
        for i in range(self.features.shape[1]):
            if self.config["scaling_type"] == "standardization":
                channel_shift[i] = self.features[:, i, :, :].mean()
                channel_scaling[i] = self.features[:, i, :, :].std()
            elif self.config["scaling_type"] == "min-max":
                channel_shift[i] = self.features[:, i, :, :].min()
                channel_scaling[i] = self.features[:, i, :, :].max() - channel_shift[i]  # max - min
            elif not self.config["scaling_type"]:
                print("[WARNING] No normalization procedure is specified !")
                channel_shift[i] = 0.0
                channel_scaling[i] = 1.0

        return channel_shift, channel_scaling

    def shift_and_scale(self, shift, scaling):
        for i in range(self.features.shape[1]):  # per channel
            self.features[:, i, :, :] = (self.features[:, i, :, :] - shift[i]) / scaling[i]

    def un_shift_and_scale(self, shift, scaling):
        for i in range(self.features.shape[1]):
            self.features[:, i, :, :] = (self.features[:, i, :, :] * scaling[i]) + shift[i]

    def rescale_to_initial(self, features, shift, scaling):
        for i in range(features.shape[1]):
            features[:, i, :, :] = (features[:, i, :, :] * scaling[i].to(features[:, i, :, :].device)) \
                                   + shift[i].to(features[:, i, :, :].device)

    def audio_full_filename(self, filename):
        return os.path.join(self.config["data_folder"], filename)

    def load_audio_source_files(self, idx):
        source_files_dir = os.path.join(self.config["data_folder"],
                                        os.path.splitext(self.filenames[idx])[0])
        source_files = sorted(os.listdir(source_files_dir))
        reference_sources = np.asarray([self.load_audio(os.path.join(source_files_dir, filename))
                                        for filename in source_files])

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
        return self.features[index], self.labels[index]

    def __len__(self):
        return self.features.shape[0]


class ICASSP2018JointSeparationClassificationDataSet(AudioDataSet):
    """

    """

    @classmethod
    def default_config(cls):
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
        super(ICASSP2018JointSeparationClassificationDataSet, self).__init__(config)

        self.config = config

        with h5py.File(config["data_file"], 'r') as hf:
            self.filenames = [file.decode() for file in list(hf.get('na_list'))]
            self.features = torch.from_numpy(np.array(hf.get('x'))).unsqueeze(1).permute(0, 1, 3, 2)
            self.labels = torch.from_numpy(np.array(hf.get('y')))

        self.mix_idx = [idx for idx, filename in enumerate(self.filenames) if 'mix' in filename]
        self.event_idx = [idx for idx, filename in enumerate(self.filenames) if 'event' in filename]
        self.background_idx = [idx for idx, filename in enumerate(self.filenames) if 'bg' in filename]
        with open(config["yaml_file"], 'r') as yaml_stream:
            self.yaml = yaml.safe_load(yaml_stream)

        with concurrent.futures.ThreadPoolExecutor(max_workers=config["thread_max_worker"]) as executor:
            audios = executor.map(lambda file: self.load_audio(os.path.join(self.config["audio_folder"], file)),
                                  self.filenames)
        self.magnitudes, self.phases = tuple(map(lambda x: np.asarray(list(x)),
                                                 zip(*[self.separated_stft(audio) for audio in audios])))
        self.classes = ['babycry', 'glassbreak', 'gunshot', 'background']

    @classmethod
    def split(cls, config, which="all"):
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
        elif which == "dev" or which == "test":
            print("WARNING: Development and Validation set are the same for this set !")
            return cls(test_config)
        else:
            raise ValueError("Set identifier " + which + " is not available.")

    def features_shape(self):
        return tuple(self.features[0].shape)

    def n_classes(self):
        return self.labels.shape[1]

    def to(self, device):
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)

    def compute_shift_and_scaling(self):
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
            elif not self.config["scaling_type"]:
                print("[WARNING] No normalization procedure is specified !")
                channel_shift[i] = 0.0
                channel_scaling[i] = 1.0

        return channel_shift, channel_scaling

    def shift_and_scale(self, shift, scaling):
        for i in range(self.features.shape[1]):  # per channel
            self.features[:, i, :, :] = (self.features[:, i, :, :] - shift[i]) / scaling[i]

    def un_shift_and_scale(self, shift, scaling):
        for i in range(self.features.shape[1]):
            self.features[:, i, :, :] = (self.features[:, i, :, :] * scaling[i]) + shift[i]

    def rescale_to_initial(self, features, shift, scaling):
        for i in range(features.shape[1]):
            features[:, i, :, :] = (features[:, i, :, :] * scaling[i].to(features[:, i, :, :].device))\
                                   + shift[i].to(features[:, i, :, :].device)

    def features_to_stft_magnitudes(self, features):
        return self.inverse_mel_filterbank[np.newaxis, np.newaxis, ...] @ (np.exp(features) - 1e-8)

    def audio_full_filename(self, filename):
        return os.path.join(self.config["audio_folder"], filename)

    def load_audio_source_files(self, idx):
        if 'event' in self.filenames[idx]:
            event = self.load_audio(self.audio_full_filename(self.filenames[idx]))
            background = np.zeros(event.shape)
        elif 'bg' in self.filenames[idx]:
            background = self.load_audio(self.audio_full_filename(self.filenames[idx]))
            event = np.zeros(background.shape)
        elif 'mix' in self.filenames[idx]:
            event = self.load_audio(self.audio_full_filename(self.filenames[idx-1]))
            background = self.load_audio(self.audio_full_filename(self.filenames[idx-2]))

        class_dict = {class_name: class_idx for class_idx, class_name in enumerate(self.classes)}

        sources = np.zeros((len(self.classes), event.shape[0]))
        sources[class_dict[self.yaml[idx//3]['event_type']]] = event
        sources[class_dict['background']] = background

        return sources

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return self.features.shape[0]
