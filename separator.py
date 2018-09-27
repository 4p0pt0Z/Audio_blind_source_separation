import torch
import librosa
import scipy

from skimage.transform import resize

import model as md
import data_set as dts

import os


class AudioSeparator:
    """

    """

    @classmethod
    def default_config(cls):
        config = {
            "checkpoint_path": "",
            "separated_audio_folder": ""
        }
        return config

    def __init__(self, data_set, model, config):
        self.data_set = data_set
        self.model = model
        self.config = config
        self.device = torch.device("cpu") if not self.config["use_gpu"] \
            else torch.device("cuda:" + str(self.config["gpu_no"]))
        # Check if the output folder exists, if not creates it, otherwise inform user and stop execution
        if not os.path.exists(self.config["separated_audio_folder"]):
            os.makedirs(self.config["separated_audio_folder"])
        else:
            if os.listdir(self.config["separated_audio_folder"]):  # if folder is not empty
                raise ValueError('Output folders already exist !')

    @classmethod
    def from_checkpoint(cls, config, which_data_set="test"):
        filename = config["checkpoint_path"]
        if not os.path.isfile(filename):
            raise ValueError("File " + filename + " is not a valid file.")
        print("Loading model ...'{}'".format(filename))

        state = torch.load(filename)
        train_config = state["config"]
        train_config.update(config)

        test_set = dts.find_data_set_class(train_config["data_set_type"]).split(train_config, which_data_set)

        model = md.find_model_class(train_config["model_type"])(train_config, test_set.features_shape())
        model.load_state_dict(state["model_state_dict"])

        return cls(test_set, model, train_config)

    def separate_spectrogram(self, masks, magnitude):
        separated_spectrograms = [None] * masks.shape[0]
        for idx, mask in enumerate(masks):
            # Interpolate the masks to the shape of the input of network
            # mask = resize(mask, self.data_set.features_shape(), preserve_range=True)
            mask = scipy.misc.imresize(mask, (self.data_set.features_shape()[1], self.data_set.features_shape()[2]))
            # From mel scale to fft frequencies
            mask = self.data_set.inverse_mel_filterbank @ mask
            # Apply mask
            separated_spectrograms[idx] = magnitude * mask

        return separated_spectrograms

    def spectrogram_to_audio(self, spectrogram, phase):
        return self.data_set.istft(spectrogram * phase)

    def save_separated_audio(self, audios, filename):
        folder_path = os.path.join(self.config["separated_audio_folder"], filename)
        os.makedirs(folder_path)
        for class_idx, audio in enumerate(audios):
            librosa.output.write_wav(os.path.join(folder_path, self.data_set.classes[class_idx]),
                                     audio,
                                     sr=self.data_set.config["sampling_rate"])

    def separate(self):
        self.model.to(self.device)
        self.model.eval()
        self.data_set.to(self.device)
        self.data_set.shift_and_scale(self.config["shift"], self.config["scaling"])

        for idx in range(self.data_set.__len__()):
            _, masks = self.model(self.data_set.__getitem__(idx)[0].unsqueeze(0))  # add batch dimension
            masks = masks.to('cpu').numpy().squeeze()  # remove batch dimension
            spectrograms = self.separate_spectrogram(masks, self.data_set.magnitudes[idx])
            audios = [self.spectrogram_to_audio(spectrogram, self.data_set.phases[idx]) for spectrogram in spectrograms]
            self.save_separated_audio(audios, self.data_set.filenames[idx])
