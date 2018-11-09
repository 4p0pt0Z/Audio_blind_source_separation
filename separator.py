import torch
import librosa
import numpy as np

import mir_eval

import segmentation_model as md
import data_set as dts
from shutil import copyfile

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
        self.config = config
        self.data_set = data_set
        self.data_set.shift_and_scale(self.config["shift"], self.config["scaling"])
        self.model = model
        self.model.eval()
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

        state = torch.load(filename, 'cpu')
        train_config = state["config"]
        train_config.update(config)

        test_set = dts.find_data_set_class(train_config["data_set_type"]).split(train_config, which_data_set)

        model = md.SegmentationModel(train_config, test_set.features_shape(), test_set.n_classes())
        model.load_state_dict(state["model_state_dict"])

        return cls(test_set, model, train_config)

    def separate_spectrogram(self, masks, features):
        """

        Args:
            masks (torch.Tensor): Shape: [n_class, ~freq, ~time]. The masks output of the segmentation model.
            features (torch.Tensor): Shape [channel, freq, time]. The input features to the segmentation model.

        Returns:

        """
        # resize the masks to the size of the features  (shape: [n_masks, channel, freq, time]
        masks = torch.nn.functional.interpolate(masks.unsqueeze(1),
                                                size=(features.shape[1], features.shape[2]),
                                                mode='bilinear',
                                                align_corners=False)
        # Multiply each mask per the features (shape: [n_masks, features.shape[0], features.shape[1]]
        spectrograms = masks * features
        # Undo the feature scaling
        self.data_set.rescale_to_initial(spectrograms, self.config["shift"], self.config["scaling"])
        # Go back to "stft output" representation
        return self.data_set.features_to_stft_magnitudes(spectrograms.cpu().numpy())

    def spectrogram_to_audio(self, spectrogram, phase):
        return self.data_set.istft(spectrogram * phase)

    def save_separated_audio(self, audios, filename):
        folder_path = os.path.join(self.config["separated_audio_folder"], os.path.splitext(filename)[0])
        os.makedirs(folder_path)
        for class_idx, audio in enumerate(audios):
            librosa.output.write_wav(os.path.join(folder_path, self.data_set.classes[class_idx]) + '.wav',
                                     audio.T,
                                     sr=self.data_set.config["sampling_rate"])
        copyfile(self.data_set.audio_full_filename(filename), os.path.join(folder_path, "original_mix.wav"))

    def separate(self):
        self.model.to(self.device)
        self.model.eval()
        self.data_set.to(self.device)

        for idx in range(self.data_set.__len__()):
            features = self.data_set.__getitem__(idx)[0]
            _, masks = self.model(features.unsqueeze(0))  # (add batch dimension)
            masks = masks.detach().squeeze()  # move "mask" dim in first position
            spectrograms = self.separate_spectrogram(masks, features)
            audios = [self.spectrogram_to_audio(spectrogram, self.data_set.phases[idx]) for spectrogram in spectrograms]
            self.save_separated_audio(audios, self.data_set.filenames[idx])

    def evaluate_separation(self):
        sdr = np.zeros((self.data_set.__len__(), len(self.data_set.classes)))
        sir = np.zeros((self.data_set.__len__(), len(self.data_set.classes)))
        sar = np.zeros((self.data_set.__len__(), len(self.data_set.classes)))

        for idx in range(self.data_set.__len__()):
            separated_sources = np.asarray([self.data_set.load_audio(os.path.join(self.config["separated_audio_folder"],
                                                                                  os.path.splitext(
                                                                                      self.data_set.filenames[idx])[0],
                                                                                  filename))
                                            for filename in sorted(  # idx is not same for filename here and in class
                                            os.listdir(os.path.join(self.config["separated_audio_folder"],
                                                                    os.path.splitext(self.data_set.filenames[idx])[0])))
                                            if 'mix' not in filename])

            reference_sources = self.data_set.load_audio_source_files(idx)
            # Crop to length of reconstructed signal (because last non-completed frames of fft is dropped)
            # Add small offset to avoid having sources always 0 (mir_eval does not like that)
            reference_sources = reference_sources[:, :separated_sources.shape[1]] + 1e-15

            sdr[idx], sir[idx], sar[idx], _ = mir_eval.separation.bss_eval_sources(reference_sources,
                                                                                   separated_sources,
                                                                                   compute_permutation=False)

            # for i_class in range(separated_sources.shape[0]):
            #     sdr[idx, i_class], sir[idx, i_class], sar[idx, i_class], _ =\
            #         mir_eval.separation.bss_eval_sources(reference_sources[i_class],
            #                                              separated_sources[i_class],
            #                                              compute_permutation=False)
        return sdr, sir, sar
