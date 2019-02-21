import torch
import librosa
import numpy as np

import mir_eval

import separation_model as md
import data_set as dts
from shutil import copyfile

import os


class AudioSeparator:
    r"""Implements a framework for using a SeparationModel to produce separated source for all files in the
        validation set and measure the separation performances in terme of signal to distortion ratio (SDR),
        signal to interference ratio (SIR) and signal to artifact ratio (SAR).

    """

    @classmethod
    def default_config(cls):
        r"""Get the required parameters for instantiating a AudioSeparator

            The configuration parameters for the model and the AudioDataSet are saved in the model checkpoint. All we
            need for instantiation is the path to the check point.
            The path to the folder to use for saving the separated audio tracks is also exposed.

        Returns:
            dict containing the required parameters
        """

        config = {
            "checkpoint_path": "",  # path to model checkpoint
            "separated_audio_folder": ""  # path to folder where to save the separated audio tracks.
        }
        return config

    def __init__(self, data_set, model, config):
        r"""Constructor. Receives the AudioDataSet and the Model and stores them as class members.

            Note: The received data_set features should not be scaled or centered.

        Args:
            data_set (AudioDataSet): The data set with the mixtures to separate
            model (SeparationModel): The separation model for performing separation
            config (dict): Configuration dictionary with parameters for the model, dataset and self.
        """

        self.config = config
        self.data_set = data_set
        # Normalize or standardize the features, to have them ready to use as model input
        self.data_set.shift_and_scale(self.config["shift"], self.config["scaling"])
        self.model = model
        self.model.eval()
        self.device = torch.device("cpu") if not self.config["use_gpu"] \
            else torch.device("cuda:" + str(self.config["gpu_no"]))

    @classmethod
    def from_checkpoint(cls, config, which_data_set="test"):
        r"""Instantiate an AudioSeparator from a model checkpoint.

            Loads the model from its checkpoint.
            The checkpoint also contains the configuration dictionary required to create the validation set related
            to the set used to train the model.

        Args:
            config (dict): Configuration dictionary with the parameters in defined in 'default_config()'
            which_data_set (str): Identifier of the set type for the 'split' method of the AudiodataSet. 'train',
                                  'test' or 'val'

        Returns:
            AudioSeparator using the model loaded from the checkpoint path in 'config'
        """

        # Load the checkpoint
        filename = config["checkpoint_path"]
        if not os.path.isfile(filename):
            raise ValueError("File " + filename + " is not a valid file.")
        print("Loading model ...'{}'".format(filename))
        state = torch.load(filename, 'cpu')

        # Get the configuration paramters used during the training of the model.
        train_config = state["config"]
        # Update those parameters with the AudioSeparator parameters.
        train_config.update(config)

        # Build the data set containing the audio to separate.
        val_set = dts.find_data_set_class(train_config["data_set_type"]).split(train_config, which_data_set)

        # Build the SeparationModel and load its parameters
        model = md.SeparationModel(train_config, val_set.features_shape(), val_set.n_classes())
        model.load_state_dict(state["model_state_dict"])

        # Build the AudioSeparator
        return cls(val_set, model, train_config)

    def separate_spectrogram(self, masks, features, features_idx):
        r"""Apply masks to models input features to generate a spectrogram for each audio source.

             There are many ways to use separation masks to produce spectrograms for each sources in the input features.
             This function does the following:
                - Rescale the masks to the shape of the SeparationModel input
                (this is only useful if the MaskModel in the SeparationModel does not preserve the shape of its input
                with padding)
                - Shift the features to [0, +inf[, apply the mask and shift back.
                (This is because the features can have negative values, and we want a value of 0 in the mask to
                correspond to the lowest possible energy)
                - The previous step provides us with 'masked features': these features should correspond to separated
                sources. The last step is to convert back these features (scaled and centered log-Mel-spectrogram,
                PCEN, ...) to a 'spectrogram' representation that can be converted back to audio with Inverse STFT.

            Note: It has be found experimentally that apply the masks at the 'features' level give worst results than
            converting the masks to 'spectrogram' representation and applying them directly to the mixture
            spectrogram, because converting the features back to the spectrogram scale often implies to take the
            exponential of the fetures which amplifies a lot the noise.
            The other processing is performed by 'separate_spectrogram_in_lin_scale()'.

        Args:
            masks (torch.Tensor): Shape: [n_class, ~freq, ~time]. The masks produced by the separation model.
            features (torch.Tensor): Shape [channel, freq, time]. The input features to the separation model.
            features_idx (int): index of the features in data_set.features
        Returns:
            Spectrogram of the sources separated by the masks. shape: [n_sources, channel=1, Frequency, Time]
        """

        # resize the masks to the size of the features  (shape: [n_masks, channel, freq, time]
        # This does something only if the masks have different shape than features (if MaskModel doesn't preserve shape)
        masks = torch.nn.functional.interpolate(masks.unsqueeze(1),
                                                size=(features.shape[1], features.shape[2]),
                                                mode='bilinear',
                                                align_corners=False)
        # Multiply each mask with the features (shape: [n_masks, channel, features.shape[0], features.shape[1]]
        shift = features.abs().max()
        spectrograms = masks * (features + shift) - shift
        # Undo the feature scaling and centering
        self.data_set.rescale_to_initial(spectrograms, self.config["shift"], self.config["scaling"])
        # From Log Mel spectrogram or PCEN to STFT magnitude (energy spectrogram)
        return self.data_set.features_to_stft_magnitudes(spectrograms.cpu().numpy(), features_idx)

    def separate_spectrogram_in_lin_scale(self, masks, features_shape, mixture_spectrogram):
        r"""Apply masks to the mixture spectrogram to generate spectrograms for each separated sources.

            The masks received in argument have the shape of the output of the MaskModel. In this function,
            these masks will first be converted to the shape of the mixture energy spectrogram (inverse Mel scaling)
            and then be directly applied to the mixture spectrogram.

        Args:
            masks (torch.tensor): Shape: [n_class, ~freq, ~time] The masks produced by the separation model
            features_shape (torch.tensor.shape): Shape of the input features to the separation model.
            mixture_spectrogram (np.ndarray): shape: [Frequency, Time] Mixture spectrogram.

        Returns:
            Spectrogram of the sources separated by the masks. shape: [n_sources, channel=1, Frequency, Time]
        """

        # resize the masks to the size of the features  (shape: [n_masks, channel, freq, time]
        # This does something only if the masks have different shape than features (if MaskModel doesn't preserve shape)
        masks = torch.nn.functional.interpolate(masks.unsqueeze(1),
                                                size=(features_shape[1], features_shape[2]),
                                                mode='bilinear',
                                                align_corners=False)
        # If Mel spectrogram were used as features: reverse Mel-scaling
        # Here we use the same inverse processing as in the implementation of
        # Qiuqiang Kong et al. "A joint-separation-classification model for sound event detection of weakly-labelled
        # data"; In: CoRR abs/1711.03037 (2017). axXiv: 1711.03037 URL: http://arxiv.org/abs/1711.03037
        if self.config['feature_type'] != 'spectrogram':
            masks = np.asarray([np.transpose(
                self.data_set.mel_filterbank / (np.sum(self.data_set.mel_filterbank, axis=0) + 1e-8)) @ mask
                                for mask in masks.squeeze()])

        # Apply the masks to the mixture spectrogram. Mask.shape: [n_sources, channel=1, Frequency, Time]
        #                              mixture_spectrogram.shape: [Frequency, Time]
        #                                           output.shape: [n_sources, channel=1, Frequency, Time]
        return masks * mixture_spectrogram

    def spectrogram_to_audio(self, spectrogram, phase):
        r"""Compute waveform from spectrogram using inverse short-time Fourier transform.

            Wrapper to call the istft function from the AudioDataSet class that performs the ISTFT with the
            parameters corresponding to the STFT.

        Args:
            spectrogram (np.ndarray): shape: [Frequency, Time]. Magnitude of STFT result
            phase (np.ndarray): shape: [Frequency, Time]. Phase of STFT result

        Returns:
            audio waveform. (1D np.ndarray)
        """

        return self.data_set.istft(spectrogram * phase)

    def save_separated_audio(self, audios, filename):
        r"""Save the audios tracks in audios, in a subfolder of self.config['separated_audio_folder'].

            'audios' should be the sources separated by the SeparationModel for the audio mixture saved in 'filename'.
            The separated tracks are saved in a folder with the same name than their corresponding mixture.
            The mixture is also copied inside the folder.

        Args:
            audios (np.ndarray): shape: [n_sources, time]. Audio waveforms of the separated sources
            filename (str): Name of the file containing the audio mixture.
        """

        # Create folder with mixture name
        folder_path = os.path.join(self.config["separated_audio_folder"], os.path.splitext(filename)[0])
        os.makedirs(folder_path)
        # Save each separated source
        for class_idx, audio in enumerate(audios):
            librosa.output.write_wav(os.path.join(folder_path, self.data_set.classes[class_idx]) + '.wav',
                                     audio.T,
                                     sr=self.data_set.config["sampling_rate"])
        # Also copy the mixture in the folder
        copyfile(self.data_set.audio_full_filename(filename), os.path.join(folder_path, "original_mix.wav"))

    def separate(self, separation_method='in_lin'):
        r"""Run separation with self.model for all the files in self.data_set and save the separated sources.

        Args:
            separation_method (str): Identifier to chose between methods for applying the masks. Chose between
                                     separate at the feature level ('separate_spectrogram') or at the energy
                                     spectrogram level ('separate_spectrogram_in_lin').
                                     Advised: 'in_lin'
        """

        # Check if the output folder exists, if not creates it, otherwise inform user and stop execution
        if not os.path.exists(self.config["separated_audio_folder"]):
            os.makedirs(self.config["separated_audio_folder"])
        else:
            if os.listdir(self.config["separated_audio_folder"]):  # if folder is not empty
                raise ValueError('Output folders already exist !')

        self.model.to(self.device)
        self.model.eval()
        self.data_set.to(self.device)

        # Loop over all the files in the dataset.
        for idx in range(self.data_set.__len__()):
            # Get the features
            features = self.data_set.__getitem__(idx)[0]
            # Get the separation masks
            _, masks = self.model(features.unsqueeze(0))  # (add batch dimension)
            masks = masks.detach().squeeze()  # move "mask" dim in first position
            # Apply the masks
            if separation_method == 'in_log':
                spectrograms = self.separate_spectrogram(masks, features, idx)
            elif separation_method == 'in_lin':
                spectrograms = self.separate_spectrogram_in_lin_scale(masks, features.shape,
                                                                      self.data_set.magnitudes[idx])
            else:
                raise ValueError('Separation method ' + separation_method + ' is not available.')
            # Get the separated audio and save
            audios = [self.spectrogram_to_audio(spectrogram, self.data_set.phases[idx]) for spectrogram in spectrograms]
            self.save_separated_audio(audios, self.data_set.filenames[idx])

    def evaluate_separation(self, indices=None):
        r"""Compute separation metrics using the separated sources in self.config['separated_audio_folder']

            Assuming 'separate()' has been previously called: the separated sources for all the audio files in
            self.data_set are stored in self.config['separated_audio_folder'].
            This function loads the separated sources and the ground-truth sources to compute separation metrics.

            Separation metrics used here are:
                - Signal to Distortion ratio (SDR)
                - Signal to Interference ratio (SIR)
                - Signal to Artifact ratio (SAR)
            Those are computed using the mir_eval library.
            Note: These estimators are not very reliable to estimate separation quality. Unfortunately they are the
            most common used in the litterature. Here we use the 'bss_eval_images' function that does use a filtered
            version of the ground-truth sources, but also do not allow for scale changes.
            For discussions of the measurements quality, see:
            Jonathan Le Roux et al. (2018). "SDR - half-baked or well done?". CoRR, abs/1811.02508.

        Args:
            indices (int): If passed: compute separation metrics for the file of the given indices. Otherwise: do
                           entire data set

        Returns:
            sdr, sir, sar: np.ndarray of shape [n_files, n_sources]
        """

        # if indices is passed: evaluate separation for the file of the given indices. Otherwise: do entire data set
        if indices is None:
            indices = np.arange(self.data_set.__len__())
        sdr = np.zeros((indices.shape[0], len(self.data_set.classes)))
        sir = np.zeros((indices.shape[0], len(self.data_set.classes)))
        sar = np.zeros((indices.shape[0], len(self.data_set.classes)))

        for idx in indices:
            # Load separated sources
            # Take care of sorting the sources here and in data_set class in the same way to have consistent labels
            # Take care not to load the 'original_mix' file which is the mixture file.
            separated_sources = np.asarray([self.data_set.load_audio(os.path.join(self.config["separated_audio_folder"],
                                                                                  os.path.splitext(
                                                                                      self.data_set.filenames[idx])[0],
                                                                                  filename))
                                            for filename in sorted(  # sort in the same order than in data_set class
                    os.listdir(os.path.join(self.config["separated_audio_folder"],
                                            os.path.splitext(self.data_set.filenames[idx])[0])))
                                            if 'mix' not in filename])  # original mix is copied over with sep. sources

            # Get the ground-truth sources from self.data_set
            reference_sources = self.data_set.load_audio_source_files(idx)
            # Crop to length of reconstructed signal (because last non-complete frame of the stft is dropped)
            # Add small offset to avoid having sources always 0 (mir_eval does not like that)
            reference_sources = reference_sources[:, :separated_sources.shape[1]] + 1e-15

            sdr[idx], _, sir[idx], sar[idx], _ = mir_eval.separation.bss_eval_images(reference_sources,
                                                                                     separated_sources,
                                                                                     compute_permutation=False)
        return sdr, sir, sar
