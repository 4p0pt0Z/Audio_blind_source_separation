import torch
import torch.nn as nn

from mask_model import find_mask_model_class
from classifier_model import find_classifier_model_class, PredictionReScaler
from pcen import PCENLayer, MultiPCENlayer


class SeparationModel(nn.Module):
    r"""Class implementing the model in the training phase: (optional) pcen layer followed by the mask model followed
        by the classifier model.

        This class helps with the interactions between the different models (input shapes, parameters...), however each
        specific model is implemented separately.
    """

    @classmethod
    def default_config(cls, mask_model_type, classifier_model_type):
        r"""Get all the parameters for the pcen layer, mask model and classifier model.

            WARNING: Note that some of these parameters are overwritten in the __init__ method for consistency between
            the mask model and classifier model !
        Args:
            mask_model_type (str): type of the mask model
            classifier_model_type (str): type of the classifier model

        Returns:
            dictionary with keys for all the parameters in the (optional) pcen layer, mask model and classifier model
        """

        # Parameters to configure the (optional) trainable pcen layer.
        # Note that if a trainable pcen is used, the features type should be mel-spectrogram.
        pcen_config = {
            # Whether to use a pcen trainable layer. Non-trainable layer are handled by
            # the AudioDataSet class (considered to be feature extraction).
            "train_pcen": False,

            # Whether to train multiple pcen layers and stack their outputs to use as input channels to the mask model.
            "train_multi_pcen": False, "n_multi_pcen": 0,

            # Train a separate parameter (alpha, delta, r) for each frequency bin (True) or one value for all
            # frequency bins (False).
            # When multiple pcen are used, The parameter 'delta', 'r', 'alpha' and 's' are directly trained,
            # and initialized to their default value in Yuxian Wang et al. "Trainable Frontend For Robust and
            # Far-Field Keyword Spotting" (2016)
            "train_pcen_per_band_param": False,

            # If True: the training is done as described in Yuxian Wang et al. "Trainable Frontend For Robust and
            # Far-Field Keyword Spotting" (2016)
            # ie: use fixed set of parameter 's', learn weights w_k to give to each filter M_s.
            # If False: will use one filter only, and train the filter parameters. Filter parameters are defined as
            # the polynomial coefficient of the transfert function: 'b' is the numerator polynomial and 'a' is the
            # denominator polynomial.
            # see "Transfer function derivation" section at https://en.wikipedia.org/wiki/Infinite_impulse_response
            # WARNING: When training multiple pcen layers, those options are not available
            "train_pcen_use_s": True,
            "train_pcen_s": [0.0],  # values of s to use for the fixed filters
            # If training the filter 'a' and 'b' polynomial, there is choice: train one value for the all
            # frequency bins (False), or train a different value for all frequency bins (True)
            "train_pcen_per_band_filter": False,
            # Polynomial coefficients initial values. The length of the list defines the order of the
            # polynomial thus the order of the filter.
            "train_pcen_b": [0.0], "train_pcen_a": [1.0],
            # Value for the epsilon parameter (used for numerical stability of normalization)
            "train_pcen_eps": 1e-6}

        # Parameters of the mask model. Those are the parameters exposed by the default_config function of the mask
        # model class. Command line argument to modify those parameters should be prefixed by 'mask_' to
        # differentiate these parameters from other parameters (eg parameters of the classifier model).
        # eg: --mask_n_blocks 3
        mask_config = {"mask_{}".format(key): value
                       for key, value in find_mask_model_class(mask_model_type).default_config().items()}
        # Parameters of the classifier model. Those are the parameters exposed by the default_config function of the
        # classifier model class. Command line argument arguments to modify those parameters should be prefixed by
        # 'class_' to differentiate with other parameters. eg: --class_n_blocks 4
        class_config = {"class_{}".format(key): value
                        for key, value in find_classifier_model_class(classifier_model_type).default_config().items()}

        # Return the parameters for all the parts of the separation model. Also include a boolean flag, to indicate
        # whether or not to use a re-scaler after the classifier model. see PredictionReScaler in classifier model.
        return {**pcen_config, **mask_config, **class_config, "rescale_classification": False}

    def __init__(self, config, input_shape, n_classes):
        r"""Constructor. Instantiate the pcen, mask and classifier model.

        Args:
            config (dict): Configuration dictionary containing parameters defined in the dict return by default_config()
            input_shape (tuple): Shape of one input to the model (does not include batch dimension):
                                 [channel, Frequency, Time]
            n_classes (int): Number of classes to predict.
        """

        super(SeparationModel, self).__init__()

        # Makes sure that the mask model last layer has as many channels as there are classes to predict
        # 1 channel <-> 1 mask <-> 1 class.
        config["mask_conv_o_c"][config['n_blocks']-1] = n_classes

        # If a trainable pcen is requested
        if config["train_pcen"]:
            # Multiple pcen layer are trained at the same time. Their output is used as different channels for the
            # input of the mask model. The pcen parameters are all initialized with default values (can not be
            # specified by user).
            if config["train_multi_pcen"]:
                # Multiple PCEN training
                self.pcen = MultiPCENlayer(config["n_multi_pcen"], config["train_pcen_eps"])
                # Change the mask model first layer input channels to account for the multiple pcen.
                config['mask_conv_i_c'][0] = config["n_multi_pcen"]
            else:
                # If training only once pcen layer, the pcen parameters can be updated by user.
                self.pcen = PCENLayer(config["train_pcen_per_band_param"], input_shape[-2],
                                      config["train_pcen_use_s"], config["train_pcen_s"],
                                      config["train_pcen_per_band_filter"],
                                      config["train_pcen_b"], config["train_pcen_a"],
                                      config["train_pcen_eps"])
        else:
            # Do not use pcen layer. Still instantiate the class member to None for compatibility.
            self.pcen = None

        # Instantiate mask model with the provided parameters. (removes the 'mask_' prefix to match mask model
        # expected parameters)
        self.mask_model = find_mask_model_class(config["mask_model_type"])(
            {key.replace('mask_', ''): value for key, value in config.items()})

        # Run a dummy variable through pcen and mask model to get the shape of the output
        x = torch.zeros((1,) + input_shape, requires_grad=False)  # (add batch dimension to provided input shape)
        if self.pcen is not None:
            x = self.pcen(x)
        x = self.mask_model(x)
        config["class_input_shape"] = tuple(x.shape)  # Input shape to the classifier model

        # Adapt classifier parameters to match mask model output shape
        if config["classifier_model_type"] == "ChannelWiseFC2d":
            # in_channels is the number of masks (channel dim). in_features is the flattened mask size.
            config["class_in_channels"], config["class_in_features"] = \
                config["class_input_shape"][1], config["class_input_shape"][2] * config["class_input_shape"][3]
        elif config["classifier_model_type"] == "ChannelWiseRNNClassifier":
            # num_channels is the number of masks (channel dim). input_size is the number of frequency bins in the mask.
            config["class_num_channels"], config["class_input_size"] = x.shape[1], x.shape[2]
        elif config["classifier_model_type"] == "DepthWiseCNNClassifier":
            # CNN classifier have a per-block architecture, so we need to set the parameters for each block.
            # conv_i_c and conv_o_c are the number of masks (n_classes).
            # conv_groups is set to n_classes too. This makes sure that the classification is made independently of
            # the classes.
            config["class_conv_i_c"] = [n_classes] * config["class_n_blocks"]
            config["class_conv_o_c"] = [n_classes] * config["class_n_blocks"]
            config["class_conv_groups"] = [n_classes] * config["class_n_blocks"]

        # Instantiate the classifier model. (removes the 'class_' prefix to match classifier model expected parameters)
        self.classifier_model = find_classifier_model_class(config["classifier_model_type"])(
            {key.replace('class_', ''): value for key, value in config.items()})

        # If a re-scaler of the predicted probability is required
        if config["rescale_classification"]:
            self.rescaler = PredictionReScaler(n_classes)
        else:
            self.rescaler = None

    def forward(self, x):
        r"""Implements the forward pass of the joint mask and classifier models (and trainable pcen if any).

        Args:
            x (torch.tensor): Input audio features. shape: [Batch, Channels=1, Frequency, Time]

        Returns:
            Predictions, masks. Predictions are the probabilities calculated by the classifier model. The masks are
            also returned.
        """

        # PCEN
        if self.pcen is not None:
            x = self.pcen(x)
        # Mask model
        x = self.mask_model(x)
        # Extract masks for output
        masks = x
        # Classifier model
        labels = self.classifier_model(x)
        # Re-scale the predicted probabilities.
        if self.rescaler is not None:
            labels = self.rescaler(labels)
        return labels.squeeze(), masks
