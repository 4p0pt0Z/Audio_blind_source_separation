import torch
import torch.nn as nn

from VGG_like_CNN_model import BlockFreqTimeCNN, ACTIVATION_DICT


def find_classifier_model_class(model_type):
    r"""Helper function returning the class (in the python sense) corresponding to a model identified by a string.

        For simplicity, the identifiers are often the name of the classes.
    Args:
        model_type (str): Identifier of a model class

    Returns:
        The python class corresponding to the identifier

    Examples:
        >>> config = find_classifier_model_class("GlobalWeightedRankPooling2d").default_config()
    """

    if model_type == "GlobalMaxPooling2d":
        return GlobalMaxPooling2d
    elif model_type == "GlobalAvgPooling2d":
        return GlobalAvgPooling2d
    elif model_type == "GlobalWeightedRankPooling2d":
        return GlobalWeightedRankPooling2d
    elif model_type == "AdaptiveGlobalWeightedRankPooling2d":
        return AdaptiveGlobalWeightedRankPooling2d
    elif model_type == "ChannelWiseFC2d":
        return ChannelWiseFC2d
    elif model_type == "DepthWiseCNNClassifier":
        return DepthWiseCNNClassifier
    elif model_type == "ChannelWiseRNNClassifier":
        return ChannelWiseRNNClassifier
    else:
        raise NotImplementedError("Classifier model " + model_type + " is not available !")


class GlobalMaxPooling2d(nn.Module):
    r"""Implements a global max pooling operation on 2D inputs as a PyTorch module layer.

        Input shape: (N, C, H, W)
        Output shape: (N, C, 1, 1)

        output[n, c, 0, 0] =  input[n, c, h, w].max(dim=2).max(dim=2)

    """

    @classmethod
    def default_config(cls):
        r"""Global max pooling has no configurable parameters

        Returns:
            Empty dictionary
        """

        return {}

    def __init__(self, config=None):
        r"""Constructor. Does not require any parameter but takes 'config' for compatibility with other classifiers.

        Args:
            config (dict): Not used. Only present for compatibility.
        """

        super(GlobalMaxPooling2d, self).__init__()

    def forward(self, x):
        return nn.functional.max_pool2d(x, (x.shape[2], x.shape[3]))


class GlobalAvgPooling2d(nn.Module):
    r"""Implements a global average pooling operation on 2D inputs as a Pytorch module layer.

        Input shape: (N, C, H, W)
        Output shape: (N, C, 1, 1)


        output[n, c, 0, 0] = input[n, c, h, w].mean(dim2).mean(dim=2)
    """

    @classmethod
    def default_config(cls):
        r"""Global average pooling has no configurable parameters

        Returns:
            Empty dictionary
        """

        return {}

    def __init__(self, config=None):
        r"""Constructor. Does not required any parameter by takes 'config' for compatibility with other classifier.

        Args:
            config (dict): Not used. Only present for compatibility.
        """

        super(GlobalAvgPooling2d, self).__init__()

    def forward(self, x):
        return nn.functional.avg_pool2d(x, (x.shape[2], x.shape[3]))


class GlobalWeightedRankPooling2d(nn.Module):
    r"""Implements a global weighted rank pooling as defined on 2D inputs as defined in Qiuquiang Kong et Al. "A
        joint separation-classification model for sound event detection of weakly labelled data". In: CoRR
        abs/1711.03037 (2017). arXiv: 1711.03037. URL:http://arxiv.org/abs/1711.03037.

        Input shape: (N, C, H, W)
        Output shape: (N, C)

        sorted = sorted_in_descending_order(input)
        output[n, c, 0, 0] = (1/Z) * sum {j=0 to H*W} sorted[j] * d_c**j

        Z = sum {j=0 to H*W} (d_c**j)

        d_c is parametrizing the weights.
    """

    @classmethod
    def default_config(cls):
        r"""The pooling depend on the parameter dc.

        Returns:
            A dictionary with the default value for the parameter dc.
        """

        return {"dc": 0.999}

    def __init__(self, config):
        r"""Constructor: initializes the parameter dc.

        Args:
            config (dict): Expected to contain an entry for the key 'dc' with a floating point number in
            [0.0, 1.0] as value.
        """

        super(GlobalWeightedRankPooling2d, self).__init__()
        self.dc = config["dc"]

    def forward(self, x):
        # Flatten the height and width axis
        x = x.view(x.shape[0], x.shape[1], -1)
        # Sort each input[n, c, :] in descending order
        x, _ = torch.sort(x, dim=2, descending=True)
        # Build a tensor for the weights: dc**(j-1)
        weights = torch.tensor([self.dc ** j for j in range(x.shape[2])], device=x.device)
        # Compute the term Z.
        norm_term = torch.sum(weights)
        # Compute global weighted rank pooling for each channel.
        y = torch.zeros((x.shape[0], x.shape[1]), device=x.device)
        for c in range(x.shape[1]):
            y[:, c] = torch.sum(x[:, c, :] * weights, dim=1) / norm_term
        return y


class AdaptiveGlobalWeightedRankPooling2d(nn.Module):
    r""""Implements a learning version of a global weighted rank pooling operation on 2D inputs (see
        GlobalWeightedRankPooling2d)

        In this version, there are as many parameters dc as there are input channels, and these parameters are
        independently updated during training.
    """

    @classmethod
    def default_config(cls):
        r"""This class uses as many parameters as there are channels in its input.

        Returns:
            A dictionary containing a **list** of values for the parameter dc.
        """

        return {"dc": [0.9]}

    def __init__(self, config):
        r"""Constructor: initializes the parameters dc.

        Args:
            config (dict): Expected to contain an entry for the key 'dc', with value a list of floats for the initial
            value of dc for each input channel
        """

        super(AdaptiveGlobalWeightedRankPooling2d, self).__init__()
        dc = torch.tensor(config["dc"])
        # To ensure dc values stay in [0, 1] during optimization, sigmoid will be applied on it.
        # Therefore: convert the given values to their inverse with respect to sigmoid
        self.dc = nn.Parameter(torch.log(dc / (1 - dc)))

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], -1)
        x, _ = torch.sort(x, dim=2, descending=True)
        y = torch.zeros((x.shape[0], x.shape[1]), device=x.device)
        for c in range(x.shape[1]):
            channel_dc = self.dc if self.dc.shape[0] == 1 else self.dc[c]  # TODO: optimize
            weights = torch.sigmoid(torch.tensor([channel_dc ** j
                                                  for j in range(x.shape[2])], device=x.device))
            y[:, c] = torch.sum(x[:, c, :] * weights, dim=1) / torch.sum(weights)
        return y


class ChannelWiseFC2d(nn.Module):
    r"""Implements as a PyTorch module a group of fully connected layer, each acting independently over a channel of
        the input.

        If the input shape is (N, C, X), then C fully connected layers are used. Each layer takes as input a vector
        of size X, one of the C channels in the input.
        If required, the input vectors (of size X) can be sorted in descending order before applying the
        Fully-connected layers.

        Input shape: (N, C, H, W)
        Output shape: (N, C, O) where O is the output shape of the fully connected layers.
    """

    @classmethod
    def default_config(cls):
        config = {
            "in_channels": 1,
            "in_features": 1,  # Number of input features
            "out_features": 1,
            "use_bias": True,
            "activation": "sig",

            "sort": False
        }
        return config

    def __init__(self, config):
        r"""Constructor: initializes all the Fully-connected layers.

        Args:
            config (dict): Dictionary expected to have entries for the keys:
                           - 'in_features'. Value should be an int and correspond to the size X of the input to each
                           fully connected layer.
                           - 'out_features'. Value should be an int. It is the size of the output of each fully
                           connected layer.
                           - 'in_channels'. Value should be an int. The number of channels in the input.
                           config['in_channels'] fully connected layers will be instantiated.
                           - 'sort'. Value should be a boolean. If True, the input is sorted in descending order
                           (each channel is sorted separately) before passing through the fully connected layers.
                                     If False, the input is passed directly through the fully connected layers.
                           - 'use_bias'. Value should be an boolean. Whether to use a bias parameter in the fully
                           connected layers.
                           - 'activation'. Value should be a string, identifying a non-linearity function to use
                           after the fully connected layers. Possible activation function are defined in
                           :const:`VGG_like_CNN_model.VGGLikeCNN.ACTIVATION_DICT` and include sigmoid, softmax,
                           leaky-ReLU...
        """

        super(ChannelWiseFC2d, self).__init__()
        self.in_features = config["in_features"]
        self.out_features = config["out_features"]
        self.in_channels = config["in_channels"]
        self.sort = config["sort"]
        # Create as many fully connected layers as there are channels in the input.
        self.FCs = nn.ModuleList([nn.Linear(self.in_features, self.out_features, config["use_bias"])
                                  for _ in range(config["in_channels"])])
        self.activation = ACTIVATION_DICT[config["activation"]]()

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], -1)
        y = torch.zeros((x.shape[0], self.in_channels, self.out_features), device=x.device)
        for channel in range(self.in_channels):
            if self.sort:
                z, _ = torch.sort(x[:, channel, :], dim=1, descending=True)
                y[:, channel, :] = self.FCs[channel](z)
            else:
                y[:, channel, :] = self.FCs[channel](x[:, channel, :])
        y = self.activation(y)
        return y


class DepthWiseCNNClassifier(nn.Module):
    r"""Implements a classifier as a convolutional neural network followed by a channel wise fully connected layer.

        The CNN part of the classifier is defined per block and uses the VGGLikeCNN class :class:VGGLikeCNN. It is
        designed to let the user free to tune each parameter of the CNN for easy prototyping.

        Input shape: (N, C, H, W)
        Output shape: (N, C, O) where O is the output size of the fully connected layers.
    """

    @classmethod
    def default_config(cls):
        r"""Provides a dictionary with keys to all the tunable parameters of the CNN and ChannelWiseFC part of the
            classifier.

            Hyper-parameters for the CNN are taken from the :meth:`VGGLikeCNN.default_config` method. Specific default
            values to use a CNN for *classification* are set here. (All values can be changed by user by command line).

            The input dimension of the ChannelWiseFC2d will be defined by the output shape of the CNN part and will
            be computed in the instantiation of the class. The other parameters can be set using the keys of the
            ChannelWiseFC2d default_config method :meth:`ChannelWiseFC2d:default_config` with the prefix 'FC_'.


        Returns:
            A dictionary with the default values for all the hyper-parameters for the CNN and channel wise fully
            connected layers.
        """

        # Get a dictionary with all hyper-parameters as keys from the CNN base class.
        config = BlockFreqTimeCNN.default_config()
        # Update it with specific classifier hyper-parameter values
        config.update({
            "n_blocs": 1,

            "freq_coord_conv": False,

            "conv_i_c": [16],  # input channels
            "conv_o_c": [16],  # output channels
            "conv_k_f": [5],  # kernel size on frequency axis
            "conv_k_t": [5],  # kernel size on time axis
            "conv_s_f": [3],  # stride on frequency axis
            "conv_s_t": [3],  # stride on time axis
            "conv_pad_type": "zero",
            "conv_p_f": [0],  # padding on frequency axis
            "conv_p_t": [0],  # padding on time axis

            "conv_groups": [16],  # Number of feature maps in each "group"

            "pooling_type": "None",
            "pool_k_f": [0],
            "pool_k_t": [0],
            "pool_s_f": [0],
            "pool_s_t": [0],
            "pool_pad_type": "reflection",
            "pool_p_f": [0],
            "pool_p_t": [0],

            "dropout_type": "1D",  # 1D or 2D (channel wise) dropout
            "dropout_probs": [0.0],  # dropout probabilities
            "use_batch_norm": False,  # If True, batch norm is used between convolution and activation
            "activations": ["lr"]
        })
        # Add to the dictionary the entries for ChannelWiseFC2d parameters. Use 'FC_' prefix to ensure parameters of
        # the CNN and the fully connected layers do not have the same names.
        channelwisefc2d_config = {"FC_{}".format(key): value for key, value in ChannelWiseFC2d.default_config().items()}
        channelwisefc2d_config.update({
            "FC_out_features": 1,
            "FC_use_bias": True,
            "FC_activation": "sig",
            "FC_sort": False
        })
        return {**config, **channelwisefc2d_config}

    def __init__(self, config):
        r"""Constructor. Initializes the CNN and the fully connected layers.

        Args:
            config (dict): Configuration dictionary containing the values of all the hyper-parameters in the dict
            returned by default_config().
        """

        super(DepthWiseCNNClassifier, self).__init__()
        # Instantiate the CNN part
        self.CNN = BlockFreqTimeCNN(config)
        # Run a dummy variable through the CNN to get the shape of its output.
        input_shape_FC = self.CNN(torch.zeros(*config["input_shape"])).shape
        # The shape of the CNN output defines the input sizes for the fully connected layers:
        config["FC_in_channels"], config["FC_in_features"] = input_shape_FC[1], input_shape_FC[2] * input_shape_FC[3]
        # Remove the 'FC_' prefix to build the configuration dictionary for the ChannelWiseFC2d class.
        self.FC = ChannelWiseFC2d({key.replace('FC_', ''): value for key, value in config.items()})

    def forward(self, x):
        x = self.CNN(x)
        x = self.FC(x.view(x.shape[0], x.shape[1], -1))
        return x


class ChannelWiseRNNClassifier(nn.Module):
    r"""Implements a recurrent neural network, to process an 2D input with one dimension representing time and the other
        frequency (like a spectrogram). It is followed by a ChannelWiseFC2d, acting on the last hidden state of the
        RNN for each channel.

        Input shape: (N, C, F, T). F is the number of frequency bins, T is the number of time steps.
        Output shape: (N, N, O) where O is the output shape of the fully connected layers.

        Note: PyTorch RNN usually expect the time dimension to be the 3rd dimension. However, when computing
        spectrograms with scipy or librosa, the time dimension is last, and this convention has been chosen here.
    """

    @classmethod
    def default_config(cls):
        r"""Provides a dictionary for all the tunable parameters of the RNN and the FC part of the classifier.

            The fully connected input parameters will be defined by the shape of the output of the RNN.

        Returns:
            A dictionary with the default values for all the hyper-parameters for the RNN and channel wise FC layers.
        """

        config = {
            "num_channels": 16,
            "RNN_cell_type": "GRU",  # Type of RNN: GRU, LSTM, Elman RNN (see pytorch Recurrent layer documentation)
            "input_size": 64,
            "hidden_size": 40,
            "num_layers": 1,
            "use_bias": True,
            "batch_first": True,
            "dropout": 0,
            "bidirectional": False
        }
        # Add to the dictionary the entries for ChannelWiseFC2d parameters. Use 'FC_' prefix to ensure parameters of
        # the CNN and the fully connected layers do not have the same names.
        channelwisefc2d_config = {"FC_{}".format(key): value for key, value in ChannelWiseFC2d.default_config().items()}
        channelwisefc2d_config.update({
            "FC_out_features": 1,
            "FC_use_bias": True,
            "FC_activation": "sig",
            "FC_sort": False
        })
        return config

    def __init__(self, config):
        r"""Constructor. Initializes the RNN and the fully connected layers.

        Args:
            config (dict): Configuration dictionary containing the values of all the hyper-parameter in the dict
            returned by default_config().
        """

        super(ChannelWiseRNNClassifier, self).__init__()
        self.hidden_size = config["hidden_size"]
        self.n_channels = config["num_channels"]

        cell_type_dict = {"GRU": nn.GRU, "LSTM": nn.LSTM, "RNN": nn.RNN}
        self.RNN = cell_type_dict[config["RNN_cell_type"]](input_size=config["input_size"],
                                                           hidden_size=config["hidden_size"],
                                                           num_layers=config["num_layers"],
                                                           bias=config["use_bias"],
                                                           batch_first=config["batch_first"],
                                                           dropout=config["dropout"],
                                                           bidirectional=config["bidirectional"])

        self.ChannelWiseFC = ChannelWiseFC2d({"in_channels": config["num_channels"],
                                              "in_features": config["hidden_size"],
                                              "out_features": config['FC_out_features'],
                                              "use_bias": config['FC_use_bias'],
                                              "activation": config['FC_activation'],
                                              "sort": config['FC_sort']})

    def forward(self, x):
        # permute time and frequency axis, to have axis in position required by PyTorch RNNs
        x = x.permute(0, 1, 3, 2)
        y = torch.zeros((x.shape[0], x.shape[1], self.hidden_size), device=x.device)
        for channel in range(self.n_channels):
            _, y[:, channel, :] = self.RNN(x[:, channel, :, :])
        y = self.ChannelWiseFC(y)
        return y


class PredictionReScaler(nn.Module):
    r"""Learns a single weight to rescale the output of a biased classifier.

        Global pooling classifiers are biased with respect to the classes, because the value of their activation is
        bounded by the typical ratio between the "area" occupied by an audio event with respect to the total area of
        a spectrogram. To correct for this, a re-scaler can be used. It learns 1 weight per class to map the
        probability of a classifier to a different average.

        input shape: [C] where C is the number of classes.
        input -> sigmoid(w * input) where w is a learned weight.
    """

    def __init__(self, input_shape):
        super(PredictionReScaler, self).__init__()
        # Initialize the weights around 1
        self.rescaling_weights = nn.Parameter(torch.randn(input_shape) * torch.log(torch.tensor(1 / input_shape)) + 1.0)

    def forward(self, x):
        return torch.sigmoid(x * self.rescaling_weights)
