import torch
import torch.nn as nn

# Dictionaries mapping command line arguments to corresponding layers in pytorch
ACTIVATION_DICT = {"relu": nn.ReLU, "lr": nn.LeakyReLU, "sig": nn.Sigmoid, "softmax": nn.Softmax2d}
NON_LINEARITY_DICT = {"lr": "leaky_relu", "sig": "sigmoid", "relu": "relu", "softmax": "sigmoid"}
PADDING_TYPE_DICT = {"zero": nn.ZeroPad2d, "reflection": nn.ReflectionPad2d}
DROPOUT_TYPE_DICT = {"1D": nn.Dropout, "2D": nn.Dropout2d}
POOLING_TYPE_DICT = {"max": nn.MaxPool2d, "avg": nn.AvgPool2d}


class BlockFreqTimeCNN(nn.Module):
    r"""Implements a CNN defined as a succession of blocks. Each block containing at least a convolution layer and a
        non-linearity, and potentially several other layers.

        Each block contains a convolution layer with its activation function. In addition, a block can include
        padding (before convolution), batch normalization, dropout and pooling.

        input -> n_blocks * [(padding) (coord-)convolution (batch-norm) dropout non-linearity (pooling)] -> output

        All hyper-parameters of the convolutions are exposed to user input: kernel size and stride in both directions,
        input channels and output, channels grouping.

        The amount and type of padding to perform on the input of the convolution layers is exposed to user input.
        Available padding types are defined in :const:`VGG_like_CNN_model.PADDING_TYPE_DICT`.

        1D dropout (randomly zeroing elements of the input) or 2D input (randomly zeroing entire channels of the
        input) is available.

        Max pooling or average pooling is available. The size of the pooling kernels and stride is exposed to user
        input.

        Instead of regular convolutions, a 'coordconvolution' can be used. It enables the network to use the
        information contained in the coordinates of the convolution - removing the translation invariance of
        convolution layers.
        A simple implementation is used here: the normalized coordinate of the frequency bins is stacked to the input
        of the convolution layer. Eg: input is [B, C, H, W], then output is [B, C+1, F, T] where the channel contains
        the normalized coordinates along the T axis.
        Rosanne Liu and (2018). An Intriguing Failing of Convolutional Neural Networks and the CoordConv. CoRR,
        abs/1807.03247, .

    """

    @classmethod
    def default_config(cls):
        r"""Provides a dictionary with all the tunable hyper-parameters of a block-defined CNN.

        Returns:
            A dictionary with entries for all the hyper-parameters.
        """

        config = {
            "n_blocks": 1,  # number of blocks. The list containing the parameters should have this length.

            "H_coord_conv": False,  # whether to use coordconvolution instead of regular convolutions in the F axis.

            "conv_i_c": [1],  # input channels
            "conv_o_c": [64],  # output channels
            "conv_k_f": [3],  # kernel size on frequency axis
            "conv_k_t": [3],  # kernel size on time axis
            "conv_s_f": [1],  # stride on frequency axis
            "conv_s_t": [1],  # stride on time axis
            "conv_pad_type": "zero",  # type of padding to use. see PADDING_TYPE_DICT.
            "conv_p_f": [0],  # padding on frequency axis (before convolution).
                              # If padding is not wanted, use [0].
                              # Same amount of padding is applied at both end of the axis.
            "conv_p_t": [0],  # padding on time axis (before convolution).
                              # If padding is not wanted, use 0.
                              # Same amount of padding is applied at both end of the axis.
            "conv_groups": [1],  # Number of feature maps in each "group" (default 1 means "no grouping")

            "pooling_type": "none",  # type of pooling after the convolution. see POOLING_TYPE_DICT.
                                     # To not use pooling, use "none".
            "pool_k_f": [0],  # kernel size on frequency axis
            "pool_k_t": [0],  # kernel size on time axis
            "pool_s_f": [0],  # kernel stride on frequency axis
            "pool_s_t": [0],  # kernel stride on time axis
            "pool_pad_type": "reflection",  # Padding type to use before pooling. see PADDING_TYPE_DICT
            "pool_p_f": [0],  # padding on frequency axis. (before pooling)
            "pool_p_t": [0],  # padding on time axis. (before pooling)

            "dropout_type": "1D",  # Whether to use 1D or 2D (channel wise) dropout. see DROPOUT_TYPE_DICT
            "dropout_probs": [0.0],  # dropout probabilities. If dropout is not wanted, use probability of 0.
            "use_batch_norm": True,  # If True, batch norm is used between convolution and non-linearity in the blocks
            "activations": ["lr"]  # non-linearity functions. see NON_LINEARITY_DICT
        }
        return config

    def __init__(self, config):
        r"""Constructor: initializes all blocks in the CNN.

            The layers in the network are grouped by type (convolution, padding, pooling...) in nn.ModuleLists. The
            index in the list indicates the block of the layer.

        Args:
            config (dict): configuration dictionary containing hyper-parameter values for all layers in the network.
                           config is expected to contain an entry for all parameters in the dict return by
                           default_config(). The parameters are passed as lists, with the value of the parameter in
                           each block.
        """

        super(BlockFreqTimeCNN, self).__init__()

        # Read the number of blocks. In the following, it is assumed that the parameters lists have the length n_blocks
        self.n_blocks = config["n_blocks"]

        # Whether we will be using coord-convolutions. If it is the case, we need to add +1 to the input channel
        # parameters of the convolutions, to account for the channel containing the coordinates.
        self.use_coord_conv = config["freq_coord_conv"]
        if self.use_coord_conv:
            config = dict(config)  # copy the dict so that changed values are not propagated
            for idx, i_c in enumerate(config["conv_i_c"]):
                config["conv_i_c"][idx] = i_c + 1

        # Padding (layer before the convolution)
        # padding layer expects "left", "right", "top", "bottom". We use same amount in left-right and top-bottom.
        self.conv_padding = nn.ModuleList(
            [PADDING_TYPE_DICT[config["conv_pad_type"]]((config["conv_p_t"][i], config["conv_p_t"][i],
                                                         config["conv_p_f"][i], config["conv_p_f"][i]))
             for i in range(config["n_blocks"])])

        # Convolution
        self.convolutions = nn.ModuleList([nn.Conv2d(config["conv_i_c"][i], config["conv_o_c"][i],
                                                     (config["conv_k_f"][i], config["conv_k_t"][i]),
                                                     (config["conv_s_f"][i], config["conv_s_t"][i]),
                                                     groups=config["conv_groups"][i])
                                           for i in range(config["n_blocks"])])
        # Initialize the convolutions weights depending on the non-linearity.
        for idx, conv in enumerate(self.convolutions):
            nn.init.kaiming_normal_(conv.weight, nonlinearity=NON_LINEARITY_DICT[config["activations"][idx]])

        # Batch-normalization
        self.use_batch_norm = config["use_batch_norm"]
        if self.use_batch_norm:
            self.batch_norms = nn.ModuleList([nn.BatchNorm2d(config["conv_o_c"][i]) for i in range(config["n_blocks"])])

        # Dropout
        self.dropouts = nn.ModuleList([DROPOUT_TYPE_DICT[config["dropout_type"]](config["dropout_probs"][i])
                                       for i in range(config["n_blocks"])])

        # Non-linearity
        self.activations = nn.ModuleList([ACTIVATION_DICT[key]() for key in config["activations"]])

        # Pooling
        # If pooling is not wanted, use pooling_type ) "none".
        # If pooling is used, provides the possibility to pad the input before the pooling.
        self.use_pooling = config["pooling_type"].lower() != "none"
        if self.use_pooling:
            # padding layer expects "left", "right", "top", "bottom". We use same amount in left-right and top-bottom.
            self.pool_padding = nn.ModuleList(
                [PADDING_TYPE_DICT[config["pool_pad_type"]]((config["pool_p_f"][i], config["pool_p_f"][i],
                                                             config["pool_p_t"][i], config["pool_p_t"][i]))
                 for i in range(config["n_blocks"])])
            # Pooling
            self.poolings = nn.ModuleList([POOLING_TYPE_DICT[config["pooling_type"]](
                (config["pool_k_f"][i], config["pool_k_t"][i]),
                (config["pool_s_f"][i], config["pool_s_t"][i])) for i in range(config["n_blocks"])])

    def forward(self, x):
        r"""Implements the network forward pass.

            input -> n_blocks * [(padding) (coord-)convolution (batch-norm) dropout non-linearity (pooling)] -> output

            The pass is a loop over the blocks.

        Args:
            x (torch.tensor): input batch tensor with shape [B, C, F, T] (batch, channel, Frequency, Time)

        Returns:
            Output of the network.
        """

        for idx in range(self.n_blocks):
            x = self.conv_padding[idx](x)
            # add frequency bin coordinates (normalized) as a channel if using coord-convolution
            if self.use_coord_conv:
                x = torch.cat((x, torch.arange(x.shape[2], dtype=x.dtype, device=x.device)
                               .unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(x.shape[0], -1, x.shape[2], x.shape[3])
                               / x.shape[2]), 1)
            x = self.convolutions[idx](x)
            if self.use_batch_norm:
                x = self.batch_norms[idx](x)
            x = self.dropouts[idx](x)  # Dropout before activation because it rescales with 1/p (bad after sigmoid)
            x = self.activations[idx](x)
            if self.use_pooling:
                x = self.pool_padding[idx](x)
                x = self.poolings[idx](x)
        return x
