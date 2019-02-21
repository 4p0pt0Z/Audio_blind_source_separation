import torch.nn as nn
from VGG_like_CNN_model import BlockFreqTimeCNN


def find_mask_model_class(model_type):
    """Helper function returning the class (in the python sense) corresponding to a mask model identified by a string.

    Args:
        model_type (str): Identifier for the model Class

    Returns:
        Class implementing the desired model
    """

    if model_type == "VGGLikeMaskModel":
        return VGGLikeMaskModel
    else:  # Only per-block CNN available so far.
        raise NotImplementedError("The mask segmentation model type " + model_type + " is not available.")


class VGGLikeMaskModel(nn.Module):
    r"""Implements the mask model as a convolutional neural network.

        The mask model is a CNN, its architecture is a stack of blocks, each containing a convolution layer (after
        padding), eventually a batch normalization layer, a dropout layer, an activation function and eventually a
        pooling layer.

        The implementation of a block CNN is done in :class:`VGG_like_CNN_model.VGGLikeCNN`. This class mainly
        defines suitable values for the CNN hyper-parameters when used to produce a segmentation mask.
    """

    @classmethod
    def default_config(cls):
        r"""Provides a dictionary with all the tunable hyper-parameters of the mask model.

            This function gets the hyper-parameters from :class:`VGG_like_CNN_model.VGGLikeCNN` and simply updates
            their values to defaults that are more suited for a model used to produce a segmentation mask.

        Returns:
            A dictionary with the default values for all the hyper-parameters for a CNN used for producing
            segmentation masks.
        """

        config = BlockFreqTimeCNN.default_config()
        config.update({
            "n_blocks": 6,

            "freq_coord_conv": False,

            "conv_i_c": [1, 64, 64, 64, 64, 64],  # input channels
            "conv_o_c": [64, 64, 64, 64, 64, 16],  # output channels
            "conv_k_f": [3, 3, 3, 3, 3, 3],  # kernel size on frequency axis
            "conv_k_t": [3, 3, 3, 3, 3, 3],  # kernel size on time axis
            "conv_s_f": [1, 1, 1, 1, 1, 1],  # stride on frequency axis
            "conv_s_t": [1, 1, 1, 1, 1, 1],  # stride on time axis
            "conv_pad_type": "zero",
            "conv_p_f": [1, 1, 1, 1, 1, 1],  # padding on frequency axis
            "conv_p_t": [1, 1, 1, 1, 1, 1],  # padding on time axis

            "conv_groups": [1, 1, 1, 1, 1, 1],  # Number of feature maps in each "group" (see pytorch conv parameter)

            "pooling_type": "none",
            "pool_k_f": [0],
            "pool_k_t": [0],
            "pool_s_f": [0],
            "pool_s_t": [0],
            "pool_pad_type": "reflection",
            "pool_p_f": [0],
            "pool_p_t": [0],

            "dropout_type": "1D",  # 1D or 2D (channel wise) dropout
            "dropout_probs": [0.3, 0.3, 0.3, 0.3, 0.3, 0.3],  # dropout probabilities
            "use_batch_norm": True,  # If True, batch norm is used between convolution and activation
            "activations": ["lr", "lr", "lr", "lr", "lr", "sig"]
        })
        return config

    def __init__(self, config):
        r"""Constructor: initializes the CNN layers.

            A mask model final activation should be either sigmoid or softmax. It is silently imposed here that the
            final block activation be sigmoid, except when user specifically asks for softmax.
            The output values of the mask model should not be influenced by dropout (to not mess up with the
            classifier model input), so the dropout probability of the last block is set to 0.

        Args:
            config (dict): configuration
        """

        super(VGGLikeMaskModel, self).__init__()
        if config["activations"][config["n_blocks"] - 1] != 'softmax':
            config["activations"][config["n_blocks"] - 1] = "sig"  # last activations set to sigmoid
        config["dropout_probs"][config["n_blocks"] - 1] = 0.0  # dropout on "masks" is 0.0
        self.net = BlockFreqTimeCNN(config)

    def forward(self, x):
        return self.net(x)
