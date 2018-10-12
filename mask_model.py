import torch.nn as nn
from VGG_like_CNN_model import VGGLikeCNN


def find_mask_model_class(model_type):
    """
        Get the class of a model from an string indentifier
    Args:
        model_type (str): Identifier for the model Class

    Returns:
        Class implementing the desired model
    """
    if model_type == "VGGLikeMaskModel":
        return VGGLikeMaskModel
    else:
        raise NotImplementedError("The mask segmentation model type " + model_type + " is not available.")


class VGGLikeMaskModel(nn.Module):

    @classmethod
    def default_config(cls):
        config = VGGLikeCNN.default_config()
        config.update({
            "n_blocs": 6,
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
        super(VGGLikeMaskModel, self).__init__()
        config["activations"][config["n_blocs"]-1] = "sig"  # last activations set to sigmoid
        config["dropout_probs"][config["n_blocs"]-1] = 0.0  # dropout on "masks" is 0.0
        self.net = VGGLikeCNN(config)

    def forward(self, x):
        return self.net(x)
