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
            "i_c": [1, 64, 64, 64, 64, 64],  # input channels
            "o_c": [64, 64, 64, 64, 64, 16],  # output channels
            "k_f": [3, 3, 3, 3, 3, 3],  # kernel size on frequency axis
            "k_t": [3, 3, 3, 3, 3, 3],  # kernel size on time axis
            "s_f": [1, 1, 1, 1, 1, 1],  # stride on frequency axis
            "s_t": [1, 1, 1, 1, 1, 1],  # stride on time axis
            "p_f": [1, 1, 1, 1, 1, 1],  # padding on frequency axis
            "p_t": [1, 1, 1, 1, 1, 1],  # padding on time axis
            "groups": [1, 1, 1, 1, 1, 1],  # Number of feature maps in each "group" (see pytorch conv parameter)
            "dropout_type": "1D",  # 1D or 2D (channel wise) dropout
            "dropout_probs": [0.3, 0.3, 0.3, 0.3, 0.3, 0.3],  # dropout probabilities
            "use_batch_norm": True,  # If True, batch norm is used between convolution and activation
            "activations": ["lr", "lr", "lr", "lr", "lr", "sig"]
        })
        return config

    def __init__(self, config):
        super(VGGLikeMaskModel, self).__init__()
        self.net = VGGLikeCNN(config)

    def forward(self, x):
        return self.net(x)
