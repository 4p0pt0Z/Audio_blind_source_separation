import torch.nn as nn
import torch.nn.functional as func


def find_model_class(model_type):
    """
        Get the class of a model from an string indentifier
    Args:
        model_type (str): Identifier for the model Class

    Returns:
        Class implementing the desired model
    """
    if model_type == "VGG_like_CNN":
        return VGG_like_CNN
    else:
        raise NotImplementedError("The model type " + model_type + " is not available.")


class VGG_like_CNN(nn.Module):
    """

    """

    @classmethod
    def default_config(cls):
        config = {
            "i_c": [1, 64, 64, 64, 64, 64],  # number of input channel to each convolution bloc
            "o_c": [64, 64, 64, 64, 64, 16],  # number of output channel for each convolution bloc
            "k_f": [5, 5, 5, 5, 5, 5],  # kernel size along the frequency axis
            "k_t": [3, 3, 3, 3, 3, 3],  # kernel size along the time axis
            "s_f": [1, 1, 1, 1, 1, 1],  # stride of each convolution along the frequency axis
            "s_t": [1, 1, 1, 1, 1, 1],  # stride of each convolution along the time axis
            "p_f": [0, 0, 0, 0, 0, 0],  # padding before each convolution along the frequency axis
            "p_t": [0, 0, 0, 0, 0, 0],  # padding before each convolution along the time axis
            "drop_out_probs": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # drop out probability during training for each bloc

            "use_batch_norm": True,  # If true, each bloc contains a batch normalization layer after the convolution

            "classification_mapping": "GMP"  # The classification mapping to use: {"GMP", "GAP", "GWRP"}
        }
        return config

    def __init__(self, config):
        super(VGG_like_CNN, self).__init__()

        self.n_blocs = len(config["i_c"])
        self.convolutions = nn.ModuleList([nn.Conv2d(config["i_c"][i], config["o_c"][i],
                                                     (config["k_f"][i], config["k_t"][i]),
                                                     (config["s_f"][i], config["s_t"][i]),
                                                     (config["p_f"][i], config["p_t"][i]))
                                           for i in range(self.n_blocs)])

        self.use_batch_norm = config["use_batch_norm"]
        if self.use_batch_norm:
            self.batch_norms = nn.ModuleList([nn.BatchNorm2d(config["o_c"][i]) for i in range(self.n_blocs)])

        self.activations = nn.ModuleList([nn.LeakyReLU() for _ in range(self.n_blocs - 1)] + [nn.Sigmoid()])

        self.dropouts = nn.ModuleList([nn.Dropout(config["drop_out_probs"][i]) for i in range(self.n_blocs)])

        self.c_map = config["classification_mapping"]

    def forward(self, x):
        for idx in range(self.n_blocs):
            x = self.convolutions[idx](x)
            if self.use_batch_norm:
                x = self.batch_norms[idx](x)
            x = self.dropouts[idx](x)  # Dropout before activation (for sigmoid) because it rescales with 1/p
            x = self.activations[idx](x)
        masks = x.detach().cpu().numpy()
        if self.c_map == "GMP":
            x = func.max_pool2d(x, (x.shape[2], x.shape[3])).squeeze()  # x.shape: (B, C, F, T)
        elif self.c_map == "GAP":
            x = func.avg_pool2d(x, (x.shape[2], x.shape[3])).squeeze()
        elif self.c_map == "GWRP":
            raise NotImplementedError("Coming up !")

        return x, masks
