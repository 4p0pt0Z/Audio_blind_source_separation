import torch.nn as nn


ACTIVATION_DICT = {"lr": nn.LeakyReLU, "sig": nn.Sigmoid}
INIT_DICT = {"lr": "leaky_relu", "sig": "sigmoid"}


class VGGLikeCNN(nn.Module):
    @classmethod
    def default_config(cls):
        config = {
            "n_blocs": 1,
            "i_c": [1],  # input channels
            "o_c": [64],  # output channels
            "k_f": [3],  # kernel size on frequency axis
            "k_t": [3],  # kernel size on time axis
            "s_f": [1],  # stride on frequency axis
            "s_t": [1],  # stride on time axis
            "p_f": [0],  # padding on frequency axis
            "p_t": [0],  # padding on time axis
            "groups": [1],  # Number of feature maps in each "group" (see pytorch conv parameter)
            "dropout_type": "1D",  # 1D or 2D (channel wise) dropout
            "dropout_probs": [0.0],  # dropout probabilities
            "use_batch_norm": True,  # If True, batch norm is used between convolution and activation
            "activations": ["lr"]
        }
        return config

    def __init__(self, config):
        super(VGGLikeCNN, self).__init__()

        self.n_blocs = config["n_blocs"]

        self.convolutions = nn.ModuleList([nn.Conv2d(config["i_c"][i], config["o_c"][i],
                                                     (config["k_f"][i], config["k_t"][i]),
                                                     (config["s_f"][i], config["s_t"][i]),
                                                     (config["p_f"][i], config["p_t"][i]),
                                                     groups=config["groups"][i])
                                           for i in range(config["n_blocs"])])
        for idx, conv in enumerate(self.convolutions):
            nn.init.kaiming_normal_(conv.weight, nonlinearity=INIT_DICT[config["activations"][idx]])

        self.use_batch_norm = config["use_batch_norm"]
        if self.use_batch_norm:
            self.batch_norms = nn.ModuleList([nn.BatchNorm2d(config["o_c"][i]) for i in range(config["n_blocs"])])

        self.activations = nn.ModuleList([ACTIVATION_DICT[key]() for key in config["activations"]])

        if config["dropout_type"] == "1D":
            self.dropouts = nn.ModuleList([nn.Dropout(config["dropout_probs"][i]) for i in range(config["n_blocs"])])
        elif config["dropout_type"] == "2D":
            self.dropouts = nn.ModuleList([nn.Dropout2d(config["dropout_probs"][i]) for i in range(config["n_blocs"])])

    def forward(self, x):
        for idx in range(self.n_blocs):
            x = self.convolutions[idx](x)
            if self.use_batch_norm:
                x = self.batch_norms[idx](x)
            x = self.dropouts[idx](x)  # Dropout before activation (for sigmoid) because it rescales with 1/p
            x = self.activations[idx](x)
        return x