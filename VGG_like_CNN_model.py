import torch.nn as nn

ACTIVATION_DICT = {"relu": nn.ReLU, "lr": nn.LeakyReLU, "sig": nn.Sigmoid}
NON_LINEARITY_DICT = {"lr": "leaky_relu", "sig": "sigmoid", "relu": "relu"}
PADDING_TYPE_DICT = {"zero": nn.ZeroPad2d, "reflection": nn.ReflectionPad2d}
DROPOUT_TYPE_DICT = {"1D": nn.Dropout, "2D": nn.Dropout2d}
POOLING_TYPE_DICT = {"max": nn.MaxPool2d, "avg": nn.AvgPool2d}


class VGGLikeCNN(nn.Module):
    @classmethod
    def default_config(cls):
        config = {
            "n_blocs": 1,
            "conv_i_c": [1],  # input channels
            "conv_o_c": [64],  # output channels
            "conv_k_f": [3],  # kernel size on frequency axis
            "conv_k_t": [3],  # kernel size on time axis
            "conv_s_f": [1],  # stride on frequency axis
            "conv_s_t": [1],  # stride on time axis
            "conv_pad_type": "zero",
            "conv_p_f": [0],  # padding on frequency axis
            "conv_p_t": [0],  # padding on time axis
            "conv_groups": [1],  # Number of feature maps in each "group"

            "pooling_type": "none",
            "pool_k_f": [0],
            "pool_k_t": [0],
            "pool_s_f": [0],
            "pool_s_t": [0],
            "pool_pad_type": "reflection",
            "pool_p_f": [0],
            "pool_p_t": [0],

            "dropout_type": "1D",  # 1D or 2D (channel wise) dropout
            "dropout_probs": [0.0],  # dropout probabilities
            "use_batch_norm": True,  # If True, batch norm is used between convolution and activation
            "activations": ["lr"]
        }
        return config

    def __init__(self, config):
        super(VGGLikeCNN, self).__init__()

        self.n_blocs = config["n_blocs"]

        self.conv_padding = nn.ModuleList(
            [PADDING_TYPE_DICT[config["conv_pad_type"]]((config["conv_p_f"][i], config["conv_p_f"][i],
                                                         config["conv_p_t"][i], config["conv_p_t"][i]))
             for i in range(config["n_blocs"])])  # padding is "left", "right", "top", "bottom"

        self.convolutions = nn.ModuleList([nn.Conv2d(config["conv_i_c"][i], config["conv_o_c"][i],
                                                     (config["conv_k_f"][i], config["conv_k_t"][i]),
                                                     (config["conv_s_f"][i], config["conv_s_t"][i]),
                                                     groups=config["conv_groups"][i])
                                           for i in range(config["n_blocs"])])
        for idx, conv in enumerate(self.convolutions):
            nn.init.kaiming_normal_(conv.weight, nonlinearity=NON_LINEARITY_DICT[config["activations"][idx]])

        self.use_batch_norm = config["use_batch_norm"]
        if self.use_batch_norm:
            self.batch_norms = nn.ModuleList([nn.BatchNorm2d(config["conv_o_c"][i]) for i in range(config["n_blocs"])])

        self.dropouts = nn.ModuleList([DROPOUT_TYPE_DICT[config["dropout_type"]](config["dropout_probs"][i])
                                       for i in range(config["n_blocs"])])

        self.activations = nn.ModuleList([ACTIVATION_DICT[key]() for key in config["activations"]])

        self.use_pooling = config["pooling_type"].lower() != "none"
        if self.use_pooling:
            self.pool_padding = nn.ModuleList(
                [PADDING_TYPE_DICT[config["pool_pad_type"]]((config["pool_p_f"][i], config["pool_p_f"][i],
                                                             config["pool_p_t"][i], config["pool_p_t"][i]))
                 for i in range(config["n_blocs"])])
            self.poolings = nn.ModuleList([POOLING_TYPE_DICT[config["pooling_type"]](
                (config["pool_k_f"][i], config["pool_k_t"][i]),
                (config["pool_s_f"][i], config["pool_s_t"][i])) for i in range(config["n_blocs"])])

    def forward(self, x):
        for idx in range(self.n_blocs):
            x = self.conv_padding[idx](x)
            x = self.convolutions[idx](x)
            if self.use_batch_norm:
                x = self.batch_norms[idx](x)
            x = self.dropouts[idx](x)  # Dropout before activation (for sigmoid) because it rescales with 1/p
            x = self.activations[idx](x)
            if self.use_pooling:
                x = self.pool_padding[idx](x)
                x = self.poolings[idx](x)
        return x
