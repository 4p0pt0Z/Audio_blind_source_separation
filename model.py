import torch
import torch.nn as nn


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


class ChannelWiseFC(nn.Module):
    def __init__(self, in_channels, in_features, out_features, bias=True):
        super(ChannelWiseFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.in_channels = in_channels
        self.FCs = nn.ModuleList([nn.Linear(in_features, out_features, bias) for _ in range(in_channels)])

    def forward(self, x):
        y = torch.zeros((x.shape[0], self.in_channels, self.out_features), device=x.device)
        for channel in range(self.in_channels):
            y[:, channel, :] = self.FCs[channel](x[:, channel, :])
        return y


class GlobalWeightedRankPooling(nn.Module):
    def __init__(self, dc, size):
        super(GlobalWeightedRankPooling, self).__init__()
        self.register_buffer('dcs', torch.Tensor([dc ** j for j in range(size)]))
        self.register_buffer('Z', torch.sum(self.dcs))

    def forward(self, x):
        x, _ = torch.sort(x, dim=2, descending=True)
        y = torch.zeros((x.shape[0], x.shape[1]), device=x.device)
        for c in range(x.shape[1]):
            y[:, c] = torch.sum(x[:, c, :] * self.dcs, dim=1) / self.Z
        return y


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
            "groups": [1, 1, 1, 1, 1, 1],  # groups parameter for each conv layer.
            "drop_out_probs": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # drop out probability during training for each bloc

            "leaky_relu_slope": 0.01,

            "use_batch_norm": True,  # If true, batch normalization is used between the convolutions and activations

            "classification_mapping": "GAP",  # The classification mapping to use: {"GMP", "GAP", "GWRP"}
            "dc": 0.5  # hyper-parameter for global weighted pooling
        }
        return config

    def __init__(self, config, input_shape):
        super(VGG_like_CNN, self).__init__()

        self.n_blocs = len(config["i_c"])
        self.convolutions = nn.ModuleList([nn.Conv2d(config["i_c"][i], config["o_c"][i],
                                                     (config["k_f"][i], config["k_t"][i]),
                                                     (config["s_f"][i], config["s_t"][i]),
                                                     (config["p_f"][i], config["p_t"][i]))
                                           for i in range(self.n_blocs)])

        for conv in self.convolutions:
            nn.init.kaiming_normal_(conv.weight, a=config["leaky_relu_slope"], nonlinearity="leaky_relu")

        self.use_batch_norm = config["use_batch_norm"]
        if self.use_batch_norm:
            self.batch_norms = nn.ModuleList([nn.BatchNorm2d(config["o_c"][i]) for i in range(self.n_blocs)])

        self.activations = nn.ModuleList([nn.LeakyReLU(config["leaky_relu_slope"])
                                          for _ in range(self.n_blocs - 1)] + [nn.Sigmoid()])

        self.dropouts = nn.ModuleList([nn.Dropout2d(config["drop_out_probs"][i]) for i in range(self.n_blocs)])

        # Run a variable through the convolutions to get the shape of the input to the classification mapping
        example_input = torch.zeros((1, input_shape[0], input_shape[1], input_shape[2]))  # add batch dimension
        for idx in range(self.n_blocs):
            example_input = self.convolutions[idx](example_input)
            example_input = self.activations[idx](example_input)

        self.mask_shape = (example_input.shape[-2], example_input.shape[-1])

        if config["classification_mapping"] == "GMP":
            self.c_map = nn.MaxPool2d(self.mask_shape)
            self.reshape = False
        elif config["classification_mapping"] == "GAP":
            self.c_map = nn.AvgPool2d(self.mask_shape)
            self.reshape = False
        elif config["classification_mapping"] == "GWRP":
            self.c_map = GlobalWeightedRankPooling(config["dc"], example_input.shape[2] * example_input.shape[3])
            self.reshape = True
        elif config["classification_mapping"] == "FC":
            self.c_map = nn.Sequential(ChannelWiseFC(example_input.shape[1],
                                                     example_input.shape[2] * example_input.shape[3], 1),
                                       nn.Sigmoid())
            self.reshape = True

    def forward(self, x):
        for idx in range(self.n_blocs):
            x = self.convolutions[idx](x)
            if self.use_batch_norm:
                x = self.batch_norms[idx](x)
            x = self.dropouts[idx](x)  # Dropout before activation (for sigmoid) because it rescales with 1/p
            x = self.activations[idx](x)

        masks = x.detach().cpu().numpy()
        if self.reshape:
            x = x.view(x.shape[0], x.shape[1], -1)
        x = self.c_map(x).squeeze()  # squeeze to remove reduced dimensions
        return x, masks
