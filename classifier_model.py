import torch
import torch.nn as nn

from VGG_like_CNN_model import VGGLikeCNN, ACTIVATION_DICT


def find_classifier_model_class(model_type):
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
    @classmethod
    def default_config(cls):
        return {}

    def __init__(self, config=None):
        super(GlobalMaxPooling2d, self).__init__()

    def forward(self, x):
        return nn.functional.max_pool2d(x, (x.shape[2], x.shape[3]))


class GlobalAvgPooling2d(nn.Module):
    @classmethod
    def default_config(cls):
        return {}

    def __init__(self, config=None):
        super(GlobalAvgPooling2d, self).__init__()

    def forward(self, x):
        return nn.functional.avg_pool2d(x, (x.shape[2], x.shape[3]))


class GlobalWeightedRankPooling2d(nn.Module):
    @classmethod
    def default_config(cls):
        return {"dc": 0.999}

    def __init__(self, config):
        super(GlobalWeightedRankPooling2d, self).__init__()
        self.dc = config["dc"]

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], -1)
        x, _ = torch.sort(x, dim=2, descending=True)
        y = torch.zeros((x.shape[0], x.shape[1]), device=x.device)
        weights = torch.tensor([self.dc ** j for j in range(x.shape[2])], device=x.device)
        norm_term = torch.sum(weights)
        for c in range(x.shape[1]):
            y[:, c] = torch.sum(x[:, c, :] * weights, dim=1) / norm_term
        return y


class AdaptiveGlobalWeightedRankPooling2d(nn.Module):
    @classmethod
    def default_config(cls):
        return {"dc": [0.9]}

    def __init__(self, config):
        super(AdaptiveGlobalWeightedRankPooling2d, self).__init__()
        dc = torch.tensor(config["dc"])
        # To ensure dc values stay in [0, 1] during optimization, sigmoid will be applied on it.
        # Therefore: convert the given values to their fiber with respect to sigmoid
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
    @classmethod
    def default_config(cls):
        config = {
            "in_channels": 1,
            "in_features": 1,  # Number of input features
            "out_features": 1,
            "use_bias": True,
            "activation": "sig"
        }
        return config

    def __init__(self, config):
        super(ChannelWiseFC2d, self).__init__()
        self.in_features = config["in_features"]
        self.out_features = config["out_features"]
        self.in_channels = config["in_channels"]
        self.FCs = nn.ModuleList([nn.Linear(self.in_features, self.out_features, config["use_bias"])
                                  for _ in range(config["in_channels"])])
        self.activation = ACTIVATION_DICT[config["activation"]]()

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], -1)
        y = torch.zeros((x.shape[0], self.in_channels, self.out_features), device=x.device)
        for channel in range(self.in_channels):
            y[:, channel, :] = self.FCs[channel](x[:, channel, :])
        y = self.activation(y)
        return y


class DepthWiseCNNClassifier(nn.Module):

    @classmethod
    def default_config(cls):
        config = VGGLikeCNN.default_config()
        config.update({
            "n_blocs": 1,
            "conv_i_c": [16],  # input channels
            "conv_o_c": [16],  # output channels
            "conv_k_f": [3],  # kernel size on frequency axis
            "conv_k_t": [3],  # kernel size on time axis
            "conv_s_f": [1],  # stride on frequency axis
            "conv_s_t": [1],  # stride on time axis
            "conv_pad_type": "zero",
            "conv_p_f": [0],  # padding on frequency axis
            "conv_p_t": [0],  # padding on time axis

            "conv_groups": [1],  # Number of feature maps in each "group"

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
        return {**config, **{"FC_{}".format(key): value for key, value in ChannelWiseFC2d.default_config().items()}}

    def __init__(self, config):
        super(DepthWiseCNNClassifier, self).__init__()
        self.CNN = VGGLikeCNN(config)
        input_shape_FC = self.CNN(torch.zeros(*config["input_shape"])).shape
        config["FC_in_channels"], config["FC_in_features"] = input_shape_FC[1], input_shape_FC[2] * input_shape_FC[3]
        self.FC = ChannelWiseFC2d({key.replace('FC_', ''): value for key, value in config.items()})

    def forward(self, x):
        x = self.CNN(x)
        x = self.FC(x.view(x.shape[0], x.shape[1], -1))
        return x


class ChannelWiseRNNClassifier(nn.Module):

    @classmethod
    def default_config(cls):
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
        return config

    def __init__(self, config):
        super(ChannelWiseRNNClassifier, self).__init__()
        self.hidden_size = config["hidden_size"]

        cell_type_dict = {"GRU": nn.GRU, "LSTM": nn.LSTM, "RNN": nn.RNN}
        self.RNNs = nn.ModuleList([cell_type_dict[config["RNN_cell_type"]](input_size=config["input_size"],
                                                                           hidden_size=config["hidden_size"],
                                                                           num_layers=config["num_layers"],
                                                                           bias=config["use_bias"],
                                                                           batch_first=config["batch_first"],
                                                                           dropout=config["dropout"],
                                                                           bidirectional=config["bidirectional"])
                                   for _ in range(config["num_channels"])])

        self.ChannelWiseFC = ChannelWiseFC2d({"in_channels": config["num_channels"],
                                              "in_features": config["hidden_size"],
                                              "out_features": 1,
                                              "use_bias": True,
                                              "activation": "sig"})

    def forward(self, x):
        x = x.permute(0, 1, 3, 2)  # permute time and frequency axis, to have axis in position required by pytorch RNNs
        y = torch.zeros((x.shape[0], x.shape[1], self.hidden_size), device=x.device)
        for channel, rnn in enumerate(self.RNNs):
            _, y[:, channel, :] = rnn(x[:, channel, :, :])
        # y = (y + 1.0) / 2.0  # rescale output of tanh to [0, 1] to have probability output
        y = self.ChannelWiseFC(y)
        return y
