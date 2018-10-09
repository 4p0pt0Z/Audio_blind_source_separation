import torch
import torch.nn as nn

from mask_model import find_mask_model_class
from classifier_model import find_classifier_model_class


class SegmentationModel(nn.Module):

    @classmethod
    def default_config(cls, mask_model_type, classifier_model_type):
        """
            Get all the parameters for the maks model and classifier model.
            WARNING: Note that some of these parameters are overwritten in the __init__ method for consistency between
            the mask model and classifier model !
        Args:
            mask_model_type (str): type of the mask model
            classifier_model_type (str): type of the classifier

        Returns:

        """
        g1_config = {"g1_{}".format(key): value
                     for key, value in find_mask_model_class(mask_model_type).default_config().items()}
        g2_config = {"g2_{}".format(key): value
                     for key, value in find_classifier_model_class(classifier_model_type).default_config().items()}
        return {**g1_config, **g2_config}

    def __init__(self, config, input_shape, n_classes):
        super(SegmentationModel, self).__init__()

        config["g1_o_c"][-1] = n_classes

        # Instantiate with sizes, etc...
        self.mask_model = find_mask_model_class(config["mask_model_type"])(
            {key.replace('g1_', ''): value for key, value in config.items()})

        x = torch.zeros((1,) + input_shape)  # add batch dimension
        x = self.mask_model(x)
        config["g2_input_shape"] = tuple(x.shape)
        if config["classifier_model_type"] == "ChannelWiseFC2d":
            config["g2_in_channels"], config["g2_in_features"] = \
                config["g2_input_shape"][1], config["g2_input_shape"][2] * config["g2_input_shape"][3]
        elif config["classifier_model_type"] == "ChannelWiseRNNClassifier":
            config["g2_num_channels"], config["g2_input_size"] = x.shape[1], x.shape[2]

        self.classifier_model = find_classifier_model_class(config["classifier_model_type"])(
            {key.replace('g2_', ''): value for key, value in config.items()})

    def forward(self, x):
        x = self.mask_model(x)
        masks = x.detach()
        labels = self.classifier_model(x)
        return labels.squeeze(), masks
