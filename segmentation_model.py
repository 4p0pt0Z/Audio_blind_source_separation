import torch
import torch.nn as nn

from mask_model import find_mask_model_class
from classifier_model import find_classifier_model_class
from pcen import PCEN


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
        mask_config = {"mask_{}".format(key): value
                       for key, value in find_mask_model_class(mask_model_type).default_config().items()}
        class_config = {"class_{}".format(key): value
                        for key, value in find_classifier_model_class(classifier_model_type).default_config().items()}
        pcen_config = {"train_pcen": False, "train_pcen_s": [0.0], "train_pcen_eps": 1e-6}
        return {**pcen_config, **mask_config, **class_config}

    def __init__(self, config, input_shape, n_classes):
        super(SegmentationModel, self).__init__()

        config["mask_conv_o_c"][-1] = n_classes

        if config["train_pcen"]:
            self.pcen = PCEN(input_shape[-2],
                             config["train_pcen_s"] if config["train_pcen_s"][0] != 0.0 else None,
                             config["train_pcen_eps"])

        # Instantiate with sizes, etc...
        self.mask_model = find_mask_model_class(config["mask_model_type"])(
            {key.replace('mask_', ''): value for key, value in config.items()})

        # Adapt parameters from output of mask model to input of classifier model
        x = torch.zeros((1,) + input_shape)  # add batch dimension
        x = self.mask_model(x)
        config["class_input_shape"] = tuple(x.shape)
        if config["classifier_model_type"] == "ChannelWiseFC2d":
            config["class_in_channels"], config["class_in_features"] = \
                config["class_input_shape"][1], config["class_input_shape"][2] * config["class_input_shape"][3]
        elif config["classifier_model_type"] == "ChannelWiseRNNClassifier":
            config["class_num_channels"], config["class_input_size"] = x.shape[1], x.shape[2]
        elif config["classifier_model_type"] == "DepthWiseCNNClassifier":
            config["class_conv_i_c"] = [n_classes] * config["class_n_blocs"]
            config["class_conv_o_c"] = [n_classes] * config["class_n_blocs"]
            config["class_conv_groups"] = [n_classes] * config["class_n_blocs"]

        self.classifier_model = find_classifier_model_class(config["classifier_model_type"])(
            {key.replace('class_', ''): value for key, value in config.items()})

    def forward(self, x):
        if hasattr(self, "pcen"):
            x = self.pcen(x)
        x = self.mask_model(x)
        masks = x
        labels = self.classifier_model(x)
        return labels.squeeze(), masks
