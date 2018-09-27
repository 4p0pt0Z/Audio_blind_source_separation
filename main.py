import argparse

import data_set as ds
import model as mod
import train as tr
import separator as sep
from helpers import str2bool

MODES = ["train", "evaluate", "separate"]


def main(exec_mode, config):
    if exec_mode == "train":
        filename = config["checkpoint_path"]
        if filename:
            config["checkpoint_path"] = ""
            tr_manager = tr.TrainingManager.from_checkpoint(filename, config)
        else:
            tr_manager = tr.TrainingManager(config)
        tr_manager.train()
    elif exec_mode == "evaluate":
        tr_manager = tr.TrainingManager.from_checkpoint(config["checkpoint_path"], config)
        test_loss, test_metric = tr_manager.evaluate(tr_manager.test_set)
        tr_manager.print_epoch(loss_value=test_loss, metric_value=test_metric, set_type="test")
    elif exec_mode == "separate":
        seperator = sep.AudioSeparator.from_checkpoint(config)
        seperator.separate()


def parse_arguments():
    parser = argparse.ArgumentParser(allow_abbrev=False,
                                     description="Training framework in pytorch. \n "
                                                 "Available audio processing parameters are the members of the "
                                                 "'default_config' dictionary of class of the command line argument "
                                                 "'data_set_type' (in data_set.py)\n"
                                                 "Available models parameters are the members of the 'default_config' "
                                                 "of the class of the command line argument 'model_type' (in model.py)"
                                                 "\n"
                                                 "Available training parameters (optimizer, learning rate scheduler, "
                                                 "loss function, metric...) are the members of the 'default_config' "
                                                 "dictionary of the 'TrainingManager' class in 'train.py'\n"
                                                 "Passed arguments update the default values in the default_config "
                                                 "dictionarries.\n"
                                                 "See README.md for more information")
    parser.add_argument("--mode", type=str, required=True,
                        help="Which mode of the script to execute: train for training a model, evaluate for evaluating "
                             "a model, and separate to generate source separated audio files")
    mode_arg = vars(parser.parse_known_args()[0])
    exec_mode = mode_arg.pop("mode")

    if exec_mode == "train" or exec_mode == "evaluate":
        parser.add_argument("-m", "--model_type", type=str, required=True,
                            help="Identifier for the class of the model. See 'find_model_class' in model.py")
        parser.add_argument("-d", "--data_set_type", type=str, required=True,
                            help="Identifier of the class of the data set. See 'find_data_set_class' in data_set.py")
        args = vars(parser.parse_known_args()[0])
        data_set_default_config = ds.find_data_set_class(args["data_set_type"]).default_config()
        model_default_config = mod.find_model_class(args["model_type"]).default_config()
        training_default_config = tr.TrainingManager.default_config()
        # merge the default dictionaries
        default_config = {**data_set_default_config, **model_default_config, **training_default_config}

    elif exec_mode == "separate":
        default_config = sep.AudioSeparator.default_config()
    else:
        raise NotImplementedError("Mode " + exec_mode + " is not Implemented")

    full_parser = argparse.ArgumentParser(allow_abbrev=False)
    # Add all default values as potential arguments
    for key, value in default_config.items():
        key = "--{}".format(key)
        if isinstance(value, list) or isinstance(value, tuple):
            full_parser.add_argument(key, default=value, nargs='*', type=type(value[0]))
        elif isinstance(value, bool):
            full_parser.add_argument(key, default=value, nargs='?', type=str2bool)
        else:
            full_parser.add_argument(key, default=value, type=type(value))

    parsed_args = vars(full_parser.parse_known_args()[0])
    parsed_args.update(mode_arg)
    if parsed_args["checkpoint_path"]:
        # Return only the values explicitly passed by user. Other values should be loaded from checkpoint
        new_args = {key: value for key, value in parsed_args.items() if value != default_config[key]}
        return exec_mode, new_args
    else:
        # update the values in config with the passed arguments or the default arguments
        default_config.update(parsed_args)
        return exec_mode, default_config


if __name__ == '__main__':
    exec_mode, conf_dict = parse_arguments()
    main(exec_mode, conf_dict)
