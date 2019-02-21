import argparse

import torch

import data_set as ds
import separation_model as mod
import train as tr
import separator as sep
from helpers import str2bool

MODES = ["train", "evaluate", "separate"]


def main(exec_mode, config):
    r"""Entry point of the separation framework: launch model training, model evaluation or audio separation evaluation

        Can chose between 3 tasks:
            - train: train a model from scratch, or continue training of a model.
            - evaluate: evaluate the classification performances of the model on the validation set
            - separate: Use model in checkpoint to separate all mixture in validation set, then compute separation
            metrics

    Args:
        exec_mode (str): Task to accomplish.
        config (dict): Configuration dictionary required to accomplish this task.
    """

    # Train a model
    if exec_mode == "train":
        # If user specified a checkpoint_path - will continue the training of a saved model.
        filename = config["checkpoint_path"]
        if filename:
            config["checkpoint_path"] = ""
            tr_manager = tr.TrainingManager.from_checkpoint(filename, config)
        # Else: train model from scratch.
        else:
            tr_manager = tr.TrainingManager(config)
        tr_manager.train()
    # Evaluate a model
    elif exec_mode == "evaluate":
        # Load the model and training/evaluation manager
        tr_manager = tr.TrainingManager.from_checkpoint(config["checkpoint_path"], config)
        # Run epoch on validation set and print results
        test_loss, test_metric = tr_manager.evaluate(tr_manager.val_set)
        tr_manager.print_epoch(loss_value=test_loss, metric_value=test_metric, set_type="test")
    # Use saved model to separate audio mixtures in the validation set and evaluate separation performances.
    elif exec_mode == "separate":
        separator = sep.AudioSeparator.from_checkpoint(config)
        # Separate the mixtures using the model
        separator.separate()
        # Compute separation metrics (slow)
        sdrs, sirs, sars = separator.evaluate_separation()
        # Print metrics
        format_string = 'mean {:^9.4f}, std {:^9.4f} \nSIR: mean {:^9.4f}, std {:^9.4f} \n' \
                        'SAR: mean {:^9.4f}, std {:^9.4f}'
        print('Average results\n: ' + format_string.format(
            sdrs.mean(), sdrs.std(),
            sirs.mean(), sirs.std(),
            sars.mean(), sars.std()))


def parse_arguments():
    r"""Parse user arguments to determine the execution mode to use. Then parse the arguments required for this mode.

        The framework can perform 3 tasks (see main()). First the methd parses the command line arguments to detect
        which task should be done.
        Then the method parses the arguments which are specific to this task.
        This is done using the following trick:
            Most objects in the code (models, training_manager, AudioSeparator, AudioDataSets, etc...) have a method
            'default_config()'. This method returns a dictionary that contains the tunable parameters for this object
            along with their default value.
            The argument parser first aggregates all the parameters dictionaries for all the objects.
            Then it is going to check if the user entered an argument corresponding to any parameters in this
            aggregated dictionary. If an argument is passed, we update the default value of this parameter with the
            received value.
        This aggregated and updated dictionary will be passed to all objects in the code so that they can receive the
        user arguments.

    Returns:
        exec_mode, config: Execution mode identifier expected by 'main()' and the configuration dictionary required
        for this mode.
    """

    # First parse the execution mode
    parser = argparse.ArgumentParser(allow_abbrev=False,
                                     description="Training framework in pytorch. \n "
                                                 "Available audio processing parameters are the members of the "
                                                 "'default_config' dictionary of class of the command line argument "
                                                 "'data_set_type' (in data_set.py)\n"
                                                 "Available models parameters are the members of the 'default_config' "
                                                 "of the class of the command line argument '**_model_type'"
                                                 " (in **_model.py)\n"
                                                 "Available training parameters (optimizer, learning rate scheduler, "
                                                 "loss function, metric...) are the members of the 'default_config' "
                                                 "dictionary of the 'TrainingManager' class in 'train.py'\n"
                                                 "Passed arguments update the default values in the default_config "
                                                 "dictionarries.\n"
                                                 "See README.md for more information")
    parser.add_argument("--mode", type=str, required=True,
                        help="Which mode of the script to execute: train for training a model, evaluate for evaluating "
                             "a model, and separate to generate source separated audio files")
    initial_args = vars(parser.parse_known_args()[0])
    exec_mode = initial_args.pop("mode")

    # Now gather the possible arguments for the input execution mode
    if exec_mode == "train":
        # In order to get the arguments for the models and dataset objects, we need to know the type of these objects
        parser.add_argument("-m", "--mask_model_type", type=str, required=True,
                            help="Identifier for the class of the model bulding the segmentation masks."
                                 "See 'find_model_class' in separation_model.py")
        parser.add_argument("-c", "--classifier_model_type", type=str, required=True,
                            help="Identifier for the class of the classifier model. "
                                 "See 'find_model_class' in separation_model.py")
        parser.add_argument("-d", "--data_set_type", type=str, required=True,
                            help="Identifier of the class of the data set. See 'find_data_set_class' in data_set.py")
        args = vars(parser.parse_known_args()[0])
        initial_args.update(args)

        # Get all the arguments for the data_set, the model and the training_manager.
        data_set_default_config = ds.find_data_set_class(args["data_set_type"]).default_config()
        model_default_config = mod.SeparationModel.default_config(args["mask_model_type"],
                                                                  args["classifier_model_type"])
        training_default_config = tr.TrainingManager.default_config()
        # merge all dictionaries together
        default_config = {**data_set_default_config, **model_default_config, **training_default_config}

    # If execution mode is evaluate, all we need to know is the checkpoint to the model to evaluate.
    # The configuration for the model and data set are saved in the checkpoint.
    elif exec_mode == "evaluate":
        # get the checkpoint path
        parser.add_argument("--checkpoint_path", type=str, required=True,
                            help="Path to the saved model checkpoint.")
        args = vars(parser.parse_known_args()[0])
        # Read the configuration from checkpoint
        state = torch.load(args["checkpoint_path"], 'cpu')
        default_config = state["config"]

    # If execution mode is separate, we just need the AudioSeparator configuration.
    # Model and dataset configuration will be read from checkpoint.
    elif exec_mode == "separate":
        default_config = sep.AudioSeparator.default_config()

    else:
        raise NotImplementedError("Mode " + exec_mode + " is not Implemented")

    # Now parse the arguments for parameters in the default_config()
    full_parser = argparse.ArgumentParser(allow_abbrev=False)
    for key, value in default_config.items():
        key = "--{}".format(key)
        if isinstance(value, list) or isinstance(value, tuple):
            full_parser.add_argument(key, default=value, nargs='*', type=type(value[0]))
        elif isinstance(value, bool):
            full_parser.add_argument(key, default=value, nargs='?', type=str2bool)
        else:
            full_parser.add_argument(key, default=value, type=type(value))

    parsed_args = vars(full_parser.parse_known_args()[0])
    parsed_args.update(initial_args)
    default_config.update(initial_args)
    # If a model will be loaded from checkpoint, we do not want to over-write the values in its config dict with the
    # default values from default_config(). Therefore we only update the values if they were explicitly passed by user.
    if parsed_args["checkpoint_path"]:
        # Return only the values explicitly passed by user. Other values should be loaded from checkpoint
        new_args = {key: value for key, value in parsed_args.items() if value != default_config[key]}
        return exec_mode, new_args
    else:
        # update the values in config with the passed arguments or the default arguments
        default_config.update(parsed_args)
        return exec_mode, default_config


if __name__ == '__main__':
    r"""Parse user arguments, then launch main"""
    mode, conf_dict = parse_arguments()
    main(mode, conf_dict)
