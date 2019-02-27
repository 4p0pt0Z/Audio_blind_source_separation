import torch
import numpy as np
import sklearn.metrics as skmetrics

import separation_model as md
import data_set as dts

import os


class TrainingManager:
    r"""Implements a training and evaluation framework for the audio separation models.

        This class regroups the different elements required to train a audio separation model:
            - the data set (training, testing and validation)
            - the model
            - the optimizer
        It provides methods to monitor the model training performances, easily save and re-load a model for evaluation.

    """

    @classmethod
    def default_config(cls):
        r"""Provides a dictionary with the tunable training parameters."""

        config = {
            "data_set_type": "",  # Identifier to pass to the dts.find_dataset_class to get the class of the data sets.
            "batch_size": 32,  # Size of mini-batches during training

            # Number of worker to use for the data loaders.
            # 0 means the loading happens in the same thread as the main program runs. This is fine if the training
            # data is already loaded in RAM.
            "n_loaders": 0,

            "use_gpu": True,
            "gpu_no": 0,  # If multiple gpus are available, chose one

            "metric": "",  # Accuracy, F-score, MCC, etc... See available in 'compute_metric()'
            "average": "weighted",  # Average argument of the sklearn metrics: how to aggregate results across classes
            # Some metrics only take binary input (eg accuracy: prediction need to be 1 or 0)
            # Can be either a list with 1 value, in which case the same threshold is used for all classes
            # or a list of values, one for each class.
            "threshold": [0.5],

            # Loss function to use: BCE, multilabelSoftMarginLoss, etc ... (see '__init__()')
            "loss_f": "BCE",

            # Weight to give to the L1 loss applied to the masks.
            "l1_loss_lambda": 0.0,

            # Optimizer parameters
            "optimizer": "Adam",
            "learning_rate": 0.0001,
            "weight_decay": 0.00001,

            # Learning rate scheduler parameters
            "scheduler_type": "",  # see '__init__' for supported scheduler types
            "scheduler_step_size": 0,  # Used with StepLR: number of epoch for each step
            "scheduler_gamma": 0.0,  # Used with stepLR, multiStepLR and reduceLROnPlateau: learning rate multiplier
            "scheduler_milestones": [0.0],  # Used with MultiStepLR: epoch number at which to change the learning rate
            "scheduler_patience": 0,  # Used with ReduceLROnPlateau: number of epochs to wait before reducing lr

            "epoch_idx": 0,  # Stores the current epoch number
            "n_epochs": 0,  # Number of epoch to train
            "test_every": 1,  # Evaluate model with an epoch on test set every this amount of training epoch

            "save_path": "",  # path to save the model and manager settings into a checkpoint file.
            "checkpoint_path": ""  # When evaluating a saved model: path to the checkpoint.
        }
        return config

    def __init__(self, config):
        r"""Constructor. Instantiates the data sets, the model, the optimizer, scheduler, loss function.

        Args:
            config (dict): Configuration dictionary with tunable training parameters
        """

        self.config = dict(config)

        self.device = torch.device("cpu") if not self.config["use_gpu"] \
            else torch.device("cuda:" + str(self.config["gpu_no"]))

        # Instantiate the data sets.
        self.train_set, self.test_set, self.val_set = \
            dts.find_data_set_class(self.config["data_set_type"]).split(self.config)

        # Scale the features
        self.shift_scale_data_sets()

        # Instantiate the model
        self.model = md.SeparationModel(config, self.train_set.features_shape(), self.train_set.n_classes())

        # Optimizer
        if self.config["optimizer"] == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=self.config["learning_rate"],
                                              weight_decay=self.config["weight_decay"])
        else:
            raise NotImplementedError('The optimizer ' + self.config["optimizer"] + ' is not available.')

        # Learning rate scheduler
        if self.config["scheduler_type"] == "stepLR":
            # Reduce lr after every step_size number of epoch
            self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer,
                                                             step_size=self.config["scheduler_step_size"],
                                                             gamma=self.config["scheduler_gamma"])
        elif self.config["scheduler_type"] == "multiStepLR":
            # Reduce the learning rate when the epochs in milestones are reached
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer,
                                                                  milestones=self.config["scheduler_milestones"],
                                                                  gamma=self.config["scheduler_gamma"])
        elif self.config["scheduler_type"] == "reduceLROnPlateau":
            # Reduce learning rate if the loss value does not decrease during 'patience' number of epoch
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer,
                                                                        patience=self.config["scheduler_patience"],
                                                                        factor=self.config["scheduler_gamma"])
        elif not self.config["scheduler_type"]:
            # Do not use any scheduler
            self.scheduler = None
        else:
            raise NotImplementedError("Learning rate scheduler " + self.config["scheduler_type"] + " is not available.")

        # Loss function
        if self.config["loss_f"] == "BCE":
            self.loss_f = torch.nn.BCELoss()
        elif self.config["loss_f"] == "MultiLabelSoftMarginLoss":
            self.loss_f = torch.nn.MultiLabelSoftMarginLoss()
        else:
            raise NotImplementedError("Loss function " + self.config["loss_f"] + " is not available.")

        # l1 loss function, to penalize masks activations when they should be 0.
        self.l1_loss_f = torch.nn.L1Loss()
        self.l1_loss_lambda = self.config["l1_loss_lambda"]

        # list storing loss function and metric values for each epoch
        self.train_losses, self.test_losses, self.val_losses = [], [], []
        self.train_metrics, self.test_metrics, self.val_metrics = [], [], []

        # List to save the trainable pcen parameters at each epoch (if any)
        self.pcen_parameters = []

    def save_state(self):
        r"""Saves the model, training metrics and losses, trainable PCEN parameters and configuration to checkpoint.

            Loading this checkpoint should be enough to resume the training or evaluate the model, or investigate
            training behaviour of PCEN parameters.
        """

        state = {"model_state_dict": self.model.state_dict(),
                 "optimizer_state_dict": self.optimizer.state_dict(),
                 "config": self.config,
                 "train_losses": self.train_losses, "train_metrics": self.train_metrics,
                 "test_losses": self.test_losses, "test_metrics": self.test_metrics,
                 "val_losses": self.val_losses, "val_metrics": self.val_losses,
                 "pcen_parameters": self.pcen_parameters}
        torch.save(state, self.config["save_path"])

    def save_metrics_and_losses(self):
        r"""Save the training metrics and PCEN parameters to checkpoint, but do not over-write the saved model."""
        try:
            state = torch.load(self.config["save_path"], 'cpu')
        except FileNotFoundError:
            print("Could not find saved model, saving metrics and losses ...")
            state = {}
        state["train_losses"], state["test_losses"], state["val_losses"] = \
            self.train_losses, self.test_losses, self.val_losses
        state["train_metrics"], state["test_metrics"], state["val_metrics"] = \
            self.train_metrics, self.test_metrics, self.val_metrics
        state["pcen_parameters"] = self.pcen_parameters  # .detach().clone()
        torch.save(state, self.config["save_path"])

    @classmethod
    def from_checkpoint(cls, filename, config_update=None):
        r"""Build a training manager from checkpoint

            Build a training manager with its data sets and model from checkpoint. This allows to continue training
            of a model, or evaluate a saved model.

        Args:
            filename (str): path to the checkpoint file
            config_update (dict): Eventually, the training parameters can be updated when resuming training.

        Returns:
            TrainingManager with model loaded from checkpoint.
        """

        if not os.path.isfile(filename):
            raise ValueError("File " + filename + " is not a valid file.")
        print("Loading from checkpoint '{}'".format(filename))

        # Load the checkpoint dictionary
        state = torch.load(filename, 'cpu')
        if config_update is not None:  # Update dict if we have updated parameters
            state["config"].update(config_update)

        # Instantiate manager
        manager = cls(state["config"])

        # Load saved losses and metrics
        manager.train_losses, manager.test_losses, manager.val_losses = \
            state["train_losses"], state["test_losses"], state["val_losses"]
        manager.train_metrics, manager.test_metrics, manager.val_metrics = \
            state["train_metrics"], state["test_metrics"], state["val_metrics"]

        # Load model parameters
        manager.model.load_state_dict(state["model_state_dict"])

        # Load optimizer parameters.
        manager.optimizer.load_state_dict(state["optimizer_state_dict"])

        # Move model and optimizer to device
        manager.model.to(manager.device)
        for state in manager.optimizer.state.values():  # due to pytorch bug, need to loop manually for optimizer params
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(manager.device)

        return manager

    def shift_scale_data_sets(self):
        r"""Shift and scale the features of all sets with statistics computed on the training set."""

        shift, scaling = self.train_set.compute_shift_and_scaling()
        # Shift and scaling parameters are saved inside the config. We might need them.
        self.config["shift"], self.config["scaling"] = shift, scaling
        self.train_set.shift_and_scale(shift, scaling)
        self.test_set.shift_and_scale(shift, scaling)
        self.val_set.shift_and_scale(shift, scaling)

    def compute_metric(self, labels, predictions, average=None):
        r"""Compute a classification metric score.

        Args:
            labels (np.ndarray): Groundtruth labels
            predictions (np.ndarray): Models predictions
            average (str): sklearn 'average' argument: how to aggregate the metric score accros classes
                           If the parameter is not passed: will use the value in self.config

        Returns:
            metric value
        """

        if average is None:
            average = self.config["average"] if self.config["average"].lower() != "none" else None

        # If metric is "area under curve" based, no need for threshold. Compute metric
        if self.config["metric"] == "roc_auc_score":
            return skmetrics.roc_auc_score(labels, predictions, average=average)
        else:
            # Apply threshold:
            # Only 1 threshold available: use the same for all classes
            if len(self.config["threshold"]) == 1:
                predictions[predictions >= self.config["threshold"][0]] = 1
                predictions[predictions < self.config["threshold"][0]] = 0
            # Several threshold are passed, but not the same number as the number of classes: raise Error
            elif len(self.config["threshold"]) != predictions.shape[1]:
                raise ValueError(
                    "Number of thresholds {}".format(len(self.config["threshold"])) + " and classes {}".format(
                        predictions.shape[0]) + "are not matching.")
            # Several threshold are passed: use each threshold for each class
            else:
                for idx in range(len(self.config["threshold"])):
                    predictions[:, idx][predictions[:, idx] >= self.config["threshold"][idx]] = 1
                    predictions[:, idx][predictions[:, idx] < self.config["threshold"][idx]] = 0

            # Compute metric
            if self.config["metric"] == "accuracy":
                return skmetrics.accuracy_score(labels, predictions)
            elif self.config["metric"] == "f1-score":
                return skmetrics.f1_score(labels, predictions, average=average)
            elif self.config["metric"] == "matthews_corrcoef":
                return skmetrics.matthews_corrcoef(labels, predictions)
            elif self.config["metric"] == "precision":
                return skmetrics.precision_score(labels, predictions, average=average)
            elif self.config["metric"] == "average_precision_score":
                return skmetrics.average_precision_score(labels, predictions, average=average)
            elif self.config["metric"] == "recall":
                return skmetrics.recall_score(labels, predictions, average=average)

    def print_epoch(self, loss_value, metric_value, set_type, epoch_idx="\b"):
        r"""Pretty print of the loss and metric value per epoch

        Args:
            loss_value (float): Loss function value
            metric_value (float): Metric value
            set_type (str): 'Training', 'Testing' or 'Validation'
            epoch_idx (int): Number of epoch since the begining of training
        """

        print("Epoch {} on {} set - ".format(epoch_idx, set_type) + self.config["loss_f"]
              + " loss: {:.4f} - ".format(loss_value), end='', flush=True)
        if self.config["average"] != "None":
            print(self.config["average"] + " ", end='', flush=True)
        print(self.config["metric"] + ": ", end='', flush=True)
        if isinstance(metric_value, np.ndarray):
            print(["{:.4f}".format(value) for value in metric_value], flush=True)
        else:
            print(": {:.4f}".format(metric_value), flush=True)

    def train(self):
        r"""Training loop"""

        # Move stuff to where it should be and make sure that the returned batches are on 'self.device'
        self.model.to(self.device)
        self.train_set.to(self.device)

        # Pass the data set to the torch.data.DataLoader wrapper (for shuffling, and potentially parallel execution)
        train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.config["batch_size"], shuffle=True,
                                                   num_workers=self.config["n_loaders"])

        max_metric = -np.inf  # Record best metric on the testing set
        self.save_metrics_and_losses()

        for idx in range(1, self.config["n_epochs"] + 1):  # Loop over the training epochs
            self.config["epoch_idx"] += 1

            # Keep all predictions, label and loses of the processed example per epoch to compute average per epoch.
            all_predictions = []
            all_labels = []
            losses = []

            self.model.train()
            for batch_idx, (features, labels) in enumerate(train_loader):  # Loop over the batches in the epoch
                self.optimizer.zero_grad()
                predictions, masks = self.model(features)
                loss = self.loss_f(predictions, labels)  # Classification loss
                # L1 loss on the mask activation, to penalize the activations in a mask when the class was not present.
                l1_loss = self.l1_loss_f(masks[1 - labels.to(torch.uint8)],
                                         torch.zeros(masks[1 - labels.to(torch.uint8)].shape).to(masks))
                # Take linear combination of the 2 losses for updating the weights
                ((1 - self.l1_loss_lambda) * loss + self.l1_loss_lambda * l1_loss).backward()
                self.optimizer.step()

                all_predictions.append(predictions.detach().cpu().numpy())
                all_labels.append(labels.detach().cpu().numpy())
                losses.append(((1 - self.l1_loss_lambda) * loss + self.l1_loss_lambda * l1_loss).item())

            # End of epoch on training set.

            if self.config["scheduler_type"] == "ReduceLROnPlateau":
                self.scheduler.step(np.mean(losses))
            elif self.scheduler is not None:
                self.scheduler.step()

            # Compute epoch averages
            all_predictions = np.concatenate(all_predictions, axis=0)  # list of values to 1D array
            all_labels = np.concatenate(all_labels, axis=0)
            self.train_losses.append(np.mean(losses))
            self.train_metrics.append(self.compute_metric(all_labels, all_predictions))
            self.print_epoch(loss_value=self.train_losses[-1], metric_value=self.train_metrics[-1],
                             set_type="training", epoch_idx=self.config["epoch_idx"])

            # Save the trainable PCEN parameters (if any)
            if hasattr(self.model, 'pcen'):
                if self.model.pcen is not None:
                    self.pcen_parameters.append({key: torch.tensor(value, requires_grad=False)
                                                 for key, value in self.model.pcen.state_dict().items()})

            # Monitor performances on testing set every once in a while. If best score is achieved: save model
            if idx % self.config["test_every"] == 0:
                test_loss, test_metric, weighted_test_metric = self.evaluate(self.test_set, special_average='weighted')
                self.test_losses.append(test_loss)
                self.test_metrics.append(test_metric)
                self.print_epoch(loss_value=test_loss, metric_value=test_metric, set_type="testing")

                if weighted_test_metric > max_metric:  # in case of metrics per class, take mean
                    print("Saving best model...")
                    self.save_state()
                    max_metric = test_metric

        print("Loading best model for evaluation on validation set... ")
        state = torch.load(self.config["save_path"], 'cpu')
        self.model.load_state_dict(state["model_state_dict"])
        self.model.to(self.device)
        val_loss, val_metric = self.evaluate(self.val_set)
        self.val_losses.append(val_loss)
        self.val_metrics.append(val_metric)
        self.print_epoch(loss_value=val_loss, metric_value=val_metric, set_type="validation")

        self.save_metrics_and_losses()

        # In case the metric was not a good indicator and we still would like to save the last epoch model.
        print("Saving model at last epoch...")
        self.config['save_path'] = os.path.splitext(os.path.basename(self.config['save_path']))[0] + '_final' + '.ckpt'
        self.save_state()
        # Put back save path
        self.config['save_path'] = os.path.splitext(os.path.basename(self.config['save_path']))[0][:-6] + 'ckpt'

    def evaluate(self, data_set, special_average=None):
        r"""Run the model through an epoch of a dataset, to compute loss function and metric averages.

        Args:
            data_set (torch.data.Dataset): Dataset to evaluate the model
            special_average (str): sklearn 'average' argument: how to aggregate the metric score accros classes
                                   The metric score is computed using the 'average' parameter in self.config - and
                                   also using this special average.
                                   This is used in the training loop: In order to know if the model performs best
                                   over-all, a single number metric score needs to be calculated even if the
                                   monitored metric is computed per class.

        Returns:
            (Average loss, Average score) on the input data set.
            (Average loss, Average score, Weighted average score) on the input data set.
        """

        # Move model and data set to device
        self.model.to(self.device)
        data_loader = torch.utils.data.DataLoader(data_set, batch_size=self.config["batch_size"], shuffle=True,
                                                  num_workers=self.config["n_loaders"])
        data_loader.dataset.to(self.device)

        # List to aggregate the loss and metric values over the batches
        all_predictions = []
        all_labels = []
        losses = []
        self.model.eval()
        with torch.no_grad():
            for (features, labels) in data_loader:  # Loop over the batches in the epoch
                predictions, masks = self.model(features)
                loss = self.loss_f(predictions, labels)
                losses.append(loss.item())
                all_predictions.append(predictions.detach().cpu().numpy())
                all_labels.append(labels.detach().cpu().numpy())

            all_predictions = np.concatenate(all_predictions, axis=0)  # list of arrays to 1D array
            all_labels = np.concatenate(all_labels, axis=0)
        if special_average is None:
            return np.mean(losses), self.compute_metric(all_labels, all_predictions)
        else:
            return np.mean(losses), \
                   self.compute_metric(all_labels, all_predictions), \
                   self.compute_metric(all_labels, all_predictions, special_average)
