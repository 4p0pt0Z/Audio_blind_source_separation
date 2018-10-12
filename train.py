import torch
import numpy as np
import sklearn.metrics as skmetrics

import segmentation_model as md
import data_set as dts

import os


class TrainingManager:
    """

    """

    @classmethod
    def default_config(cls):
        config = {
            "model_type": "",  # Identifier to pass to the md.find_model_class function to get the class of the model.

            "data_set_type": "",  # Identifier to pass to the dts.find_dataset_class to get the class of the data sets.
            "batch_size": 32,
            "n_loaders": 0,  # 0 means loading happens in the same thread as the main thread

            "use_gpu": True,
            "gpu_no": 0,

            "metric": "",  # Accuracy, F-score, MCC, etc... See available in 'compute_metric'
            "average": "weighted",  # Average argument of the sklearn metrics: how to aggregate results across classes
            "threshold": [0.5],  # If required by the metric. Either 1 threshold common to all classes or a list

            "loss_f": "BCE",  # Loss function to use: BCE, multilabelSoftMarginLoss, etc ... (see 'init')

            # Optimizer parameters
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "weight_decay": 0.00001,

            # Learning rate scheduler parameters
            "scheduler_type": "",
            "scheduler_step_size": 0,  # Used with StepLR
            "scheduler_gamma": 0.0,  # Used with stepLR, multiStepLR and reduceLROnPlateau
            "scheduler_milestones": [0.0],  # Used with MultiStepLR
            "scheduler_patience": 0,  # Used with ReduceLROnPlateau

            "epoch_idx": 0,  # Stores the current epoch number
            "n_epochs": 0,  # Number of epoch to train
            "dev_every": 1,  # Evaluate model with an epoch on the dev after this much training epoch

            "save_path": "",  # path to save the model and manager settings
            "checkpoint_path": ""
        }
        return config

    def __init__(self, config):
        self.config = dict(config)

        self.device = torch.device("cpu") if not self.config["use_gpu"] \
            else torch.device("cuda:" + str(self.config["gpu_no"]))

        self.train_set, self.dev_set, self.test_set = \
            dts.find_data_set_class(self.config["data_set_type"]).split(self.config)

        self.shift_scale_data_sets()

        self.model = md.SegmentationModel(config, self.train_set.features_shape(), self.train_set.n_classes())

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

        # list storing loss function and metric values for each epoch
        self.train_losses, self.dev_losses, self.test_losses = [], [], []
        self.train_metrics, self.dev_metrics, self.test_metrics = [], [], []

    def save_state(self):
        state = {"model_state_dict": self.model.state_dict(),
                 "optimizer_state_dict": self.optimizer.state_dict(),
                 "config": self.config,
                 "train_losses": self.train_losses, "train_metrics": self.train_metrics,
                 "dev_losses": self.dev_losses, "dev_metrics": self.dev_metrics,
                 "test_losses": self.test_losses, "test_metrics": self.test_losses}
        torch.save(state, self.config["save_path"])

    def save_metrics_and_losses(self):
        try:
            state = torch.load(self.config["save_path"])
        except FileNotFoundError:
            print("Could not find saved model, saving metrics and losses ...")
            state = {}
        state["train_losses"], state["dev_losses"], state["test_losses"] = \
            self.train_losses, self.dev_losses, self.test_losses
        state["train_metrics"], state["dev_metrics"], state["test_metrics"] = \
            self.train_metrics, self.dev_metrics, self.test_metrics
        torch.save(state, self.config["save_path"])

    @classmethod
    def from_checkpoint(cls, filename, config_update=None):
        if not os.path.isfile(filename):
            raise ValueError("File " + filename + " is not a valid file.")
        print("Loading from checkpoint '{}'".format(filename))

        state = torch.load(filename)
        if config_update is not None:  # Update dict if we have updated parameters
            state["config"].update(config_update)
        manager = cls(state["config"])
        manager.train_losses, manager.dev_losses, manager.test_losses = \
            state["train_losses"], state["dev_losses"], state["test_losses"]
        manager.train_metrics, manager.dev_metrics, manager.test_metrics = \
            state["train_metrics"], state["dev_metrics"], state["test_metrics"]
        manager.model.load_state_dict(state["model_state_dict"])
        manager.optimizer.load_state_dict(state["optimizer_state_dict"])
        manager.model.to(manager.device)
        for state in manager.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(manager.device)
        return manager

    def shift_scale_data_sets(self):
        """
            Shift and scale the features of all sets with parameters computed on the training set
        """
        shift, scaling = self.train_set.compute_shift_and_scaling()
        self.config["shift"], self.config["scaling"] = shift, scaling
        self.train_set.shift_and_scale(shift, scaling)
        self.dev_set.shift_and_scale(shift, scaling)
        self.test_set.shift_and_scale(shift, scaling)

    def compute_metric(self, labels, predictions, average=None):
        """

        Args:
            predictions ():
            labels ():
            average ():

        Returns:

        """
        if average is None:
            average = self.config["average"] if self.config["average"].lower() != "none" else None
        if self.config["metric"] == "roc_auc_score":
            return skmetrics.roc_auc_score(labels, predictions, average=average)
        else:
            # Apply threshold:
            if len(self.config["threshold"]) == 1:
                predictions[predictions >= self.config["threshold"][0]] = 1
                predictions[predictions < self.config["threshold"][0]] = 0
            elif len(self.config["threshold"]) != predictions.shape[1]:
                raise ValueError(
                    "Number of thresholds {}".format(len(self.config["threshold"])) + " and classes {}".format(
                        predictions.shape[0]) + "are not matching.")
            else:
                for idx in range(len(self.config["threshold"])):
                    predictions[:, idx][predictions[:, idx] >= self.config["threshold"][idx]] = 1
                    predictions[:, idx][predictions[:, idx] < self.config["threshold"][idx]] = 0
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
        print("Epoch {} on {} set - ".format(epoch_idx, set_type) + self.config["loss_f"]
              + " loss: {:.4f} - ".format(loss_value), end='')
        if self.config["average"] != "None":
            print(self.config["average"] + " ", end='')
        print(self.config["metric"] + ": ", end='')
        if isinstance(metric_value, np.ndarray):
            print(["{:.4f}".format(value) for value in metric_value])
        else:
            print(": {:.4f}".format(metric_value))

    def train(self):
        """

        Returns:

        """
        # Move stuff to where it should be and make sure that the returned batches are on 'self.device'
        self.model.to(self.device)
        self.train_set.to(self.device)

        train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.config["batch_size"], shuffle=True,
                                                   num_workers=self.config["n_loaders"])

        max_metric = -np.inf

        for idx in range(1, self.config["n_epochs"] + 1):
            self.config["epoch_idx"] += 1
            all_predictions = []
            all_labels = []
            losses = []
            self.model.train()
            for batch_idx, (features, labels) in enumerate(train_loader):
                self.optimizer.zero_grad()
                predictions, _ = self.model(features)
                loss = self.loss_f(predictions, labels)
                loss.backward()
                self.optimizer.step()

                all_predictions.append(predictions.detach().cpu().numpy())
                all_labels.append(labels.detach().cpu().numpy())
                losses.append(loss.item())

            if self.config["scheduler_type"] == "ReduceLROnPlateau":
                self.scheduler.step(np.mean(losses))
            elif self.scheduler is not None:
                self.scheduler.step()

            all_predictions = np.concatenate(all_predictions, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            self.train_losses.append(np.mean(losses))
            self.train_metrics.append(self.compute_metric(all_labels, all_predictions))
            self.print_epoch(loss_value=self.train_losses[-1], metric_value=self.train_metrics[-1],
                             set_type="train", epoch_idx=self.config["epoch_idx"])

            # Monitor performances on development set
            if idx % self.config["dev_every"] == 0:
                dev_loss, dev_metric, weighted_dev_metric = self.evaluate(self.dev_set, special_average='weighted')
                self.dev_losses.append(dev_loss)
                self.dev_metrics.append(dev_metric)
                self.print_epoch(loss_value=dev_loss, metric_value=dev_metric, set_type="dev")

                if weighted_dev_metric > max_metric:  # in case of metrics per class, take mean
                    print("Saving best model...")
                    self.save_state()
                    max_metric = dev_metric

        print("Loading best model for evaluation on test set... ")
        state = torch.load(self.config["save_path"])
        self.model.load_state_dict(state["model_state_dict"])
        self.model.to(self.device)
        test_loss, test_metric = self.evaluate(self.test_set)
        self.test_losses.append(test_loss)
        self.test_metrics.append(test_metric)
        self.print_epoch(loss_value=test_loss, metric_value=test_metric, set_type="test")

        self.save_metrics_and_losses()

    def evaluate(self, data_set, special_average=None):
        self.model.to(self.device)
        data_loader = torch.utils.data.DataLoader(data_set, batch_size=self.config["batch_size"], shuffle=True,
                                                  num_workers=self.config["n_loaders"])
        data_loader.dataset.to(self.device)
        all_predictions = []
        all_labels = []
        losses = []
        self.model.eval()
        with torch.no_grad():
            for (features, labels) in data_loader:
                predictions, _ = self.model(features)
                loss = self.loss_f(predictions, labels)
                losses.append(loss.item())
                all_predictions.append(predictions.detach().cpu().numpy())
                all_labels.append(labels.detach().cpu().numpy())

            all_predictions = np.concatenate(all_predictions, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
        if special_average is None:
            return np.mean(losses), self.compute_metric(all_labels, all_predictions)
        else:
            return np.mean(losses),\
                   self.compute_metric(all_labels, all_predictions),\
                   self.compute_metric(all_labels, all_predictions, special_average)
