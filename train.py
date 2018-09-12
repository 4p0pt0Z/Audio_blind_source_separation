import torch
import numpy as np
import sklearn.metrics as skmetrics

import model as md
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
            "batch_size": 0,
            "n_loaders": 0,

            "use_gpu": True,
            "gpu_no": 0,

            "metric": "",  # Accuracy, F-score, MCC, etc... See available in 'compute_metric'
            "threshold": [0],  # If required by the metric. Either 1 threshold common to all classes or a list

            "loss_f": "",  # Loss function to use: BCE, multilabelSoftMarginLoss, etc ... (see 'init')

            # Optimizer parameters
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "weight_decay": 0.00001,

            # Learning rate scheduler parameters
            "scheduler_type": "",
            "scheduler_step_size": 0,  # Used with StepLR
            "scheduler_gamma": 0.0,  # Used with StepLR, MultiStepLR and ReduceLROnPlateau
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

        self.model = md.find_model_class(config["model_type"])(config)

        self.train_set, self.dev_set, self.test_set = \
            dts.find_data_set_class(self.config["data_set_type"]).split(self.config)

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
                                                                  milestones=self.config["milestones"],
                                                                  gamma=self.config["scheduler_gamma"])
        elif self.config["scheduler_type"] == "ReduceLROnPlateau":
            # Reduce learning rate if the loss value does not decrease during 'patience' number of epoch
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer,
                                                                        patience=self.config["patience"],
                                                                        factor=self.config["scheduler_gamma"])
        elif not self.config["scheduler_type"]:
            # Do not use any scheduler
            self.scheduler = None
        else:
            raise NotImplementedError("Learning rate scheduler " + self.config["scheduler_type"] + " is not available.")

        # Loss function
        if self.config["loss_f"] == "BCE":
            self.loss_f = torch.nn.BCELoss()
        else:
            raise NotImplementedError("Loss function " + self.config["loss_f"] + " is not available.")

    def save_state(self):
        state = {"model_state_dict": self.model.state_dict(),
                 "optimizer_state_dict": self.optimizer.state_dict(),
                 "config": self.config}
        torch.save(state, self.config["save_path"])

    @classmethod
    def load_state(cls, filename, config_update=None):
        if not os.path.isfile(filename):
            raise ValueError("File " + filename + " is not a valid file.")
        print("Loading from checkpoint '{}'".format(filename))

        state = torch.load(filename)
        if config_update is not None:  # Update dict if we have updated parameters
            state["config"].update(config_update)
        manager = cls(state["config"])
        manager.model.load_state_dict(state["model_state_dict"])
        manager.optimizer.load_state_dict(state["optimizer_state_dict"])
        manager.model.to(manager.device)
        for state in manager.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(manager.device)
        return manager

    def compute_metric(self, labels, predictions):
        """

        Args:
            predictions ():
            labels ():

        Returns:

        """
        if self.config["metric"] == "roc_auc_score":
            return skmetrics.roc_auc_score(labels, predictions)
        else:
            # Apply threshold:
            predictions[predictions >= self.config["threshold"]] = 1
            predictions[predictions < self.config["threshold"]] = 0
            if self.config["metric"] == "accuracy":
                return skmetrics.accuracy_score(labels, predictions)
            elif self.config["metric"] == "f1-score":
                return skmetrics.f1_score(labels, predictions)
            elif self.config["metric"] == "matthews_corrcoef":
                return skmetrics.matthews_corrcoef(labels, predictions)
            elif self.config["metric"] == "precision":
                return skmetrics.precision_score(labels, predictions)
            elif self.config["metric"] == "recall":
                return skmetrics.recall_score(labels, predictions)

    def print_epoch(self, loss_value, metric_value, set_type, epoch_idx="\b"):
        print("Epoch {} on {} set - ".format(epoch_idx, set_type) + self.config["loss_f"]
              + " loss: {:.4f} - ".format(loss_value) + self.config["metric"] + ": {:.4f}".format(metric_value))

    def train(self):
        """

        Returns:

        """
        # Move stuff to where it should be and make sure that the returned batches are on 'self.device'
        self.model.to(self.device)
        self.train_set.to(self.device)
        self.dev_set.to(self.device)
        self.test_set.to(self.device)

        train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.config["batch_size"], shuffle=True,
                                                   num_workers=self.config["n_loaders"])
        dev_loader = torch.utils.data.DataLoader(self.dev_set, batch_size=self.config["batch_size"], shuffle=True,
                                                 num_workers=self.config["n_loaders"])
        test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=self.config["batch_size"], shuffle=True,
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
            self.print_epoch(loss_value=np.mean(losses), metric_value=self.compute_metric(all_labels, all_predictions),
                             set_type="train", epoch_idx=self.config["epoch_idx"])

            # Monitor performances on development set
            if idx % self.config["dev_every"] == 0:
                dev_loss, dev_metric = self.evaluate(dev_loader)
                self.print_epoch(loss_value=dev_loss, metric_value=dev_metric, set_type="dev", epoch_idx="")

                if dev_metric > max_metric:
                    print("Saving best model...")
                    self.save_state()
                    max_metric = dev_metric

        print("Loading best model for evaluation on test set... ")
        test_loss, test_metric = self.evaluate(test_loader)
        self.print_epoch(loss_value=test_loss, metric_value=test_metric, set_type="test", epoch_idx="")

    def evaluate(self, data_loader):
        self.model.to(self.device)
        data_loader.dataset.to(self.device)
        all_predictions = []
        all_labels = []
        losses = []
        self.model.eval()
        for (features, labels) in data_loader:
            predictions, _ = self.model(features)
            loss = self.loss_f(predictions, labels)
            losses.append(loss.item())
            all_predictions.append(predictions.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

        all_predictions = np.concatenate(all_predictions, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        return np.mean(losses), self.compute_metric(all_labels, all_predictions)
