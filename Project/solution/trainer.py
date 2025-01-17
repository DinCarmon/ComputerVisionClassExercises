"""Train models on a given dataset."""
import os
import json
from dataclasses import dataclass

import torch
from numpy.f2py.symbolic import number_types
from torch import nn
from torch.utils.data import Dataset, DataLoader

from common import OUTPUT_DIR, CHECKPOINT_DIR


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@dataclass
class LoggingParameters:
    """Data class holding parameters for logging."""
    model_name: str
    dataset_name: str
    optimizer_name: str
    optimizer_params: dict

# pylint: disable=R0902, R0913, R0914
class Trainer:
    """Abstract model trainer on a binary classification task."""
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim,
                 criterion,
                 batch_size: int,
                 train_dataset: Dataset,
                 validation_dataset: Dataset,
                 test_dataset: Dataset):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.epoch = 0

    def train_one_epoch(self) -> tuple[float, float]:
        """Train the model for a single epoch on the training dataset.
        Returns:
            (avg_loss, accuracy): tuple containing the average loss and
            accuracy across all dataset samples.
        """
        # Set the model to training mode
        # This is essential because certain layers and behaviors in a torch NN behave differently
        # during training and evaluation
        self.model.train()

        total_loss = 0
        avg_loss = 0
        accuracy = 0
        nof_samples = 0     # number of samples
        correct_labeled_samples = 0

        train_dataloader = DataLoader(self.train_dataset,
                                      self.batch_size,          # How many samples per batch to load
                                      shuffle=True)             # Allow the data to reshuffle at every sampling
        print_every = int(len(train_dataloader) / 10)

        for batch_idx, (inputs, targets) in enumerate(train_dataloader):
            # inputs: A tensor of size batch size,
            #           where each sample is of the input dimension (For images for example: of size 3 X Width X Height)
            # targets: A tensor of size batch size - of the classification for each of the samples.
            inputs : torch.Tensor
            targets : torch.Tensor

            # Move the data to the device
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the gradients - i.e reset the gradients of all parameters in the optimizer to zero.
            # This is a critical step in the training process because gradients accumulate by default in PyTorch,
            # which can lead to incorrect updates if not cleared properly.
            self.optimizer.zero_grad()

            # Forward pass
            current_predictions : torch.Tensor = self.model(inputs)

            # Compute the loss
            loss : torch.Tensor = self.criterion(current_predictions, targets)

            # Backward pass.
            # In the background using the given loss function, and activation function, the gradients functions are
            # calculated. Those along with the forward pass values give all the inputs needed to run the backward
            # algorithm and compute all gradients.
            loss.backward()

            # Optimizer step.
            # Un the background all the weights of the NN are updated according to the calculated gradients.
            self.optimizer.step()

            # Update total loss and accuracy metrics
            total_loss += loss.item()
            nof_samples += targets.size(0)
            correct_labeled_samples += (current_predictions.argmax(dim=1) == targets).sum().item()

            # Calculate average loss and accuracy
            avg_loss : float = total_loss / (batch_idx + 1)
            accuracy : float = 100.0 * correct_labeled_samples / nof_samples

            if batch_idx % print_every == 0 or \
                    batch_idx == len(train_dataloader) - 1:
                print(f'Epoch [{self.epoch:03d}] | Loss: {avg_loss:.3f} | '
                      f'Acc: {accuracy:.2f}[%] '
                      f'({correct_labeled_samples}/{nof_samples})')

        return avg_loss, accuracy

    def evaluate_model_on_dataloader(
            self, dataset: torch.utils.data.Dataset) -> tuple[float, float]:
        """Evaluate model loss and accuracy for dataset.

        Args:
            dataset: the dataset to evaluate the model on.

        Returns:
            (avg_loss, accuracy): tuple containing the average loss and
            accuracy across all dataset samples.
        """
        # Set the model to evaluation mode
        # This is essential because certain layers and behaviors in a torch NN behave differently
        # during training and evaluation
        self.model.eval()

        dataloader = DataLoader(dataset,
                                batch_size=self.batch_size,
                                shuffle=False)
        total_loss = 0
        avg_loss = 0
        accuracy = 0
        nof_samples = 0
        correct_labeled_samples = 0
        print_every = max(int(len(dataloader) / 10), 1)

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # inputs: A tensor of size batch size,
            #           where each sample is of the input dimension (For images for example: of size 3 X Width X Height)
            # targets: A tensor of size batch size - of the classification for each of the samples.
            inputs: torch.Tensor
            targets: torch.Tensor

            inputs, targets = inputs.to(device), targets.to(device)

            # Disable gradient computation
            with torch.no_grad():
                # Forward pass
                pred : torch.Tensor = self.model(inputs)

                # Compute the loss
                loss : torch.Tensor = self.criterion(pred, targets)

            # Update total loss and accuracy metrics
            total_loss += loss.item()
            nof_samples += targets.size(0)
            correct_labeled_samples += (pred.argmax(dim=1) == targets).sum().item()

            # Calculate average loss and accuracy
            avg_loss : float = total_loss / (batch_idx + 1)
            accuracy : float = 100.0 * correct_labeled_samples / nof_samples

            if batch_idx % print_every == 0 or batch_idx == len(dataloader) - 1:
                print(f'Epoch [{self.epoch:03d}] | Loss: {avg_loss:.3f} | '
                      f'Acc: {accuracy:.2f}[%] '
                      f'({correct_labeled_samples}/{nof_samples})')

        return avg_loss, accuracy

    def validate(self):
        """Evaluate the model performance."""
        return self.evaluate_model_on_dataloader(self.validation_dataset)

    def test(self):
        """Test the model performance."""
        return self.evaluate_model_on_dataloader(self.test_dataset)

    @staticmethod
    def write_output(logging_parameters: LoggingParameters, data: dict):
        """Write logs to json.

        Args:
            logging_parameters: LoggingParameters. Some parameters to log.
            data: dict. Holding a dictionary to dump to the output json.
        """
        if not os.path.isdir(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        output_filename = f"{logging_parameters.dataset_name}_" \
                          f"{logging_parameters.model_name}_" \
                          f"{logging_parameters.optimizer_name}.json"
        output_filepath = os.path.join(OUTPUT_DIR, output_filename)

        print(f"Writing output to {output_filepath}")
        # Load output file
        if os.path.exists(output_filepath):
            # pylint: disable=C0103
            with open(output_filepath, 'r', encoding='utf-8') as f:
                all_output_data = json.load(f)
        else:
            all_output_data = []

        # Add new data and write to file
        all_output_data.append(data)
        # pylint: disable=C0103
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(all_output_data, f, indent=4)

    def run(self, epochs, logging_parameters: LoggingParameters):
        """Train, evaluate and test model on dataset, finally log results."""
        output_data = {
            "model": logging_parameters.model_name,
            "dataset": logging_parameters.dataset_name,
            "optimizer": {
                "name": logging_parameters.optimizer_name,
                "params": logging_parameters.optimizer_params,
            },
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "test_loss": [],
            "test_acc": [],
        }
        best_acc = 0
        model_filename = f"{logging_parameters.dataset_name}_" \
                         f"{logging_parameters.model_name}_" \
                         f"{logging_parameters.optimizer_name}.pt"
        checkpoint_filename = os.path.join(CHECKPOINT_DIR, model_filename)
        for self.epoch in range(1, epochs + 1):
            print(f'Epoch {self.epoch}/{epochs}')

            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc = self.validate()
            test_loss, test_acc = self.test()

            output_data["train_loss"].append(train_loss)
            output_data["train_acc"].append(train_acc)
            output_data["val_loss"].append(val_loss)
            output_data["val_acc"].append(val_acc)
            output_data["test_loss"].append(test_loss)
            output_data["test_acc"].append(test_acc)

            # Save checkpoint
            # For each epoch where the evaluation dataset accuracy is improved,
            # We wish to save a state too the checkpoint file
            if val_acc > best_acc:
                print(f'Saving checkpoint {checkpoint_filename}')
                state = {
                    'model': self.model.state_dict(),
                    'val_acc': val_acc,
                    'test_acc': test_acc,
                    'epoch': self.epoch,
                }
                torch.save(state, checkpoint_filename)
                best_acc = val_acc
        self.write_output(logging_parameters, output_data)
