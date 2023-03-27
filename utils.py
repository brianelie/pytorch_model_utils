'''
PyTorch Utilities Functions
Author: Brian E. Harrington

This covers several key utility functions for pytorch models.
This allows more semantic code elsewhere and consistency across models.
'''
import numpy as np
import pandas as pd

from tqdm import tqdm

import torch
import torch.nn as nn
from bayesian_torch.models.dnn_to_bnn import get_kl_loss

class EarlyStopping():
    """Class designed for early stopping functionality.
    Used as a contained manner for specific variables
    """

    def __init__(self, val_patience=None, model_path='.'):
        self.val_best = float('inf')
        self.val_patience = val_patience
        self.model_path = model_path
        self.best_epoch = 0

    def __call__(self, epoch, val_loss, model):
        """Returns True if early stopping is triggered, otherwise returns False

        Args:
            epoch (int): Current Epoch
            val_loss (float): Epoch Validation Loss

        Returns:
            bool: Early stopping condition
        """
        # No early stopping is used
        if not self.val_patience:
            return False

        # New best validation loss
        if val_loss < self.val_best:
            self.val_best = val_loss
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.model_path)

        # Exceeded validation patience without an improvement in validation loss
        elif (epoch - self.best_epoch) >= self.val_patience:
            print('Stopping on early stopping',
                  f'best epoch: {self.best_epoch} with val loss: {self.val_best:.3f}')
            return True
        return False


def train(model, dataloads, criterion, optimizer, metrics=None,
          epochs=10, val_patience=None, model_path='.', bayesian=False):
    """Generic Train Function for Pytorch model

    Args:
        model (nn.Module): Pytorch model to train
        dataloads (dict): Dictionary of dataloaders with keys "train" and "val"
        criterion (nn.Module.Loss): Loss Function to minimize.
        Does not include KL divergence for bayesian models
        optimizer (torch.optim): Optimizer
        metrics (dict, optional): Dictionary of torchmetrics. Expects key for
        'train', 'val', and 'test' for each of their datasets respectively. 
        Also looks for the keys 'short' and 'name' which should be strings for
        how to label the metric. 'short' is for progress bar display, and
        'name' is for learning curves
        epochs (int, optional): Number of Epochs to train. Defaults to 10.
        val_patience (int, optional): Number of epochs to train
        beyond lowest validation loss before stopping. Defaults to None.
        model_path (str, optional): Path to save model to. Defaults to '.'.
        bayesian (bool, optional): Flag for bayesian models. If True, uses get_kl_loss()
        from Bayesian Torch to include kl divergence in loss function. Defaults to False.

    Returns:
        results (pd.DataFrame): DataFrame with columns for train/val loss
        and train/val metrics to review learning curves
        model (nn.Module): Trained model
    """
    # Use GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = pd.DataFrame(columns=['train_loss', 'val_loss'])
    if metrics is not None:
        results[f'train_{metrics["short"]}'] = None
        results[f'val_{metrics["short"]}'] = None

    # Instantiates EarlyStopping() class. This also saves the model weights on new best
    early_stopping = EarlyStopping(val_patience, model_path)

    for epoch in range(1, epochs+1):
        with tqdm(total=len(dataloads['train']), unit='batches') as tepoch:
            tepoch.set_description(f'Epoch: {epoch}/{epochs}')
            # Each epoch interates twice: for train and val
            for phase in ['train', 'val']:
                # Set model to train/eval as necessary
                # Used for dropout and batch norm and other layers
                # That have different functionality for each
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0

                for num, (data, y_true) in enumerate(dataloads[phase]):
                    # Put data, y_true on GPU if available
                    data = data.to(device)
                    y_true = y_true.to(device)

                    # Zero out gradients
                    optimizer.zero_grad()

                    # Allow gradient updates in train only
                    with torch.set_grad_enabled(phase == 'train'):
                        # Forward pass
                        y_pred = model(data)

                        # Loss calculation
                        ce_loss = criterion(y_pred, y_true)

                        # If Bayesian, add in kl divergence
                        if bayesian:
                            kl_loss = get_kl_loss(model)
                            loss = ce_loss + kl_loss / dataloads[phase].batch_size
                        else:
                            loss = ce_loss

                        if phase == 'train':
                            # Backprop and weight updates
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item()
                    metrics[phase].update(y_pred, y_true)

                    epoch_loss = running_loss / (num+1)
                    epoch_metric = metrics[phase].compute()

                    results.loc[epoch, f'{phase}_loss'] = epoch_loss
                    results.loc[epoch, f'{phase}_acc'] = epoch_metric.cpu(
                    ).numpy().round(decimals=4)

                    # Set epoch stats on tqdm
                    tepoch.set_postfix(results.loc[epoch].to_dict())
                    if phase == 'train':
                        tepoch.update(1)

        # Zero out metrics each epoch
        metrics[phase].reset()

        # Evaluate early stopping and set best weights and stop if necessary
        if early_stopping(epoch, epoch_loss, model):
            model.load_state_dict(torch.load(model_path))
            break

    return results


def learning_curves(results, fig, metrics):
    """Create learning curves plot

    Args:
        results (pd.DataFrame): Expects output of train function
        lr_curve_path (str): path to save the learning curves to
    """
    axes = fig.subplots(nrows=2, ncols=1, sharex='col')
    loss_ax, met_ax = axes.flat
    loss_ax.plot(results['train_loss'], label='Train Loss', color='y')
    loss_ax.plot(results['val_loss'], label='Val Loss', color='b')
    loss_ax.set_ylabel("Loss")
    loss_ax.legend()

    met_ax.plot(results[f'train_{metrics["short"]}'],
                label=f'Train {metrics["name"]}', color='y')
    met_ax.plot(results[f'val_{metrics["short"]}'],
                label=f'Val {metrics["name"]}', color='b')
    met_ax.set_xlabel("Epochs")
    met_ax.set_ylabel(metrics["name"])
    met_ax.legend()
    
def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def evaluate(model, dataload, criterion, metrics, dropout=False):
    """Evaluates a trained pytorch model

    Args:
        model (nn.Module): trained pytorch model
        dataload (pytorch DataLoader): dataloader to evaluate
        criterion (nn.Module.Loss): Loss function to use
        dropout (boolean): If true, sets all dropout layers to train mode

    Returns:
        y_pred_all: logits for all data samples in dataload
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with tqdm(total=len(dataload), unit='batches') as tepoch:
        running_loss = 0

        model.eval()
        enable_dropout(model)

        y_pred_all = None

        for data, y_true in dataload:
            data = data.to(device)
            y_true = y_true.to(device)

            with torch.no_grad():
                y_pred = model(data)
                loss = criterion(y_pred, y_true)

            running_loss += loss.item()
            metrics['test'].update(y_pred, y_true)

            epoch_loss = running_loss / len(dataload)
            epoch_metric = metrics['test'].compute()

            # We apply softmax here because we will use the logit outputs for
            # uncertainty quantification
            y_pred = nn.Softmax(dim=1)(y_pred)

            # Stack all predictions into one torch array
            if y_pred_all is not None:
                y_pred_all = torch.vstack((y_pred_all, y_pred))
            else:
                y_pred_all = y_pred

            tepoch.set_postfix(
                {'loss': epoch_loss, 'acc': epoch_metric.cpu().numpy().round(decimals=4)})
            tepoch.update(1)

    return y_pred_all

def uncertainty_quantification(y_preds):
    """Calculates necessary prediction metrics and parameters for a bayesian model:
        -- preds (n_samples, n_classes): Result of bayesian model averaging over mc inferences
        -- max_pred (n_samples): Predicted class
        -- var (n_samples): Variance over mc inferences
        -- entropy (n_samples): entropy of average predictions
        -- aleatoric (n_samples): aleatoric uncertainty
        -- epistemitc (n_samples): epistemic uncertainty
        
    Aleatoric and Epistemic Uncertainy are calculated using the method in 
    https://openreview.net/forum?id=Sk_P2Q9sG

    Args:
        y_preds (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Instantiate necessary variables
    mc_inf, n_samples, n_classes = y_preds.shape
    preds = np.zeros((n_samples, n_classes))
    max_pred = np.zeros((n_samples))
    var = np.zeros((n_samples))
    aleatoric = np.zeros((n_samples))
    epistemic = np.zeros((n_samples))
    entropy = np.zeros((n_samples))

    for i in range(n_samples):
        # Pred_val (mc_inf, n_classes) - All predictions for a single sample
        pred_val = y_preds[:,i,:].numpy()

        preds[i] = np.mean(pred_val, axis=0)
        max_pred[i] = np.argmax(preds[i])

        var[i] = np.sum(np.var(pred_val, axis=0))
        entropy[i] = -np.sum(preds[i]*np.log2(preds[i]+1e-14))

        # We take the trace of the matrix because off-diagonal components are smalld
        # This allows for an easier analysis if each sample has a single value vs a matrix
        aleatoric[i] = np.trace(np.diag(preds[i])-pred_val.T.dot(pred_val)/mc_inf)
        tmp = pred_val - preds[i]
        epistemic[i] = np.trace(tmp.T.dot(tmp)/mc_inf)

    return {'preds':torch.Tensor(preds), 'max_pred':torch.Tensor(max_pred.astype(int)), 'var': var,
            'entropy': entropy, 'aleatoric': aleatoric, 'epistemic': epistemic}


def calibrated_uncertainty(stats, y_true, metric, total_classes, masks):
    """Generates data for a calibrated uncertainty plot

    Args:
        stats (dict): output of uncertainty quantification
        y_true (torch.tensor): True values
        metric (torchmetric): torchmetric metric to evaluate on
        total_classes (int): Total number of classes (used to normalize entropy)
        masks (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Normalize entropy for the masks to work universally
    entropy = stats['entropy']/np.log2(total_classes)
    metric_val = []
    data = []
    for mask in masks:
        mask_entropy = entropy <= mask
        entropy_filt = entropy[mask_entropy]
        # Make sure entropy is not empty (no entropy values less than mask/10)
        if len(entropy_filt) > 0:
            y_true_filt = y_true[mask_entropy]
            y_pred_filt = stats['max_pred'][mask_entropy]
            metric_val.append(metric(y_pred_filt, y_true_filt).item())
            data.append(len(y_true_filt)/len(y_true))
    return metric_val, data
