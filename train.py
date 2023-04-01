"""Train a model on the data using given parameters"""
import argparse

import numpy as np
import pandas as pd
import torch
import tqdm
import xarray as xr
from torch import nn
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import IterableWrapper

import wandb
from args_setup import setup_train_args
from model_utils import EarlyStopping, r2_loss
from network.model_config import ModelConfig
from network.models.conv_lstm_model import ConvLSTMModel
from network.window_iter_ds import WindowIterDS


def get_args() -> argparse.Namespace:
    """
    Gets and checks the command line arguments for errors
    """
    parser = setup_train_args()
    args = parser.parse_args()

    # Assert horizon and context steps are both positive integers
    assert args.horizon > 0, 'Horizon must be a positive integer'
    assert args.context_steps > 0, 'Context steps must be a positive integer'

    return args


def setup_loaders_config(
        args: argparse.Namespace) -> tuple[DataLoader,
                                           DataLoader,
                                           DataLoader,
                                           ModelConfig,
                                           torch.device]:
    """
    Sets up the data loaders and model config

    Args:
        args (argparse.Namespace): The command line arguments

    Returns:
        tuple[DataLoader, DataLoader, DataLoader, ModelConfig]: The train, val, test data loaders and the model config
    """

    print("Preparing data...")

    # Load the data from netcdf files
    train_data = xr.open_dataset(args.train_path)
    val_data = xr.open_dataset(args.val_path)
    test_data = xr.open_dataset(args.val_path)

    print("Data loaded, creating datasets...")

    # Geat feature variables from train_data
    feature_variables = train_data.data_vars.keys()

    # Check that the target features are in the feature variables
    for feat in args.target_feats:
        assert feat in feature_variables, f'{feat} is not a valid feature'

    device = torch.device(
        "cpu") if not torch.cuda.is_available() else torch.device("cuda")

    # Create the datasets from window iter ds
    train_data = WindowIterDS(train_data,
                              args.context_steps,
                              args.horizon,
                              args.target_feats,
                              device)

    val_data = WindowIterDS(val_data,
                            args.context_steps,
                            args.horizon,
                            args.target_feats,
                            device)

    test_data = WindowIterDS(test_data,
                             args.context_steps,
                             args.horizon,
                             args.target_feats,
                             device)

    # Create the model config
    config = ModelConfig(train_data, wb_config=wandb.config)

    if args.debug:
        print("Configuration:")
        print(f'\tInput Channels: {config.input_chans}')
        print(f'\tOutput Channels: {config.output_chans}')
        print(f'\tHidden Layers: {config.hidden_layers}')
        print(f'\tLSTM Channels: {config.lstm_chans}')

    train_dp = IterableWrapper(train_data)
    val_dp = IterableWrapper(val_data)
    test_dp = IterableWrapper(test_data)

    # Create the data loaders
    train_loader = DataLoader(train_dp,
                              batch_size=wandb.config.batch_size,
                              shuffle=True)

    val_loader = DataLoader(val_dp,
                            batch_size=wandb.config.batch_size,
                            shuffle=False)

    test_loader = DataLoader(test_dp,
                             batch_size=wandb.config.batch_size,
                             shuffle=False)

    return train_loader, val_loader, test_loader, config, device


def fit(max_epochs: int, model: nn.Module, optimizer: torch.optim, train_loader, val_loader, loss_fn,
        scheduler: torch.optim.lr_scheduler, device: torch.device):
    model = model.to(device)

    early_stopping = EarlyStopping(tolerance=5)

    pbar = tqdm.tqdm(range(max_epochs), desc='Epoch', position=0)
    for epoch in pbar:
        train_mloss, train_r2 = training_loop(
            model, optimizer, train_loader, loss_fn)
        val_mloss, val_r2 = validation_loop(
            model, val_loader, loss_fn)

        pbar.set_postfix({'Prev Val Loss': val_mloss, 'Prev Val R2': val_r2})
        wandb.log({'Epoch Loss/val': val_mloss, 'Epoch R2/val': val_r2, 'Epoch Loss/train': train_mloss,
                   'Epoch R2/train': train_r2, 'epoch': epoch})

        if early_stopping(val_mloss):
            # Log early stopping as event
            wandb.log({'Early Stopping': True})
            break

        scheduler.step(val_mloss)


def training_loop(model: nn.Module, optimizer: torch.optim, train_loader, loss_fn):
    model.train()
    horizon = wandb.config.horizon
    losses = []
    r2s = []
    pbar = tqdm.tqdm(enumerate(train_loader), desc='Training Batch', position=1,
                     leave=False, total=len(train_loader))
    for batch_idx, batch in pbar:
        x, y = batch

        forecast_step = np.random.randint(1, horizon)
        y_hat = model(x, forecast_step)
        loss = loss_fn(y_hat,
                       y[:, forecast_step, :])

        r2 = r2_loss(y_hat, y[:, forecast_step, :])

        count = 1
        for step in range(forecast_step, horizon):
            y_hat = model(x, torch.tensor(step).long())
            loss += loss_fn(y_hat,
                            y[:, step, :])
            count += 1

            r2 += r2_loss(y_hat, y[:, step, :])

        loss /= count
        r2 /= count

        losses.append(loss.item())
        r2s.append(r2.item())

        if batch_idx % 50 == 0:
            wandb.log({'Loss/train': np.mean(losses)})
            wandb.log({'R2/train': np.mean(r2s)})

        pbar.set_postfix({'Loss': np.mean(losses), 'R2': np.mean(r2s)})

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return np.mean(losses), np.mean(r2s)


def validation_loop(model: nn.Module, val_loader, loss_fn):
    """
        Return mean loss over the validation set
    """
    model.eval()

    horizon = wandb.config.horizon
    losses = []
    r2s = []
    pbar = tqdm.tqdm(enumerate(val_loader), desc='Validation Batch', position=1,
                     leave=False, total=len(val_loader))
    for batch_idx, batch in pbar:
        x, y = batch

        forecast_step = np.random.randint(1, horizon)
        y_hat = model(x, forecast_step)
        loss = loss_fn(y_hat,
                       y[:, forecast_step, :])
        r2 = r2_loss(y_hat, y[:, forecast_step, :])

        count = 1

        for step in range(forecast_step, horizon):
            y_hat = model(x, torch.tensor(step).long())
            loss += loss_fn(y_hat,
                            y[:, step, :])
            r2 += r2_loss(y_hat, y[:, step, :])
            count += 1

        loss /= count
        r2 /= count

        losses.append(loss.item())
        r2s.append(r2.item())

        if batch_idx % 50 == 0:
            wandb.log({'Loss/val': np.mean(losses), })

        pbar.set_postfix({'Loss': np.mean(losses), 'R2': np.mean(r2s)})

    return np.mean(losses), np.mean(r2s)


def test(model: nn.Module, test_loader, loss_fn, device: torch.device):
    """
        Return mean loss over the test set.

        Also logs to wandb
    """
    model = model.to(device)

    model.eval()

    horizon = wandb.config.horizon
    losses = []
    r2s = []
    for batch_idx, batch in tqdm.tqdm(enumerate(test_loader), desc='Test Batch', position=1,
                                      total=len(test_loader)):
        x, y = batch

        forecast_step = np.random.randint(1, horizon)
        y_hat = model(x, forecast_step)
        loss = loss_fn(y_hat,
                       y[:, forecast_step, :])
        r2 = r2_loss(y_hat, y[:, forecast_step, :])

        count = 1

        for step in range(forecast_step, horizon):
            y_hat = model(x, torch.tensor(step).long())
            loss += loss_fn(y_hat,
                            y[:, step, :])
            r2 += r2_loss(y_hat, y[:, step, :])
            count += 1

        loss /= count
        r2 /= count
        losses.append(loss.item())
        r2s.append(r2.item())

    test_mloss = np.mean(losses)
    test_r2 = np.mean(r2s)
    wandb.log({'Loss/test': test_mloss})
    wandb.log({'R2/test': test_r2})

    # Print the test loss
    print(f'Test Loss: {test_mloss:.4f}')
    print(f'Test R2: {test_r2:.4f}')
    return test_mloss, test_r2


def main(args: argparse.Namespace):
    # Set up the data loaders and model config
    train_loader, val_loader, test_loader, config, device = setup_loaders_config(
        args)

    # Create the model
    if args.model_type == 'conv_lstm':
        model = ConvLSTMModel(0.7,
                              config)
    else:
        raise NotImplementedError(
            f'{args.model_type} is not a valid model type')

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                              patience=3,
                                                              verbose=True)

    wandb.watch(model, criterion=loss_fn, log='all', log_freq=100)

    profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA
        ],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(args.log_dir),
        with_stack=True
    )

    with profiler as prof:
        # Train the model
        fit(args.epochs, model, optimizer, train_loader, val_loader, loss_fn,
            lr_scheduler, device)

        # Test the model
        test(model, test_loader, loss_fn, device)

    print("Finished training and testing, saving model and profile")
    # Save the model
    model_art = wandb.Artifact(args.model_name, type='model')
    model_art.add_dir(args.model_dir)
    model_art.save()

    profile_art = wandb.Artifact(f'{args.model_name}_trace', type='profile')
    profile_art.add_dir(args.log_dir)
    profile_art.save()


if __name__ == '__main__':
    args = get_args()

    wandb.init(project='weather-prediction', config=args)

    main(args)
