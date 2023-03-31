"""Train a model on the data using given parameters"""
import argparse
from typing import Union

import pandas as pd
import xarray as xr
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


from network.model import Model
from network.model_config import ModelConfig
from network.window_iter_ds import WindowIterDS
from args_setup import setup_train_args

TARGET_FEATS = ['t2m', 'tp']


def get_args() -> argparse.Namespace:
    """
    Gets and checks the command line arguments for errors
    """
    parser = setup_train_args()
    args = parser.parse_args()

    # Max workers is positive integer
    assert args.max_workers >= 0, 'Max workers must be a positive integer'

    return args


class LightningModel(pl.LightningModule):
    def __init__(self,
                 model_type: str,
                 dropout: float,
                 kernel_size: Union[int, tuple[int]],
                 config: ModelConfig,
                 learning_rate: float = 0.001):
        super().__init__()
        self.model = Model.from_dataset(config,
                                        model_type=model_type,
                                        dropout=dropout,
                                        kernel_size=kernel_size)
        self.learning_rate = learning_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def setup_loaders_config(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, DataLoader, ModelConfig]:
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

    # Create the datasets from window iter ds
    train_data = WindowIterDS(train_data,
                              args.context_steps,
                              args.horizon,
                              args.target_feats)

    val_data = WindowIterDS(val_data,
                            args.context_steps,
                            args.horizon,
                            args.target_feats)

    test_data = WindowIterDS(test_data,
                             args.context_steps,
                             args.horizon,
                             args.target_feats)

    # Create the model config
    config = ModelConfig(train_data)

    print("Data loaded, creating dataloaders...")

    # Data loaders Training data can't have multiple workers or it will crash most likely due to its size
    train_loader = DataLoader(train_data,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=0)

    val_loader = DataLoader(val_data,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.max_workers)

    test_loader = DataLoader(test_data,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.max_workers)

    return train_loader, val_loader, test_loader, config


if __name__ == '__main__':
    args = get_args()

    # Set up the data loaders and model config
    train_loader, val_loader, test_loader, train_data = setup_loaders_config(args)

    # Create the model
    model = LightningModel('conv_lstm', 0.2, (3, 3),
                           train_data, args.learning_rate)
    print(model.model.summarize("full"))
    print(model.model.hparams)

    # Create the callbacks
    early_stop_callback = EarlyStopping(monitor='val_loss',
                                        min_delta=0.00,
                                        patience=3,
                                        verbose=False,
                                        mode='min')
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          dirpath=args.model_dir,
                                          filename='model-{epoch:02d}-{val_loss:.2f}',
                                          save_top_k=1,
                                          mode='min')

    # Create the trainer
    trainer = pl.Trainer(gpus=1,
                         max_epochs=args.epochs,
                         callbacks=[early_stop_callback, checkpoint_callback],
                         logger=pl.loggers.TensorBoardLogger(args.log_dir))

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Save the model
    trainer.save_checkpoint(f'{args.model_dir}/{args.model_name}.ckpt')

    # Test the model
    trainer.test()

    # Save the model config
    model.config.save(args.model_dir)

    # Save the model hyperparameters
    pd.DataFrame({
        'epochs': [args.epochs],
        'batch_size': [args.batch_size],
        'learning_rate': [args.learning_rate],
        'model': [args.model],
        'train_path': [args.train_path],
        'val_path': [args.val_path],
        'target_lon': [args.target_lon],
        'target_lat': [args.target_lat],
        'target_cubed_size': [args.target_cubed_size],
        'context_cube_size': [args.context_cubed_size],
        'context_downsampling_factor': [args.context_downsampling_factor],
        'context_time_size': [args.context_time_size],
        'prediction_delta': [args.prediction_delta]
    }).to_csv(f'{args.model_dir}/hyperparameters_{args.model_name}.csv', index=False)
