"""Train a model on the data using given parameters"""
import argparse

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import xarray as xr
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from network.data_pipeline import get_ts_dataset
from network.model import Model
from network.model_config import ModelConfig

TARGET_FEATS = ['t2m', 'tp']


def get_args() -> argparse.Namespace:
    """Returns the command line arguments"""

    parser = argparse.ArgumentParser(
        description='Train a model on the data using given parameters')

    parser.add_argument(
        '--epochs',
        type=int,
        default=4,
        help='Number of epochs to train the model for')

    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size to use when training the model')

    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Learning rate to use when training the model')

    parser.add_argument(
        '--model_type',
        type=str,
        default='conv_lstm',
        help='Model type to use [conv_lstm, lstm, gru]. Note that non,\
            Convoluted LSTM models can only handle single point outputs. Default: conv_lstm.')

    parser.add_argument(
        '--train_path',
        type=str,
        default='cache/train.nc',
        help='Path to the netCDF4 file to train the model on')

    parser.add_argument(
        '--val_path',
        type=str,
        default='cache/val.nc',
        help='Path to the netCDF4 file to validate the model on')

    parser.add_argument(
        '--model_dir',
        type=str,
        default='models',
        help='Directory to save the trained model to')

    parser.add_argument(
        '--model_name',
        type=str,
        default='model',
        help='Name of the model to save'
    )

    parser.add_argument(
        '--log_dir',
        type=str,
        default='logs',
        help='Directory to save the training logs to')

    # Target longitude and latitude
    parser.add_argument(
        '--target_lon',
        type=float,
        default=-80,
        help='Target longitude to predict')

    parser.add_argument(
        '--target_lat',
        type=float,
        default=43,
        help='Target latitude to predict')

    # Target cubed size
    parser.add_argument(
        '--target_apothem',
        type=int,
        default=3,
        help='Target apothem to predict')

    # Context cubed size
    parser.add_argument(
        '--context_apothem',
        type=int,
        default=3,
        help='Context apothem for input')

    # Context downsampling factor
    parser.add_argument(
        '--context_downsampling_factor',
        type=int,
        default=2,
        help='Context downsampling factor for input')

    # Context time size
    parser.add_argument(
        '--context_steps',
        type=int,
        default=6,
        help='Context time steps (hours) used for input')

    # Prediction delta
    parser.add_argument(
        '--target_steps',
        type=int,
        default=3,
        help='Target time steps (hours). How many hours to predict into the future')

    # Debug mode
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Debug mode. Prints extra information')

    return parser.parse_args()


def load_data(train_path: str, val_path: str) -> tuple[xr.Dataset, xr.Dataset]:
    """Loads the data from the given paths

    Args:
        train_path (str): Path to the netCDF4 file to train the model on
        val_path (str): Path to the netCDF4 file to validate the model on

    Returns:
        tuple[xr.Dataset, xr.Dataset]: The training and validation data
    """
    # Load the training data
    train_data = xr.open_dataset(train_path)

    # Load the validation data
    val_data = xr.open_dataset(val_path)

    return train_data, val_data


class LightningModel(pl.LightningModule):
    def __init__(self,
                 config: ModelConfig,
                 learning_rate: float = 0.001):
        super().__init__()
        self.config = config
        self.model = Model(0.3, self.config)
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


if __name__ == '__main__':
    args = get_args()

    print("Preparing data...")

    # Load the data
    train_data, val_data = load_data(args.train_path, args.val_path)

    # Geat feature variables from train_data
    feature_variables = list[train_data.variables]

    train_tsds = get_ts_dataset(train_data,
                                TARGET_FEATS,
                                context_steps=args.context_steps,
                                target_steps=args.target_steps,
                                target_lon=args.target_lon,
                                target_lat=args.target_lat,
                                context_apothem=args.context_apothem,
                                debug=args.debug
                                )
    print("Train dataset created")

    val_tsds = get_ts_dataset(val_data,
                              TARGET_FEATS,
                              context_steps=args.context_steps,
                              target_steps=args.target_steps,
                              target_lon=args.target_lon,
                              target_lat=args.target_lat,
                              context_apothem=args.context_apothem,
                              debug=args.debug
                              )
    
    print("Validation dataset created")

    # Data loaders
    train_loader = DataLoader(train_tsds,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=4)
    val_loader = DataLoader(val_tsds,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=4)

    # Get batch dimensions
    batch = next(iter(train_loader))
    batch_dims = batch[0].shape

    print(f'Batch dimensions: {batch_dims}')
    print(f'Input channels: {batch_dims[2]}')

    config = ModelConfig(
        feature_variables,
        TARGET_FEATS,
        model_type=args.model_type,
        context_apothem=args.context_apothem,
        context_steps=args.context_steps,
        target_apothem=args.target_apothem,
        target_steps=args.target_steps,
        input_chans=batch_dims[2],
    )

    # Create the model
    model = LightningModel(args.learning_rate)

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
