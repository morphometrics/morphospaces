from pathlib import Path
from typing import Tuple

import typer
from typing_extensions import Annotated

app = typer.Typer(
    help="Skeletonize segementations with a multiscale 3D UNet.",
    no_args_is_help=True,
)


@app.command(no_args_is_help=True)
def train(
    train_data_pattern: Annotated[
        str,
        typer.Argument(
            help="The glob pattern to use to find the training data."
        ),
    ],
    val_data_pattern: Annotated[
        str,
        typer.Argument(
            help="The glob pattern to use to find the validation data."
        ),
    ],
    logdir_path: Annotated[
        Path,
        typer.Argument(help="The directory to save the logs and checkpoints."),
    ],
    batch_size: Annotated[
        int,
        typer.Option(help="The number of patches to include in each batch."),
    ] = 2,
    patch_shape: Annotated[
        Tuple[int, int, int],
        typer.Option(
            help="The shape of the patches to extract from the data."
        ),
    ] = (120, 120, 120),
    patch_stride: Annotated[
        Tuple[int, int, int],
        typer.Option(help="The stride to use when extracting patches."),
    ] = (120, 120, 120),
    patch_threshold: Annotated[
        float,
        typer.Option(
            help="The minimum fraction of voxels in the patch that must"
            " include a positive label to be included in training."
        ),
    ] = 0.01,
    lr: Annotated[
        float, typer.Option(help="The learning rate for the optimizer.")
    ] = 0.0004,
    skeletonization_target: Annotated[
        str,
        typer.Option(
            help="The key in the HDF5 file to use for "
            "the skeletonization ground truth."
        ),
    ] = "skeletonization_target",
    log_every_n_iterations: Annotated[
        int, typer.Option(help="The number of iterations " "between logging.")
    ] = 25,
    val_check_interval: Annotated[
        float,
        typer.Option(
            help="The interval for validation in fractions of an epoch."
        ),
    ] = 0.25,
    lr_reduction_patience: Annotated[
        int,
        typer.Option(
            help="The number of epochs to wait "
            "before reducing the learning rate."
        ),
    ] = 25,
    lr_scheduler_step: Annotated[
        int,
        typer.Option(
            help="The number of iterations between "
            "learning rate scheduler updates"
        ),
    ] = 1000,
    accumulate_grad_batches: Annotated[
        int,
        typer.Option(
            help="The number of batches to accumulate "
            "before performing a backward pass."
        ),
    ] = 4,
    seed_value: Annotated[
        int, typer.Option(help="The value to use for " "seeding the RNGs.")
    ] = 42,
    n_dataset_workers: Annotated[
        int,
        typer.Option(
            help="The number of workers to use for each dataset loader ."
        ),
    ] = 4,
    gpus: Annotated[
        Tuple[int, ...], typer.Option(help="The indices of the GPUs to use.")
    ] = (0,),
):
    """Train the multiscale skeletonization network."""
    # lazy import to save time when user just prompts for help
    from morphospaces.training.multiscale_skeletonization import (
        train as train_func,
    )

    if type(gpus) is int:
        # coerce GPUS input to tuple
        gpus = (gpus,)
    train_func(
        train_data_pattern=train_data_pattern,
        val_data_pattern=val_data_pattern,
        logdir_path=logdir_path,
        batch_size=batch_size,
        patch_shape=patch_shape,
        patch_stride=patch_stride,
        patch_threshold=patch_threshold,
        lr=lr,
        skeletonization_target=skeletonization_target,
        log_every_n_iterations=log_every_n_iterations,
        val_check_interval=val_check_interval,
        lr_reduction_patience=lr_reduction_patience,
        lr_scheduler_step=lr_scheduler_step,
        accumulate_grad_batches=accumulate_grad_batches,
        seed_value=seed_value,
        n_dataset_workers=n_dataset_workers,
        gpus=gpus,
    )


@app.command(no_args_is_help=True)
def predict(username: str):
    """Predict with the skeletonization netowrk."""
    print("Not implemented yet!")


if __name__ == "__main__":
    app()
