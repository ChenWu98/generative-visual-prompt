import os

import torch
from torchvision import utils


def save_images(images: torch.Tensor, output_dir: str, file_prefix: str, nrows: int, iteration: int) -> None:
    utils.save_image(
        images,
        os.path.join(output_dir, f"{file_prefix}_{str(iteration).zfill(6)}.png"),
        nrow=nrows,
    )
