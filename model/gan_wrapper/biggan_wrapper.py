# Created by Chen Henry Wu
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from ..lib.biggan import BigGAN
from ..model_utils import requires_grad


class BigGANWrapper(torch.nn.Module):

    def __init__(self, args):
        super(BigGANWrapper, self).__init__()

        self.args = args

        # Set up generator
        self.generator = BigGAN.from_pretrained(pretrained_model_name_or_path=args.model_name_or_path)
        self.latent_dim = self.generator.config.z_dim
        self.y_latent_dim = self.generator.config.z_dim  # Same as z_dim
        # Freeze.
        requires_grad(self.generator, False)

        self.y_scale = torch.mean(
            torch.mean(
                self.generator.embeddings.weight.data ** 2,
                dim=0,
            ),
            dim=0,
        ) ** 0.5
        # print('y_scale:', self.y_scale)

        # Post process.
        self.post_process = transforms.Compose(  # To un-normalize from [-1.0, 1.0] (GAN output) to [0, 1]
            [transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])]
        )

    def forward(self, z, y):
        # Eval mode for the generator.
        self.generator.eval()

        # Truncation.
        z = torch.clamp(z, -2.0, 2.0)
        z = self.args.sample_truncation * z

        img = self.generator(z=z, y=y, truncation=self.args.sample_truncation)
        # Post process.
        img = self.post_process(img)

        return img

    @property
    def device(self):
        return next(self.parameters()).device

