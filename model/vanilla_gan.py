# Created by Chen Henry Wu
import torch
import torch.nn as nn

from .model_utils import requires_grad, MAX_SAMPLE_SIZE
from .gan_wrapper.get_gan_wrapper import get_gan_wrapper


class VanillaGAN(nn.Module):

    def __init__(self, args):
        super(VanillaGAN, self).__init__()

        # Set up gan_wrapper
        self.gan_wrapper = get_gan_wrapper(args.gan)
        # Freeze.
        requires_grad(self.gan_wrapper, True)  # Otherwise, no trainable params.

        # Fixed noise for better visualization.
        self.register_buffer(
            "fixed_z",
            torch.randn(MAX_SAMPLE_SIZE, self.gan_wrapper.latent_dim),
        )

    def get_z_gaussian(self, sample_id=None):
        if self.training:
            bsz = sample_id.shape[0]
            z = torch.randn(bsz, self.gan_wrapper.latent_dim, device=self.device)
        else:
            assert sample_id.dim() == 1
            z = self.fixed_z[sample_id, :]

        return z

    def forward(self, sample_id):
        # Eval mode for the gan_wrapper.
        self.gan_wrapper.eval()

        assert not self.training

        z = self.get_z_gaussian(sample_id=sample_id)  # (B, style_dim)

        img = self.gan_wrapper(z=z)

        # Placeholders
        losses = dict()
        weighted_loss = torch.zeros_like(sample_id).float()

        return img, weighted_loss, losses


Model = VanillaGAN
