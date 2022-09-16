# Created by Chen Henry Wu
import torch
import torch.nn as nn

from .gan_wrapper.get_gan_wrapper import get_gan_wrapper
from .mapping import create_inn
from .model_utils import MAX_SAMPLE_SIZE


class PromptGANBase(nn.Module):

    def __init__(self, args):
        super(PromptGANBase, self).__init__()

        self.args = args.model
        self.gan_type = args.gan.gan_type

        # Set up gan_wrapper
        self.gan_wrapper = get_gan_wrapper(args.gan)

        # Set up epsilon to z
        self.eps_to_z = self.create_base_to_latent(self.gan_wrapper.latent_dim)

        # Fixed noise for better visualization.
        self.register_buffer(
            "fixed_eps",
            torch.randn(MAX_SAMPLE_SIZE, self.gan_wrapper.latent_dim),
        )

        # If we are using BigGAN
        if args.gan.gan_type in ['BigGAN']:
            # Set up xi to c
            self.xi_to_y = self.create_base_to_latent(self.gan_wrapper.y_latent_dim)  # Same structure as eps_to_z
            # Fixed noise for better visualization.
            self.register_buffer(
                "fixed_xi",
                torch.randn(MAX_SAMPLE_SIZE, self.gan_wrapper.y_latent_dim) * self.gan_wrapper.y_scale,
            )

    def create_base_to_latent(self, latent_dim):

        if self.args.component == 'inn':
            if self.args.alpha_dim is not None:
                base_to_latent = create_inn(
                    latent_dim,
                    self.args.n_inn_layer,
                    self.args.inn_block,
                    c_dim=self.args.alpha_dim,
                )
            else:
                base_to_latent = create_inn(
                    latent_dim,
                    self.args.n_inn_layer,
                    self.args.inn_block,
                )
        else:
            raise ValueError()

        return base_to_latent

    def get_eps_gaussian(self, sample_id=None, sample_size=None):
        if self.training or sample_id is None:
            bsz = sample_id.shape[0] if sample_id is not None else sample_size
            eps = torch.randn(bsz, self.gan_wrapper.latent_dim, device=self.device)
        else:
            assert sample_id.dim() == 1
            eps = self.fixed_eps[sample_id, :]

        return eps

    def get_xi_gaussian(self, sample_id=None, sample_size=None):
        if self.training or sample_id is None:
            bsz = sample_id.shape[0] if sample_id is not None else sample_size
            xi = torch.randn(bsz, self.gan_wrapper.y_latent_dim, device=self.device) * self.gan_wrapper.y_scale
        else:
            assert sample_id.dim() == 1
            xi = self.fixed_xi[sample_id, :]

        return xi

    def transform_eps_to_z(self, eps, alpha):
        """

        Args:
            eps: shape = (Batch, Dim)

        Returns:

        """

        if self.args.component == 'inn':
            if alpha is not None:
                z, log_det = self.eps_to_z(eps, c=[alpha])
            else:
                z, log_det = self.eps_to_z(eps)
        else:
            raise ValueError()

        return z, log_det

    def transform_xi_to_y(self, xi):
        """

        Args:
            xi: shape = (Batch, Dim)

        Returns:

        """

        if self.args.component == 'inn':
            y, log_det_y = self.xi_to_y(xi)
        else:
            raise ValueError()

        return y, log_det_y

    @property
    def device(self):
        return next(self.parameters()).device


