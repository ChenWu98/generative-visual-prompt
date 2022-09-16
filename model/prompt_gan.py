# Created by Chen Henry Wu
import torch
import torch.nn as nn

from .prompt_gan_base import PromptGANBase
from .energy.get_energy import get_energy, parse_key


class PromptGAN(PromptGANBase):

    def __init__(self, args):
        super(PromptGAN, self).__init__(args)

        # Energy
        self.energy_names, self.energy_weights, self.energy_modules = [], [], nn.ModuleList()
        self.energy_names_paired, self.energy_weights_paired, self.energy_modules_paired = [], [], nn.ModuleList()
        for key, value in args:
            key, suffix = parse_key(key)
            if key.endswith('Energy') and ((suffix is None) or isinstance(suffix, int)):
                self.energy_names.append(key)
                self.energy_weights.append(value.weight)
                energy_kwargs = {kw: arg for kw, arg in value if kw != 'weight'}
                self.energy_modules.append(
                    get_energy(name=key, energy_kwargs=energy_kwargs, gan_wrapper=self.gan_wrapper)
                )
            elif key.endswith('Energy') and suffix == 'Pair':
                self.energy_names_paired.append(key)
                self.energy_weights_paired.append(value.weight)
                energy_kwargs = {kw: arg for kw, arg in value if kw != 'weight'}
                self.energy_modules_paired.append(
                    get_energy(name=key, energy_kwargs=energy_kwargs, gan_wrapper=self.gan_wrapper)
                )

    def get_y(self, sample_id=None):
        xi = self.get_xi_gaussian(sample_id=sample_id)

        y, log_det_y = self.transform_xi_to_y(xi)

        return y, log_det_y

    def get_z_clean_fid(self, sample_size, alpha):
        eps = self.get_eps_gaussian(sample_size=sample_size)
        z, log_det = self.transform_eps_to_z(eps, alpha=alpha)

        return z, log_det

    def forward(self, sample_id, alpha=None):

        eps = self.get_eps_gaussian(sample_id=sample_id)
        z, log_det = self.transform_eps_to_z(eps, alpha=alpha)

        if self.gan_type in ['BigGAN']:
            y, log_det_y = self.get_y(sample_id=sample_id)
            img = self.gan_wrapper(z=z, y=y)
        else:
            y, log_det_y = None, torch.zeros(sample_id.shape[0], device=self.device)
            img = self.gan_wrapper(z=z)

        losses = dict()
        weighted_loss = 0
        # Energy.
        for name, weight, module in zip(self.energy_names,
                                        self.energy_weights,
                                        self.energy_modules):
            inputs = module.prepare_inputs(img, z, y, alpha)
            loss = module(**inputs)
            losses[name] = loss
            weighted_loss += weight * loss
        # Energy paired.
        for name, weight, module in zip(self.energy_names_paired,
                                        self.energy_weights_paired,
                                        self.energy_modules_paired):
            z_0, _ = self.transform_eps_to_z(eps, alpha=torch.zeros_like(alpha))
            img_0 = self.gan_wrapper(z=z_0)
            inputs = module.prepare_inputs(img, img_0, alpha)
            loss = module(**inputs)
            losses[f'{name}_paired'] = loss
            weighted_loss += weight * loss
        # Jac.
        losses['-log_det'] = -log_det
        weighted_loss -= log_det
        losses['-log_det_y'] = -log_det_y
        weighted_loss -= log_det_y

        return img, weighted_loss, losses

    def forward_eps(self, eps):

        z, _ = self.transform_eps_to_z(eps, alpha=None)

        img = self.gan_wrapper(z=z)

        return img

    def get_eps_z_img(self, sample_size):

        if self.gan_type in ['BigGAN']:
            raise ValueError()
        else:
            eps = self.get_eps_gaussian(sample_size=sample_size)
            z, _ = self.transform_eps_to_z(eps, alpha=None)

            img = self.gan_wrapper(z=z)
            return eps, z, img

    def forward_clean_fid(self, eps):

        if self.args.alpha_dim is not None:
            raise NotImplementedError()  # Implemented in prompt_gan_expression.py and prompt_gan_pose.py
        else:
            z, _ = self.get_z_clean_fid(eps.shape[0], alpha=None)  # (B, L, style_dim) or (B, style_dim)

        img = self.gan_wrapper(z=z)

        # Clean KID FID post process (to 255 uint8).
        img = img.mul(255).add(0.5).clamp(0, 255).floor()

        return img

    def forward_with_size(self, batch_size):

        z, _ = self.get_z_clean_fid(batch_size, alpha=None)  # (B, L, style_dim) or (B, style_dim)

        img = self.gan_wrapper(z=z)

        return img


Model = PromptGAN
