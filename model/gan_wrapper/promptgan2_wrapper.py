# Created by Chen Henry Wu
import os
import torch
from collections import OrderedDict

from ..model_utils import requires_grad
from ..prompt_gan import PromptGAN
from utils.config_utils import get_config


class PromptGAN2Wrapper(torch.nn.Module):

    def __init__(self, args):
        super(PromptGAN2Wrapper, self).__init__()

        prompt_args = get_config(args.arg_path)
        # Set up generator
        self.generator = PromptGAN(args=prompt_args)
        self.latent_dim = self.generator.gan_wrapper.latent_dim
        module_state_dict = torch.load(os.path.join(args.ckpt, 'pytorch_model.bin'))
        assert all([k.startswith('module.') for k in module_state_dict])
        state_dict = OrderedDict(
            [
                (k[len('module.'):], v)
                for k, v in module_state_dict.items()
            ]
        )
        self.generator.load_state_dict(state_dict=state_dict, strict=True)
        # Freeze.
        requires_grad(self.generator, False)

    def forward(self, z):
        # Eval mode for the generator.
        self.generator.eval()

        img = self.generator.forward_eps(z)

        return img

    @property
    def device(self):
        return next(self.parameters()).device




