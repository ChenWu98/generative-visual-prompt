# Created by Chen Henry Wu
from .stylegan2_wrapper import StyleGAN2Wrapper
from .biggan_wrapper import BigGANWrapper


def get_gan_wrapper(args):

    if args.gan_type == "StyleGAN2":
        return StyleGAN2Wrapper(args)
    elif args.gan_type == "BigGAN":
        return BigGANWrapper(args)
    elif args.gan_type == "StyleNeRF":
        from .stylenerf_wrapper import StyleNeRFWrapper
        return StyleNeRFWrapper(args)
    elif args.gan_type == "DiffAE":
        from .diffae_wrapper import DiffAEWrapper
        return DiffAEWrapper(args)
    elif args.gan_type == "PromptGAN2":
        from .promptgan2_wrapper import PromptGAN2Wrapper
        return PromptGAN2Wrapper(args)
    else:
        raise ValueError()

