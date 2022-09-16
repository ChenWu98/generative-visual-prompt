# Created by Chen Henry Wu
import torch.nn as nn

from FrEIA.framework import SequenceINN
from FrEIA.modules import AllInOneBlock


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_inn(style_dim, n_layer, block, c_dim=None):
    # Define the INN.
    # Affine coupling block
    def subnet_fc(dims_in, dims_out):
        return nn.Sequential(
            nn.Linear(dims_in, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, dims_out),
        )

    if block == 'all_in_one':
        style_component = SequenceINN(style_dim)
        # A simple chain of operations is collected by ReversibleSequential
        for k in range(n_layer):
            if c_dim is not None:
                style_component.append(AllInOneBlock, cond=0, cond_shape=(c_dim, ), subnet_constructor=subnet_fc, permute_soft=True)
            else:
                style_component.append(AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)

    else:
        raise ValueError()

    print("Number of trainable parameters of INN: {}".format(count_parameters(style_component)))
    return style_component
