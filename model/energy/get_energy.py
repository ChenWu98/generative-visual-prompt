# Created by Chen Henry Wu
from .clip_guide import CLIPEnergy
from .prior_z import PriorZEnergy
from .prior_y import PriorYEnergy
from .pose import PoseEnergy
from .id import IDEnergy
from .debias import DebiasEnergy
from .class_condition import ClassEnergy


def get_energy(name, energy_kwargs, gan_wrapper):

    if name == "CLIPEnergy":
        return CLIPEnergy(**energy_kwargs)
    elif name == "DebiasEnergy":
        return DebiasEnergy(**energy_kwargs)
    elif name == "PriorZEnergy":
        return PriorZEnergy()
    elif name == "PriorYEnergy":
        return PriorYEnergy(y_scale=gan_wrapper.y_scale)
    elif name == "PoseEnergy":
        return PoseEnergy(**energy_kwargs)
    elif name == "IDEnergy":
        return IDEnergy(**energy_kwargs)
    elif name == "ClassEnergy":
        return ClassEnergy(**energy_kwargs)
    else:
        raise ValueError()


def parse_key(key):
    if key.endswith('1'):
        return key[:-1], 1
    elif key.endswith('2'):
        return key[:-1], 2
    elif key.endswith('Pair'):
        return key[:-len('Pair')], 'Pair'
    else:
        return key, None
