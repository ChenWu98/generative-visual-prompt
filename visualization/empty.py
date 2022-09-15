# Created by Chen Henry Wu
import os
import math

from utils.file_utils import save_images
import torch.nn.functional as F


class Visualizer(object):

    def __init__(self, args):
        self.args = args

    def visualize(self,
                  images,
                  model,
                  description: str,
                  save_dir: str,
                  step: int,
                  ):
        pass
