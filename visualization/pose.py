# Created by Chen Henry Wu
import os
import math
import torch
import torch.nn.functional as F

from utils.file_utils import save_images


class Visualizer(object):

    def __init__(self, args):
        self.args = args

    def get_paper_images(self, model):
        images = []
        for i in range(7):
            images.append(model.get_paper_image_pose(i))
        images = torch.cat(images, dim=0)
        return images

    def visualize(self,
                  images,
                  model,
                  description: str,
                  save_dir: str,
                  step: int,
                  ):

        # Visualization in the paper.
        with torch.no_grad():
            paper_images = self.get_paper_images(model)
            b, c, h, w = paper_images.shape
            assert b == 9 * 7
            paper_images = paper_images.\
                reshape(7, 3, 3, c, h, w).\
                transpose(0, 1).\
                reshape(3, 1, 21, c, h, w).\
                transpose(0, 1).\
                reshape(63, c, h, w)
        save_images(
            paper_images,
            output_dir=save_dir,
            file_prefix=f'{description}_paper',
            nrows=21,
            iteration=step,
        )

        # Lower resolution
        paper_images_256 = F.interpolate(
            paper_images,
            (256, 256),
            mode='bicubic',
        )
        save_images(
            paper_images_256,
            output_dir=save_dir,
            file_prefix=f'{description}_256_paper',
            nrows=21,
            iteration=step,
        )
