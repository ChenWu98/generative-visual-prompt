# Created by Chen Henry Wu
import torch
from torchvision import utils
from cleanfid import fid

RESOLUTION = 1024
DATASET_NAME = "FFHQ"
DATASET_SPLIT = "trainval70k"
BATCH_SIZE = 16


def save_image(image_path, image):
    assert image.shape == (3, RESOLUTION, RESOLUTION)
    utils.save_image(image, image_path)


class Evaluator(object):

    def __init__(self, args, meta_args):
        self.args = args
        self.meta_args = meta_args

    def fid(self, model):
        fid_score = fid.compute_fid(
            gen=lambda gaussian: model.forward_clean_fid_pose(gaussian=gaussian),
            dataset_name=DATASET_NAME,
            dataset_res=RESOLUTION,
            dataset_split=DATASET_SPLIT,
            batch_size=BATCH_SIZE,
        )

        return fid_score

    def evaluate(self, images, model, weighted_loss, losses, data, split):
        """

        Args:
            images: list of images, or list of tuples of images
            model: model to evaluate
            weighted_loss: list of scalar tensors
            losses: dictionary of lists of scalar tensors
            data: list of dictionary
            split: str

        Returns:

        """
        assert split in ['eval', 'test']
        fid = self.fid(model)
        print("FID: {:.4f}".format(fid))

        with torch.no_grad():
            within_distance, between_distance = model.forward_embedding_distance()

        # Add metrics here.
        summary = {
            "within_distance": within_distance.item(),
            "between_distance": between_distance.item(),
            "fid": fid,
        }

        return summary
