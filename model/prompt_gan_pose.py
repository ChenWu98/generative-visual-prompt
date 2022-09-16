# Created by Chen Henry Wu
import torch
import numpy as np
from tqdm import tqdm

from .prompt_gan import PromptGAN
from preprocess.pose import MAX_VALUE_POSE


class PromptGANPose(PromptGAN):

    def __init__(self, args):
        super(PromptGANPose, self).__init__(args)

    def get_paper_image_pose(self, idx):

        sample_id = torch.LongTensor([idx]).to(self.device)
        eps = self.get_eps_gaussian(sample_id=sample_id)

        images = []

        for alpha_x in [-MAX_VALUE_POSE, 0, MAX_VALUE_POSE]:
            for alpha_y in [-MAX_VALUE_POSE, 0, MAX_VALUE_POSE]:
                alpha = torch.FloatTensor([[alpha_x, alpha_y, 0]]).to(self.device)
                z, _ = self.transform_eps_to_z(eps, alpha=alpha)

                img = self.gan_wrapper(z=z)
                images.append(img)

        images = torch.cat(images, dim=0)

        return images

    def forward_clean_fid_pose(self, gaussian):

        sample_size = gaussian.shape[0]
        alpha = torch.FloatTensor(
            [
                [
                    np.random.uniform(-MAX_VALUE_POSE, MAX_VALUE_POSE),
                    np.random.uniform(-MAX_VALUE_POSE, MAX_VALUE_POSE),
                    0,
                ] for _ in range(sample_size)
            ]
        ).to(self.device)

        eps = self.get_eps_gaussian(sample_size=sample_size)
        z, _ = self.transform_eps_to_z(eps, alpha=alpha)

        img = self.gan_wrapper(z=z)

        # Clean KID FID post process (to 255 uint8).
        img = img.mul(255).add(0.5).clamp(0, 255).floor()

        return img

    def forward_embedding_distance(self):

        batch_size = 2
        n_batches = 1024
        all_within_euclidean_distances = []
        all_between_euclidean_distances = []
        for i in tqdm(range(n_batches)):
            esp = self.get_eps_gaussian(sample_size=batch_size)
            assert "IDEnergy" in self.energy_names_paired
            module = self.energy_modules_paired[self.energy_names_paired.index("IDEnergy")]

            # Within.
            alpha_1 = torch.FloatTensor(
                [
                    [
                        np.random.uniform(-MAX_VALUE_POSE, MAX_VALUE_POSE),
                        np.random.uniform(-MAX_VALUE_POSE, MAX_VALUE_POSE),
                        0,
                    ] for _ in range(batch_size)
                ]
            ).to(self.device)
            alpha_2 = torch.FloatTensor(
                [
                    [
                        np.random.uniform(-MAX_VALUE_POSE, MAX_VALUE_POSE),
                        np.random.uniform(-MAX_VALUE_POSE, MAX_VALUE_POSE),
                        0,
                    ] for _ in range(batch_size)
                ]
            ).to(self.device)

            z_1, _ = self.transform_eps_to_z(esp, alpha=alpha_1)
            z_2, _ = self.transform_eps_to_z(esp, alpha=alpha_2)

            img_1 = self.gan_wrapper(z=z_1)
            img_2 = self.gan_wrapper(z=z_2)

            within_id_loss = module(img_1, img_2)  # id_loss = 1 - cosine
            within_euclidean_dist = torch.sqrt(2 * within_id_loss.clamp(min=0))  # euclidean_dist = sqrt(2 - 2 * cosine)
            all_within_euclidean_distances.append(within_euclidean_dist.cpu())

            # Between.
            alpha = torch.FloatTensor(
                [
                    [
                        np.random.uniform(-MAX_VALUE_POSE, MAX_VALUE_POSE),
                        np.random.uniform(-MAX_VALUE_POSE, MAX_VALUE_POSE),
                        0,
                    ] for _ in range(batch_size)
                ]
            ).to(self.device)

            z, _ = self.transform_eps_to_z(esp, alpha=alpha)
            img = self.gan_wrapper(z=z)

            img_shift = torch.cat(
                [img[1:], img[:1]], dim=0
            )
            between_id_loss = module(img, img_shift)  # id_loss = 1 - cosine
            between_euclidean_dist = torch.sqrt(2 * between_id_loss.clamp(min=0))  # euclidean_dist = sqrt(2 - 2 * cosine)
            all_between_euclidean_distances.append(between_euclidean_dist.cpu())

        all_within_euclidean_distances = torch.cat(all_within_euclidean_distances, dim=0)
        all_between_euclidean_distances = torch.cat(all_between_euclidean_distances, dim=0)
        return torch.mean(all_within_euclidean_distances, dim=0), torch.mean(all_between_euclidean_distances, dim=0)


Model = PromptGANPose
