import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import mrcfile

IMAGES = ['EricCartman.png', 'mhw_logo.webp']
SIGMAS = [0.0, 0.1, 0.2, 0.5, 1.0]
NUM_IMAGES = 100
IMAGE_SIZE = (256, 256)


def pre_process_image(image_path):
    image = Image.open(image_path).convert('L')
    image = image.resize(IMAGE_SIZE)
    name, ext = image_path.split('.')
    image.save(f'{name}Gray.{ext}')
    return TF.to_tensor(image)


def apply_transform(template, noise_sigma):
    angle_rad = np.random.uniform(0, 2 * np.pi)
    angle_deg = np.degrees(angle_rad)

    # Sample translations (in pixels) from N(0, (0.05*size)^2)
    tx = np.random.normal(0, 0.05 * IMAGE_SIZE[0])
    ty = np.random.normal(0, 0.05 * IMAGE_SIZE[1])
    translation = [tx, ty]

    # Sample intensity scale from 5 * Beta(2, 5)
    intensity_scale = 5 * np.random.beta(2, 5)

    # Apply the affine transformation using the sampled parameters.
    transformed_tensor = TF.affine(template, angle=angle_deg, translate=translation, scale=1, shear=0, fill=0)

    # Apply the intensity scaling to the transformed image.
    scaled_tensor = transformed_tensor * intensity_scale

    # Add Gaussian noise: sigma * standard_normal noise
    noise = torch.randn_like(transformed_tensor) * noise_sigma
    noisy_tensor = scaled_tensor + noise

    return noisy_tensor


def main():
    tensors = [pre_process_image(image_path) for image_path in IMAGES]

    transformed_images = {}
    for noise_sigma in SIGMAS:
        transformed_images[noise_sigma] = {}
        for img_idx in range(len(tensors)):
            transformed_images[noise_sigma][img_idx] = []
            for _ in range(NUM_IMAGES):
                transformed_images[noise_sigma][img_idx].append(apply_transform(tensors[img_idx], noise_sigma))

    for noise_sigma in SIGMAS:
        for img_idx in range(len(tensors)):
            images_stack = np.stack(transformed_images[noise_sigma][img_idx])
            filename = f"{IMAGES[img_idx].split('.')[0]}_sigma_{noise_sigma}.mrc"

            # Write the image stack to an MRC file
            with mrcfile.new(filename, overwrite=True) as mrc:
                mrc.set_data(images_stack.astype(np.float32))

            print(f"Saved {filename}")


if __name__ == '__main__':
    main()
