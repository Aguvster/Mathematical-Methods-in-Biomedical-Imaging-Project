import argparse
import numpy as np
import math
import torch
import mrcfile
import time
from PIL import Image
from torch import vmap
from torch.optim.lr_scheduler import ExponentialLR
from torchvision.utils import save_image
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_WIDTH = 256
IMG_HEIGHT = 256

ALPHA = 2.0
BETA = 5.0

TRANSLATION_SIGMA = 0.1  # 5% of image with/height, so in relative terms, 0.05 * 2 = 0.1

MAX_RUN_TIME = 10 * 60
EPS_DECAY_RATE = 1.5

DEFAULT_ORIGINAL_IMAGE = 'EricCartmanGray.png'
DEFAULT_INPUT = 'EricCartman_sigma_0.1.mrc'



def plot_results(initial, result, orig, data, sigma, dataset_size, base_learning_rates, batch_size, grid_size):
    """
    Plot the original image, noisy transformed image, initial image, and final image.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 5))
    plt.suptitle(
        f'Noise Sigma = {sigma}, Data Size = {dataset_size}, Theta Base LR = {base_learning_rates[0]}, Sigma Base LR = {base_learning_rates[1]}, Batch Size = {batch_size}, Grid Size = {grid_size}')
    axes[0][0].imshow(orig, cmap='gray')
    axes[0][0].axis('off')
    axes[0][0].set_title('Original Image')
    axes[0][1].imshow(data.squeeze(), cmap='gray')
    axes[0][1].axis('off')
    axes[0][1].set_title(f'Noisy Transformed Image')
    axes[1][0].imshow(initial.squeeze().cpu().numpy(), cmap='gray')
    axes[1][0].axis('off')
    axes[1][0].set_title('Initial Image')
    axes[1][1].imshow(result.squeeze().cpu().numpy(), cmap='gray')
    axes[1][1].axis('off')
    axes[1][1].set_title('Final Image')
    plt.tight_layout()
    plt.show()


def transform_image(img, angle_deg, translation, intensity_scale):
    """
    Apply a rotation, translation, and intensity scaling to an image.
    :param img: Image tensor, hoping to be (N, C, H, W)
    :param angle_deg: Rotation in degrees
    :param translation: Translation in relative pixels (-1 to 1)
    :param intensity_scale: Intensity scaling factor
    :return:
    """
    if img.dim() == 3:
        img = img.unsqueeze(0)
    elif img.dim() == 2:
        img = img.unsqueeze(0).unsqueeze(0)

    theta_rad = angle_deg * torch.pi / 180.0

    cos_val = torch.cos(theta_rad)
    sin_val = torch.sin(theta_rad)

    # Construct the 2x3 affine transformation matrix for the rotation and translation
    affine_matrix = torch.stack([
        torch.stack([cos_val, -sin_val, translation[0]]),
        torch.stack([sin_val, cos_val, translation[1]])
    ])

    # Add the batch dimension to get shape (N, 2, 3).
    affine_matrix = affine_matrix.unsqueeze(0)

    # Create the sampling grid using the affine matrix.
    grid = F.affine_grid(affine_matrix, img.size(), align_corners=False)
    # Apply the transformation.
    transformed_img = F.grid_sample(img, grid, align_corners=False, padding_mode="zeros")

    # Apply intensity scaling.
    transformed_img = transformed_img * intensity_scale

    return transformed_img


def prior_rotation(x):
    """
    Prior distribution for the rotation angle.
    :param x: Rotation angle in degrees (but it really doesn't matter)
    :return: Prior probability density for the rotation angle
    """
    return torch.tensor(1 / (2 * math.pi), dtype=x.dtype)


def prior_translation(x, y):
    """
    Prior distribution for the translation parameters.
    :param x: Translation in x-direction (relative)
    :param y: Translation in y-direction (relative)
    :return: Prior probability density for the translation parameters
    """
    var = TRANSLATION_SIGMA ** 2
    factor = torch.tensor(2 * np.pi * var, device=device)
    return (1 / factor) * torch.exp(-0.5 * (x ** 2 + y ** 2) / var)


def prior_scale(x):
    """
    Prior distribution for the intensity scaling factor.
    :param x: Intensity scaling factor
    :return: Prior probability density for the intensity scaling factor
    """
    alpha_tensor = torch.tensor(ALPHA, dtype=x.dtype)
    beta_tensor = torch.tensor(BETA, dtype=x.dtype)

    y = x / 5
    norm = torch.exp(torch.lgamma(alpha_tensor)) * torch.exp(torch.lgamma(beta_tensor)) / torch.exp(
        torch.lgamma(alpha_tensor + beta_tensor))

    return (y ** (ALPHA - 1) * (1 - y) ** (BETA - 1)) / norm


def calc_dist(X, theta, angle, scale, translation_x, translation_y):
    """
    Calculate the distance (squared) between the transformed image and the original image.
    :param X: Dataset image tensor
    :param theta: Current image tensor
    :param angle: Rotation angle in degrees
    :param scale: Intensity scaling factor
    :param translation_x: Translation in x-direction (relative)
    :param translation_y: Translation in y-direction (relative)
    :return: Squared distance between the images
    """
    transformed = transform_image(theta, angle, [translation_x, translation_y], scale)
    return torch.norm((X - transformed)) ** 2


def calc_prob(X, theta, sigma, angle, scale, translation_x, translation_y):
    """
    Calculate the log-probability of the transformation parameters given the data.
    :param X: Dataset image tensor
    :param theta: Current image tensor
    :param sigma: Current standard deviation of the Gaussian noise
    :param angle: Rotation angle in degrees
    :param scale: Intensity scaling factor
    :param translation_x: Translation in x-direction (relative)
    :param translation_y: Translation in y-direction (relative)
    :return: Log-probability of the transformation parameters
    """
    const = torch.tensor(2 * np.pi, device=device)
    return (- IMG_WIDTH * IMG_HEIGHT * torch.log(sigma * torch.sqrt(const))
            - calc_dist(X, theta, angle, scale, translation_x, translation_y) / (2 * sigma ** 2)
            + torch.log(prior_scale(scale))
            + torch.log(prior_rotation(angle))
            + torch.log(prior_translation(translation_x, translation_y)))


def calc_loss(theta, data, sigma,
              grid_sizes=(70, 5, 7, 7),
              grid_ranges=((0, 360), (0, 5), (-0.2, 0.2), (-0.2, 0.2)),
              approx_transforms=None,
              eps=None):
    """
    Calculate the loss function for the given data and parameters.
    :param theta: Current image tensor
    :param data: Dataset tensor
    :param sigma: Current standard deviation of the Gaussian noise
    :param grid_sizes: Number of grid points for each parameter
    :param grid_ranges: Ranges for each parameter
    :param approx_transforms: Approximate optimal transformations for each data sample
    :param eps: Search radius for each parameter
    :return: Average loss over the dataset and updated approx_transforms
    """
    data_tensor = torch.stack([torch.from_numpy(x).unsqueeze(0).to(device) for x in data])

    sample_losses = []
    updated_approx_transforms = []

    # Loop over each sample
    for i, sample in enumerate(data_tensor):
        if approx_transforms is not None and eps is not None:
            approx_t = approx_transforms[i]
            # For each parameter, subtract and add eps then clamp to the global grid range.
            angle_lower = max(approx_t[0] - eps[0], grid_ranges[0][0])
            angle_upper = min(approx_t[0] + eps[0], grid_ranges[0][1])
            angles = torch.linspace(angle_lower, angle_upper, grid_sizes[0], device=device)

            scale_lower = max(approx_t[1] - eps[1], grid_ranges[1][0])
            scale_upper = min(approx_t[1] + eps[1], grid_ranges[1][1])
            scales = torch.linspace(scale_lower, scale_upper, grid_sizes[1], device=device)

            trans_x_lower = max(approx_t[2] - eps[2], grid_ranges[2][0])
            trans_x_upper = min(approx_t[2] + eps[2], grid_ranges[2][1])
            translations_x = torch.linspace(trans_x_lower, trans_x_upper, grid_sizes[2], device=device)

            trans_y_lower = max(approx_t[3] - eps[3], grid_ranges[3][0])
            trans_y_upper = min(approx_t[3] + eps[3], grid_ranges[3][1])
            translations_y = torch.linspace(trans_y_lower, trans_y_upper, grid_sizes[3], device=device)
        else:
            # Use the fixed global grid if no approx_transforms and eps are provided.
            angles = torch.linspace(*grid_ranges[0], grid_sizes[0], device=device)
            scales = torch.linspace(*grid_ranges[1], grid_sizes[1], device=device)
            translations_x = torch.linspace(*grid_ranges[2], grid_sizes[2], device=device)
            translations_y = torch.linspace(*grid_ranges[3], grid_sizes[3], device=device)

        grid = torch.cartesian_prod(angles, scales, translations_x, translations_y)
        grid_size = grid.size(0)

        # Compute probabilities for every transformation in the grid
        probs = vmap(lambda args: calc_prob(sample, theta, sigma, *args))(grid)

        # Compute the sample loss
        loss_val = -torch.logsumexp(probs, dim=0) + torch.log(torch.tensor(grid_size, dtype=probs.dtype, device=device))
        sample_losses.append(loss_val)

        # Update the approx transform for the sample
        best_idx = torch.argmax(probs)
        best_transform = grid[best_idx]
        updated_approx_transforms.append(best_transform)

    # Average the loss over all samples
    avg_loss = torch.stack(sample_losses).mean()
    return avg_loss, updated_approx_transforms


def cyclic_batch_generator(lst, batch_size):
    """
    Generate batches from a list cyclically.
    :param lst: List to generate batches from
    :param batch_size: Size of each batch
    :return: Batch and batch index inside the list
    :note: The last batch may be smaller than the batch size.
    """
    n = len(lst)
    index = 0
    while True:
        batch = [lst[(index + i) % n] for i in range(batch_size)]
        yield batch, index
        index = (index + batch_size) % n


def calc_empirical_sigma(data):
    """
    Calculate the empirical standard deviation of the data
    :param data: Data tensor
    :return: Empirical standard deviation
    """
    return torch.std(torch.from_numpy(data).float().to(device))


def parse_args():
    """
    Parse command line arguments.
    :return: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Image Transformation and Training Script with CLI options."
    )
    parser.add_argument(
        "--input_mrc",
        type=str,
        default=DEFAULT_INPUT,
        help="Path to the input MRC file containing the image stack. Format: '{name}_sigma_{sigma_value_float}.mrc'."
    )
    parser.add_argument(
        "--orig_image",
        type=str,
        default=DEFAULT_ORIGINAL_IMAGE,
        help="Path to the original image file for reference."
    )
    parser.add_argument(
        "--lr_theta",
        type=float,
        default=1e-3,
        help="Learning rate for theta (default: 1e-3)."
    )
    parser.add_argument(
        "--lr_sigma",
        type=float,
        default=1e-6,
        help="Learning rate for sigma (default: 1e-6)."
    )
    parser.add_argument(
        "--convergence_threshold",
        type=float,
        default=1e-1,
        help="Convergence threshold for the epoch loss (default: 1e-1).")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for training (default: 4)."
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        nargs=4,
        default=[20, 4, 5, 5],
        help="Number of grid points for each parameter [rotations, scalings, x-axis translations, y-axis translations] (default: 20 4 5 5)."
    )
    parser.add_argument(
        "--dataset_size",
        type=int,
        default=100,
        help="Number of samples to use from the dataset (default: 100). Should be a multiple of the batch size."
    )
    parser.add_argument(
        "--max_run_time",
        type=int,
        default=MAX_RUN_TIME,
        help="Maximum run time in seconds (default: 600 seconds, i.e. 10 minutes)."
    )
    parser.add_argument("--radius_decay_ratio", type=float, default=2.0,
                        help="Decay ratio for the radius (default: 2.0).")
    return parser.parse_args()


def main():
    args = parse_args()
    start = time.time()

    with mrcfile.open(args.input_mrc, permissive=True) as mrc:
        data = mrc.data.copy()[:args.dataset_size]

    # Initialize the parameters
    theta = torch.zeros(1, int(IMG_HEIGHT), int(IMG_WIDTH), requires_grad=True, device=device)
    initial = theta.clone()
    sigma = calc_empirical_sigma(data)
    sigma = torch.tensor(torch.log(sigma).item(), requires_grad=True, device=device)

    # Initialize the optimizer and scheduler
    optimizer = torch.optim.SGD([
        {'params': [theta], 'lr': args.lr_theta},
        {'params': [sigma], 'lr': args.lr_sigma}
    ])

    scheduler = ExponentialLR(optimizer, gamma=0.9)

    # Normalize theta to [0, 1]
    with torch.no_grad():
        theta_min = torch.min(theta)
        theta_max = torch.max(theta)
        if theta_max > theta_min:
            theta.data = (theta.data - theta_min) / (theta_max - theta_min)

    # Initialize the approximate transformations and search radii
    approx_transforms = [[180, 1, 0, 0]] * args.dataset_size
    eps = [180, 4, 0.2, 0.2]

    grid_size = args.grid_size
    batch_size = args.batch_size

    # Training loop
    epoch_losses = []
    last_epoch_loss = float('inf')
    epoch_loss = float('inf')
    epoch_start_time = 0
    expected_epochs = 1000
    epoch = 0
    while epoch < expected_epochs:
        # Check for convergence
        if last_epoch_loss != float('inf') and abs(epoch_loss - last_epoch_loss) / abs(
                last_epoch_loss) < args.convergence_threshold:
            break

        if epoch == 0:
            epoch_start_time = time.time()

        last_epoch_loss = epoch_loss
        epoch_loss = 0

        gen = cyclic_batch_generator(data, batch_size)
        steps = int(args.dataset_size / batch_size)
        # Training loop for each epoch
        for step in range(steps):
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            # Get the next batch
            batch, index = next(gen)

            # Calculate the loss and update the approx_transforms
            loss, new_approx_transforms = calc_loss(theta, batch, torch.exp(sigma), grid_sizes=grid_size,
                                                    approx_transforms=approx_transforms[index: index + batch_size],
                                                    eps=eps)
            approx_transforms[index: index + batch_size] = new_approx_transforms
            epoch_loss += loss.item() / steps

            # Update the parameters
            loss.backward()
            optimizer.step()

            # Normalize theta to [0, 1]
            with torch.no_grad():
                theta_min = torch.min(theta)
                theta_max = torch.max(theta)
                if theta_max > theta_min:
                    theta.data = (theta.data - theta_min) / (theta_max - theta_min)

            theta_image = theta.unsqueeze(0)
            save_image(theta_image, 'theta.png')

        epoch_losses.append(epoch_loss)
        scheduler.step()
        # Decay the search radius
        eps = [eps[0] / args.radius_decay_ratio, eps[1] / args.radius_decay_ratio, eps[2] / args.radius_decay_ratio,
               eps[3] / args.radius_decay_ratio]

        if epoch == 0:
            # Estimate the number of epochs based on the first epoch run time
            epoch_run_time = time.time() - epoch_start_time
            expected_epochs = int(args.max_run_time / epoch_run_time)

        epoch += 1

    print("Time taken:", time.time() - start)
    orig_image = Image.open(args.orig_image)
    orig_sigma = float('.'.join(args.input_mrc.split('_')[-1].split('.')[0:2]))
    plot_results(initial.detach(), theta.detach(), orig_image, data[0], orig_sigma, args.dataset_size,
                 [args.lr_theta, args.lr_sigma], batch_size, grid_size)


if __name__ == '__main__':
    main()
