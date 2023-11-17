import numpy as np
from skimage.util import img_as_float


def log_images(
    input,
    target,
    prediction,
    iteration_index,
    logger,
    final_activation_function=None,
    prefix="",
    mask=None,
):

    if final_activation_function is not None:
        prediction = final_activation_function(prediction)

    if mask is not None:
        mask = mask.cpu().numpy()

    inputs_map = {
        "inputs": input,
        "targets": target,
        "predictions": prediction,
    }
    img_sources = {}
    for name, batch in inputs_map.items():
        img_sources[name] = batch.data.cpu().numpy()

    for name, batch in img_sources.items():
        for tag, image in create_tensorboard_tagged_image(name, batch, mask):
            logger.add_image(
                prefix + tag, np.expand_dims(image, axis=0), iteration_index
            )


def create_tensorboard_tagged_image(data_name, batch, mask=None):

    tag_template = "{}/batch_{}/channel_{}/slice_{}"

    tagged_images = []

    if batch.ndim == 5:
        # NCDHW
        slice_idx = batch.shape[2] // 2  # get the middle slice
        for batch_idx in range(batch.shape[0]):
            for channel_idx in range(batch.shape[1]):
                tag = tag_template.format(
                    data_name, batch_idx, channel_idx, slice_idx
                )
                img = img_as_float(
                    batch[batch_idx, channel_idx, slice_idx, ...]
                )
                if mask is not None:
                    mask_slice = mask[batch_idx, 0, slice_idx, ...]
                    img[np.logical_not(mask_slice)] = 0

                tagged_images.append((tag, normalize_image(img)))
    else:
        # batch has no channel dim: NDHW
        slice_idx = batch.shape[1] // 2  # get the middle slice
        for batch_idx in range(batch.shape[0]):
            tag = tag_template.format(data_name, batch_idx, 0, slice_idx)
            img = batch[batch_idx, slice_idx, ...]
            tagged_images.append((tag, normalize_image(img)))

    return tagged_images


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Rescale the image to go from 0-1.

    Parameters
    ----------
    image : np.ndarray
        The image to be normalized.

    Returns
    -------
    normalized_image : np.ndarray
        The input image rescaled to go from 0 to 1.
    """
    image_range = np.ptp(image)
    if image_range == 0:
        return np.ones_like(image)
    else:
        return np.nan_to_num((image - np.min(image)) / image_range)
