from typing import Union, Optional, List, Tuple, Text, BinaryIO
import pathlib
import torch
import math
import warnings
import numpy as np
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageColor
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
__all__ = ["make_grid", "save_image", "draw_bounding_boxes"]

def backwarp(batch_size: int, number_previous_frames: int, pre_vis_upsampled: torch.Tensor, mv_upsampled: torch.Tensor, cur_vis_upsampled: torch.Tensor, mask, device, width: int, height: int, scale_factor: int):
    pre_vis_warped = []
    # print('start warp')
    # print('1111:',mv_upsampled.shape)
    mv_upsampled = mv_upsampled[:, :2, :, :]
    # print('2222:', mv_upsampled.shape)
    for id in range(batch_size):
        offset = id * number_previous_frames
        list_previous_viss_warped = []
        for k in range(number_previous_frames):
            if k != number_previous_frames - 1:
                tmp_previous_viss_warped = warp(
                    pre_vis_upsampled[offset + k].unsqueeze(0),
                    mv_upsampled[offset + k].unsqueeze(0),
                    pre_vis_upsampled[offset + k + 1].unsqueeze(0),
                    width,
                    height,
                    scale_factor
                ).to(device) #* mask[offset + k]
            else:
                tmp_previous_viss_warped = warp(
                    pre_vis_upsampled[offset + k].unsqueeze(0),
                    mv_upsampled[offset + k].unsqueeze(0),
                    cur_vis_upsampled[id].unsqueeze(0),
                    width,
                    height,
                    scale_factor
                ).to(device) #* mask[offset + k]
            for i in range(k + 1, number_previous_frames):
                if i != number_previous_frames - 1:
                    tmp_previous_viss_warped = warp(
                        tmp_previous_viss_warped,
                        mv_upsampled[offset + i].unsqueeze(0),
                        pre_vis_upsampled[offset + i + 1].unsqueeze(0),
                        width,
                        height,
                        scale_factor
                    ).to(device) #* mask[offset + i]
                else:
                    tmp_previous_viss_warped = warp(
                        tmp_previous_viss_warped,
                        mv_upsampled[offset + i].unsqueeze(0),
                        cur_vis_upsampled[id].unsqueeze(0),
                        width,
                        height,
                        scale_factor
                    ).to(device) #* mask[offset + i]

            list_previous_viss_warped.append(tmp_previous_viss_warped)

        list_previous_viss_warped = torch.cat(list_previous_viss_warped, dim=0)
        pre_vis_warped.append(list_previous_viss_warped)

    return torch.cat(pre_vis_warped, dim=0)
def warp(pre: torch.Tensor, mv: torch.Tensor, cur: torch.Tensor, width: int, height: int, scale_factor: int):
    b, c, h, w = pre.shape
    device = pre.device

    x = torch.linspace(-1, 1, steps=w)
    y = torch.linspace(-1, 1, steps=h)
    grid_y, grid_x = torch.meshgrid(y, x)
    grid = torch.stack([grid_x, grid_y], dim=0).to(device)

    mx, my = torch.split(mv, 1, dim=1)

    mx_ = mx * 2 * ((width * scale_factor - 1) / (w - 1))
    my_ = my * 2 * ((height * scale_factor - 1) / (h - 1))

    # mx_ = mx * 2 * ((width - 1) / (w - 1))
    # my_ = my * 2 * ((height - 1) / (h - 1))

    mv_ = torch.cat([mx_, my_], dim=1)

    gridMV = (grid - mv_).permute(0, 2, 3, 1)

    # return F.grid_sample(pre, gridMV, align_corners=True)

    warped = F.grid_sample(pre, gridMV, align_corners=True)
    oox, ooy = torch.split((gridMV < -1) | (gridMV > 1), 1, dim=3)
    oo = (oox | ooy).permute(0, 3, 1, 2)
    return torch.where(oo, cur, warped)
@torch.no_grad()
def make_grid(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    value_range: Optional[Tuple[int, int]] = None,
    scale_each: bool = False,
    pad_value: int = 0,
    **kwargs
) -> torch.Tensor:
    """Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by :attr:`range`. Default: ``False``.
        value_range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.

    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_

    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if "range" in kwargs.keys():
        warning = "range will be deprecated, please use value_range instead."
        warnings.warn(warning)
        value_range = kwargs["range"]

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if value_range is not None:
            assert isinstance(value_range, tuple), \
                "value_range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, low, high):
            img.clamp_(min=low, max=high)
            img.sub_(low).div_(max(high - low, 1e-5))

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)

    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            # Tensor.copy_() is a valid method but seems to be missing from the stubs
            # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.copy_
            grid.narrow(1, y * height + padding, height - padding).narrow(  # type: ignore[attr-defined]
                2, x * width + padding, width - padding
            ).copy_(tensor[k])
            k = k + 1
    return grid

def upsample_zero_2d(img: torch.Tensor,
                     size: Union[Tuple[int, int], None] = None,
                     scale_factor: Union[Tuple[int, int], List[int], int, None] = None) \
        -> torch.Tensor:
    """
    IMPORTANT: we only support integer scaling factors for now!!
    """
    # input shape is: batch x channels x height x width
    # output shape is:
    if size is not None and scale_factor is not None:
        raise ValueError("Should either define both size and scale_factor!")
    if size is None and scale_factor is None:
        raise ValueError("Should either define size or scale_factor!")
    input_size = torch.tensor(img.size(), dtype=torch.int)
    input_image_size = input_size[2:]
    data_size = input_size[:2]
    if size is None:
        # Get the last two dimensions -> height x width
        # compare to given scale factor
        b_ = np.asarray(scale_factor)
        b = torch.tensor(b_)
        # check that the dimensions of the tuples match.
        if len(input_image_size) != len(b):
            raise ValueError("scale_factor should match input size!")
        output_image_size = (input_image_size * b).type(torch.int)
    else:
        output_image_size = size
    if scale_factor is None:
        scale_factor = output_image_size / input_image_size
    else:
        scale_factor = torch.tensor(np.asarray(scale_factor), dtype=torch.int)
    ##
    output_size = torch.cat((data_size, output_image_size))
    output = torch.zeros(tuple(output_size.tolist()), device=img.device)
    ##
    # todo: use output.view(...) instead.
    output[:, :, ::scale_factor[0], ::scale_factor[1]] = img
    return output
@torch.no_grad()
def save_image(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    fp: Union[Text, pathlib.Path, BinaryIO],
    format: Optional[str] = None,
    **kwargs
) -> None:
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """

    grid = make_grid(tensor, **kwargs)#c h w

    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer

    if format=='exr':
        ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
        cv2.imwrite(fp,ndarr[:,:,::-1])
    else:
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        im = Image.fromarray(ndarr)
        im.save(fp, format=format)


@torch.no_grad()
def draw_bounding_boxes(
    image: torch.Tensor,
    boxes: torch.Tensor,
    labels: Optional[List[str]] = None,
    colors: Optional[List[Union[str, Tuple[int, int, int]]]] = None,
    fill: Optional[bool] = False,
    width: int = 1,
    font: Optional[str] = None,
    font_size: int = 10
) -> torch.Tensor:

    """
    Draws bounding boxes on given image.
    The values of the input image should be uint8 between 0 and 255.
    If filled, Resulting Tensor should be saved as PNG image.

    Args:
        image (Tensor): Tensor of shape (C x H x W)
        boxes (Tensor): Tensor of size (N, 4) containing bounding boxes in (xmin, ymin, xmax, ymax) format. Note that
            the boxes are absolute coordinates with respect to the image. In other words: `0 <= xmin < xmax < W` and
            `0 <= ymin < ymax < H`.
        labels (List[str]): List containing the labels of bounding boxes.
        colors (List[Union[str, Tuple[int, int, int]]]): List containing the colors of bounding boxes. The colors can
            be represented as `str` or `Tuple[int, int, int]`.
        fill (bool): If `True` fills the bounding box with specified color.
        width (int): Width of bounding box.
        font (str): A filename containing a TrueType font. If the file is not found in this filename, the loader may
            also search in other directories, such as the `fonts/` directory on Windows or `/Library/Fonts/`,
            `/System/Library/Fonts/` and `~/Library/Fonts/` on macOS.
        font_size (int): The requested font size in points.
    """

    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Tensor expected, got {type(image)}")
    elif image.dtype != torch.uint8:
        raise ValueError(f"Tensor uint8 expected, got {image.dtype}")
    elif image.dim() != 3:
        raise ValueError("Pass individual images, not batches")

    ndarr = image.permute(1, 2, 0).numpy()
    img_to_draw = Image.fromarray(ndarr)

    img_boxes = boxes.to(torch.int64).tolist()

    if fill:
        draw = ImageDraw.Draw(img_to_draw, "RGBA")

    else:
        draw = ImageDraw.Draw(img_to_draw)

    txt_font = ImageFont.load_default() if font is None else ImageFont.truetype(font=font, size=font_size)

    for i, bbox in enumerate(img_boxes):
        if colors is None:
            color = None
        else:
            color = colors[i]

        if fill:
            if color is None:
                fill_color = (255, 255, 255, 100)
            elif isinstance(color, str):
                # This will automatically raise Error if rgb cannot be parsed.
                fill_color = ImageColor.getrgb(color) + (100,)
            elif isinstance(color, tuple):
                fill_color = color + (100,)
            draw.rectangle(bbox, width=width, outline=color, fill=fill_color)
        else:
            draw.rectangle(bbox, width=width, outline=color)

        if labels is not None:
            draw.text((bbox[0], bbox[1]), labels[i], fill=color, font=txt_font)

    return torch.from_numpy(np.array(img_to_draw)).permute(2, 0, 1)



def upsample_zero_2d(img: torch.Tensor,
                     size: Union[Tuple[int, int], None] = None,
                     scale_factor: Union[Tuple[int, int], List[int], int, None] = None) \
        -> torch.Tensor:
    """
    IMPORTANT: we only support integer scaling factors for now!!
    """
    # input shape is: batch x channels x height x width
    # output shape is:
    if size is not None and scale_factor is not None:
        raise ValueError("Should either define both size and scale_factor!")
    if size is None and scale_factor is None:
        raise ValueError("Should either define size or scale_factor!")
    input_size = torch.tensor(img.size(), dtype=torch.int)
    input_image_size = input_size[2:]
    data_size = input_size[:2]
    if size is None:
        # Get the last two dimensions -> height x width
        # compare to given scale factor
        b_ = np.asarray(scale_factor)
        b = torch.tensor(b_)
        # check that the dimensions of the tuples match.
        if len(input_image_size) != len(b):
            raise ValueError("scale_factor should match input size!")
        output_image_size = (input_image_size * b).type(torch.int)
    else:
        output_image_size = size
    if scale_factor is None:
        scale_factor = output_image_size / input_image_size
    else:
        scale_factor = torch.tensor(np.asarray(scale_factor), dtype=torch.int)
    ##
    output_size = torch.cat((data_size, output_image_size))
    output = torch.zeros(tuple(output_size.tolist()), device=img.device)
    ##
    # todo: use output.view(...) instead.
    output[:, :, ::scale_factor[0], ::scale_factor[1]] = img
    return output

