# Standard Library
import re

# Third Party
import numpy as np

# First Party
from smdebug.core.utils import make_numpy_array

# Local
from .proto.summary_pb2 import HistogramProto, Summary
from .util import convert_to_HWC

_INVALID_TAG_CHARACTERS = re.compile(r"[^-/\w\.]")


def _clean_tag(name):
    """Cleans a tag. Removes illegal characters for instance.
    Adapted from the TensorFlow function `clean_tag()` at
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/summary_op_util.py
    Parameters
    ----------
        name : str
            The original tag name to be processed.
    Returns
    -------
        The cleaned tag name.
    """
    # In the past, the first argument to summary ops was a tag, which allowed
    # arbitrary characters. Now we are changing the first argument to be the node
    # name. This has a number of advantages (users of summary ops now can
    # take advantage of the tf name scope system) but risks breaking existing
    # usage, because a much smaller set of characters are allowed in node names.
    # This function replaces all illegal characters with _s, and logs a warning.
    # It also strips leading slashes from the name.
    if name is not None:
        new_name = _INVALID_TAG_CHARACTERS.sub("_", name)
        new_name = new_name.lstrip("/")  # Remove leading slashes
        if new_name != name:
            name = new_name
    return name


def scalar_summary(tag, scalar):
    """Outputs a `Summary` protocol buffer containing a single scalar value.
    The generated Summary has a Tensor.proto containing the input Tensor.
    Adapted from the TensorFlow function `scalar()` at
    https://github.com/tensorflow/tensorflow/blob/r1.6/tensorflow/python/summary/summary.py
    Parameters
    ----------
      tag : str
          A name for the generated summary. Will also serve as the series name in TensorBoard.
      scalar : int, MXNet `NDArray`, or `numpy.ndarray`
          A scalar value or an ndarray of shape (1,).
    Returns
    -------
      A `Summary` protobuf of the `scalar` value.
    Raises
    ------
      ValueError: If the scalar has the wrong shape or type.
    """
    tag = _clean_tag(tag)
    scalar = make_numpy_array(scalar)
    assert scalar.squeeze().ndim == 0, "scalar should be 0D"
    scalar = float(scalar)
    return Summary(value=[Summary.Value(tag=tag, simple_value=scalar)])


def _make_histogram(values, bins, max_bins):
    """Converts values into a histogram proto using logic from
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/lib/histogram/histogram.cc"""
    if values.size == 0:
        raise ValueError("The input has no element.")
    values = values.reshape(-1)
    counts, limits = np.histogram(values, bins=bins)
    num_bins = len(counts)
    if max_bins is not None and num_bins > max_bins:
        subsampling = num_bins // max_bins
        subsampling_remainder = num_bins % subsampling
        if subsampling_remainder != 0:
            counts = np.pad(
                counts,
                pad_width=[[0, subsampling - subsampling_remainder]],
                mode="constant",
                constant_values=0,
            )
        counts = counts.reshape(-1, subsampling).sum(axis=-1)
        new_limits = np.empty((counts.size + 1,), limits.dtype)
        new_limits[:-1] = limits[:-1:subsampling]
        new_limits[-1] = limits[-1]
        limits = new_limits

    # Find the first and the last bin defining the support of the histogram:
    cum_counts = np.cumsum(np.greater(counts, 0, dtype=np.int32))
    start, end = np.searchsorted(cum_counts, [0, cum_counts[-1] - 1], side="right")
    start = int(start)
    end = int(end) + 1
    del cum_counts

    # TensorBoard only includes the right bin limits. To still have the leftmost limit
    # included, we include an empty bin left.
    # If start == 0, we need to add an empty one left, otherwise we can just include the bin left to the
    # first nonzero-count bin:
    counts = counts[start - 1 : end] if start > 0 else np.concatenate([[0], counts[:end]])
    limits = limits[start : end + 1]

    if counts.size == 0 or limits.size == 0:
        raise ValueError("The histogram is empty, please file a bug report.")

    sum_sq = values.dot(values)
    return HistogramProto(
        min=values.min(),
        max=values.max(),
        num=len(values),
        sum=values.sum(),
        sum_squares=sum_sq,
        bucket_limit=limits.tolist(),
        bucket=counts.tolist(),
    )


def _get_default_bins():
    """Ported from the C++ function InitDefaultBucketsInner() in the following file.
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/lib/histogram/histogram.cc
    See the following tutorial for more details on how TensorFlow initialize bin distribution.
    https://www.tensorflow.org/programmers_guide/tensorboard_histograms"""
    v = 1e-12
    buckets = []
    neg_buckets = []
    while v < 1e20:
        buckets.append(v)
        neg_buckets.append(-v)
        v *= 1.1
    return neg_buckets[::-1] + [0] + buckets


def histogram_summary(tag, values, bins, max_bins=None):
    """Outputs a `Summary` protocol buffer with a histogram.
    Adding a histogram summary makes it possible to visualize the data's distribution in
    TensorBoard. See detailed explanation of the TensorBoard histogram dashboard at
    https://www.tensorflow.org/get_started/tensorboard_histograms
    This op reports an `InvalidArgument` error if any value is not finite.
    Adapted from the TensorFlow function `histogram()` at
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/summary/summary.py
    and Pytorch's function
    https://github.com/pytorch/pytorch/blob/2655b2710c8f0b3253fe2cfe0d2674f5d3592d22/torch/utils/tensorboard/summary.py#L232
    Parameters
    ----------
        tag : str
            A name for the summary of the histogram. Will also serve as a series name in
            TensorBoard.
        values : `numpy.ndarray`
            Values for building the histogram.
    Returns
    -------
        A `Summary` protobuf of the histogram.
    """
    tag = _clean_tag(tag)
    values = make_numpy_array(values)
    hist = _make_histogram(values.astype(float), bins, max_bins)
    return Summary(value=[Summary.Value(tag=tag, histo=hist)])

def _draw_single_box(image, xmin, ymin, xmax, ymax, display_str, color='black', color_text='black', thickness=2):
    from PIL import ImageDraw, ImageFont
    font = ImageFont.load_default()
    draw = ImageDraw.Draw(image)
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line([(left, top), (left, bottom), (right, bottom),
               (right, top), (left, top)], width=thickness, fill=color)
    if display_str:
        text_bottom = bottom
        # Reverse list and print from bottom to top.
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle(
            [(left, text_bottom - text_height - 2 * margin),
             (left + text_width, text_bottom)], fill=color
        )
        draw.text(
            (left + margin, text_bottom - text_height - margin),
            display_str, fill=color_text, font=font
        )
    return image

def image(tag, tensor, rescale=1, rois=None, labels=None, dataformats='CHW'):
    """Outputs a `Summary` protocol buffer with images.
    The summary has up to `max_images` summary values containing images. The
    images are built from `tensor` which must be 3-D with shape `[height, width,
    channels]` and where `channels` can be:
    *  1: `tensor` is interpreted as Grayscale.
    *  3: `tensor` is interpreted as RGB.
    *  4: `tensor` is interpreted as RGBA.
    Args:
      tag: A name for the generated node. Will also serve as a series name in
        TensorBoard.
      tensor: A 3-D `uint8` or `float32` `Tensor` of shape `[height, width,
        channels]` where `channels` is 1, 3, or 4.
        'tensor' can either have values in [0, 1] (float32) or [0, 255] (uint8).
        The image() function will scale the image values to [0, 255] by applying
        a scale factor of either 1 (uint8) or 255 (float32).
    Returns:
      A scalar `Tensor` of type `string`. The serialized `Summary` protocol
      buffer.
    """
    tag = _clean_tag(tag)
    tensor = make_numpy_array(tensor)
    tensor = convert_to_HWC(tensor, dataformats)
    # Do not assume that user passes in values in [0, 255], use data type to detect
    if tensor.dtype != np.uint8:
        tensor = (tensor * 255.0).astype(np.uint8)
    image = make_image(tensor, rescale=rescale, rois=rois, labels=labels)
    return Summary(value=[Summary.Value(tag=tag, image=image)])

def image_boxes(tag, tensor_image, tensor_boxes, rescale=1, dataformats='CHW', labels=None):
    '''Outputs a `Summary` protocol buffer with images.'''
    tensor_image = make_numpy_array(tensor_image)
    tensor_image = convert_to_HWC(tensor_image, dataformats)
    tensor_boxes = make_numpy_array(tensor_boxes)

    if tensor_image.dtype != np.uint8:
        tensor_image = (tensor_image * 255.0).astype(np.uint8)

    image = make_image(tensor_image,
                       rescale=rescale,
                       rois=tensor_boxes, labels=labels)
    return Summary(value=[Summary.Value(tag=tag, image=image)])

def _draw_single_box(image, xmin, ymin, xmax, ymax, display_str, color='black', color_text='black', thickness=2):
    from PIL import ImageDraw, ImageFont
    font = ImageFont.load_default()
    draw = ImageDraw.Draw(image)
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line([(left, top), (left, bottom), (right, bottom),
               (right, top), (left, top)], width=thickness, fill=color)
    if display_str:
        text_bottom = bottom
        # Reverse list and print from bottom to top.
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle(
            [(left, text_bottom - text_height - 2 * margin),
             (left + text_width, text_bottom)], fill=color
        )
        draw.text(
            (left + margin, text_bottom - text_height - margin),
            display_str, fill=color_text, font=font
        )
    return image

def draw_boxes(disp_image, boxes, labels=None):
    # xyxy format
    num_boxes = boxes.shape[0]
    list_gt = range(num_boxes)
    for i in list_gt:
        disp_image = _draw_single_box(disp_image,
                                      boxes[i, 0],
                                      boxes[i, 1],
                                      boxes[i, 2],
                                      boxes[i, 3],
                                      display_str=None if labels is None else labels[i],
                                      color='Red')
    return disp_image


def make_image(tensor, rescale=1, rois=None, labels=None):
    """Convert an numpy representation image to Image protobuf"""
    from PIL import Image
    height, width, channel = tensor.shape
    scaled_height = int(height * rescale)
    scaled_width = int(width * rescale)
    image = Image.fromarray(tensor)
    if rois is not None:
        image = draw_boxes(image, rois, labels=labels)
    image = image.resize((scaled_width, scaled_height), Image.ANTIALIAS)
    import io
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return Summary.Image(height=height,
                         width=width,
                         colorspace=channel,
                         encoded_image_string=image_string)
