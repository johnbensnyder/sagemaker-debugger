# Third Party
import numpy as np

# First Party
from smdebug.core.logger import get_logger

# Local
from .proto.tensor_pb2 import TensorProto
from .proto.tensor_shape_pb2 import TensorShapeProto

logger = get_logger()

# hash value of ndarray.dtype is not the same as np.float class
# so we need to convert the type classes below to np.dtype object
_NP_DATATYPE_TO_PROTO_DATATYPE = {
    np.dtype(np.float16): "DT_HALF",
    np.dtype(np.float32): "DT_FLOAT",
    np.dtype(np.float64): "DT_DOUBLE",
    np.dtype(np.int32): "DT_INT32",
    np.dtype(np.int64): "DT_INT64",
    np.dtype(np.uint8): "DT_UINT8",
    np.dtype(np.uint16): "DT_UINT16",
    np.dtype(np.uint32): "DT_UINT32",
    np.dtype(np.uint64): "DT_UINT64",
    np.dtype(np.int8): "DT_INT8",
    np.dtype(np.int16): "DT_INT16",
    np.dtype(np.complex64): "DT_COMPLEX64",
    np.dtype(np.complex128): "DT_COMPLEX128",
    np.dtype(bool): "DT_BOOL",
    np.dtype([("qint8", "i1")]): "DT_QINT8",
    np.dtype([("quint8", "u1")]): "DT_QUINT8",
    np.dtype([("qint16", "<i2")]): "DT_QINT16",
    np.dtype([("quint16", "<u2")]): "DT_UINT16",
    np.dtype([("qint32", "<i4")]): "DT_INT32",
}


def _get_proto_dtype(npdtype):
    if hasattr(npdtype, "kind"):
        if npdtype.kind == "U" or npdtype.kind == "O" or npdtype.kind == "S":
            return False, "DT_STRING"
    try:
        return True, _NP_DATATYPE_TO_PROTO_DATATYPE[npdtype]
    except KeyError:
        raise TypeError(f"Numpy Datatype: {np.dtype(npdtype)} is currently not supported")


def make_tensor_proto(nparray_data, tag):
    (isnum, dtype) = _get_proto_dtype(nparray_data.dtype)
    dimensions = [
        TensorShapeProto.Dim(size=d, name="{0}_{1}".format(tag, d)) for d in nparray_data.shape
    ]
    tps = TensorShapeProto(dim=dimensions)
    if isnum:
        tensor_proto = TensorProto(
            dtype=dtype, tensor_content=nparray_data.tostring(), tensor_shape=tps
        )
    else:
        tensor_proto = TensorProto(tensor_shape=tps)
        for s in nparray_data:
            sb = bytes(s, encoding="utf-8")
            tensor_proto.string_val.append(sb)
    return tensor_proto

def make_grid(I, ncols=8):
    # I: N1HW or N3HW
    import numpy as np
    assert isinstance(
        I, np.ndarray), 'plugin error, should pass numpy array here'
    if I.shape[1] == 1:
        I = np.concatenate([I, I, I], 1)
    assert I.ndim == 4 and I.shape[1] == 3 or I.shape[1] == 4
    nimg = I.shape[0]
    H = I.shape[2]
    W = I.shape[3]
    ncols = min(nimg, ncols)
    nrows = int(np.ceil(float(nimg) / ncols))
    canvas = np.zeros((I.shape[1], H * nrows, W * ncols), dtype=I.dtype)
    i = 0
    for y in range(nrows):
        for x in range(ncols):
            if i >= nimg:
                break
            canvas[:, y * H:(y + 1) * H, x * W:(x + 1) * W] = I[i]
            i = i + 1
    return canvas

def convert_to_HWC(tensor, input_format):  # tensor: numpy array
    import numpy as np
    assert(len(set(input_format)) == len(input_format)), "You can not use the same dimension shordhand twice. \
        input_format: {}".format(input_format)
    assert(len(tensor.shape) == len(input_format)), "size of input tensor and input format are different. \
        tensor shape: {}, input_format: {}".format(tensor.shape, input_format)
    input_format = input_format.upper()

    if len(input_format) == 4:
        index = [input_format.find(c) for c in 'NCHW']
        tensor_NCHW = tensor.transpose(index)
        tensor_CHW = make_grid(tensor_NCHW)
        return tensor_CHW.transpose(1, 2, 0)

    if len(input_format) == 3:
        index = [input_format.find(c) for c in 'HWC']
        tensor_HWC = tensor.transpose(index)
        if tensor_HWC.shape[2] == 1:
            tensor_HWC = np.concatenate([tensor_HWC, tensor_HWC, tensor_HWC], 2)
        return tensor_HWC

    if len(input_format) == 2:
        index = [input_format.find(c) for c in 'HW']
        tensor = tensor.transpose(index)
        tensor = np.stack([tensor, tensor, tensor], 2)
        return tensor