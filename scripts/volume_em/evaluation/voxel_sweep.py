"""Shared parts of the voxel-size sweeps of the baseline methods, segment_mitonet.py and segment_microsam.py.

Block lookup, resampling to a target voxel size and back, the physically matched size filter, and the
names of the setting folders. It only needs numpy, h5py and skimage, so that the environments of the
baseline methods, which have neither synapse_net nor torch_em, can import it.
"""

import json
import os

import numpy as np
from skimage.transform import resize

TEST_SPLIT = "/mnt/lustre-grete/projects/nim00020/data/volume-em/moebius/test_split"

# The voxel size of the blocks, in nm and ZYX. It is hardcoded because the 'voxel_size' attributes
# of the files cannot be trusted: 4007 stores 17.388 for every axis, 4009 stores
# [0.025, 0.005, 0.005], and the downsampled test_split_* copies disagree with both.
NATIVE_VOXEL_SIZE = (25.0, 5.0, 5.0)

# The synapse-net post-processing drops objects below 1000 native voxels, see min_size_for.
MIN_SIZE_NATIVE = 1000


def find_test_blocks(test_split: str):
    """Find the held-out test blocks, as a mapping of dataset name to filepath.

    The same lookup as compare_models.find_test_blocks, which cannot be imported in these environments.
    """
    blocks = {}
    for dataset in sorted(os.listdir(test_split)):
        folder = os.path.join(test_split, dataset)
        if not os.path.isdir(folder):
            continue
        files = sorted(name for name in os.listdir(folder) if name.endswith(".h5"))
        if len(files) != 1:
            raise ValueError(f"Expect exactly one test block in {folder}, found {len(files)}.")
        blocks[dataset] = os.path.join(folder, files[0])
    if not blocks:
        raise ValueError(f"Did not find any test blocks in {test_split}.")
    return blocks


def find_split_blocks(split_file: str, key: str):
    """Find the blocks of one key of a split file, as a mapping of data root name to filepath.

    Resolves the '<root name>/<relative path>' entries like synapse_net.training.split._resolve_split,
    which cannot be imported in these environments, with the roots the split file carries.
    """
    with open(split_file) as f:
        split = json.load(f)
    roots = split.get("roots", {})
    if not isinstance(roots, dict):
        roots = {os.path.basename(root.rstrip("/")): root for root in roots}
    roots = {name: root.rstrip("/") for name, root in roots.items()}

    blocks = {}
    for entry in split.get(key, []):
        name, _, relative_path = entry.partition("/")
        if name not in roots:
            raise ValueError(f"The split file {split_file} refers to the unknown data root '{name}'.")
        if name in blocks:
            raise ValueError(f"The '{key}' split has more than one block from '{name}', so they cannot "
                             "be named by their data root.")
        blocks[name] = os.path.join(roots[name], *relative_path.split("/"))
    if not blocks:
        raise ValueError(f"The split file {split_file} has no '{key}' entries.")
    return blocks


def min_size_for(voxel_size):
    """The size filter at a voxel size, as the same physical volume as synapse-net's.

    synapse-net drops objects below 1000 native voxels. At 30 nm isotropic that is 23 voxels, while
    empanada's default of 500 voxels would be ~21,600 native voxels, which would drop 18% of the
    mitochondria annotated in the training blocks (see the README) and would flatter MitoNet on
    blocks with large mitochondria. Scaling it also keeps a voxel-size sweep from changing the
    filter strength along with the voxel size.
    """
    return round(MIN_SIZE_NATIVE * float(np.prod(NATIVE_VOXEL_SIZE)) / float(np.prod(voxel_size)))


def setting_prefix(method, voxel_size, invert):
    """The folder prefix that names a setting, e.g. 'mitonet-iso30' or 'microsam-z25-xy10'."""
    z, y, x = (f"{v:g}" for v in voxel_size)
    if z == y == x:
        prefix = f"{method}-iso{z}"
    elif y == x:
        prefix = f"{method}-z{z}-xy{y}"
    else:
        prefix = f"{method}-z{z}-y{y}-x{x}"
    return prefix + ("-inv" if invert else "")


def to_target_resolution(raw, voxel_size):
    """Resample the raw data from the native voxel size to the target voxel size.

    An axis whose shape does not change is left untouched, and a block whose shape does not change
    at all is returned as it is.
    """
    if raw.dtype != np.uint8:
        # empanada normalizes by the maximum of the integer dtype and refuses floats.
        raise ValueError(f"Expect uint8 raw data, got {raw.dtype}.")
    shape = tuple(int(round(n * size / target)) for n, size, target in zip(raw.shape, NATIVE_VOXEL_SIZE, voxel_size))
    if shape == raw.shape:
        return raw
    # skimage only smooths axes that are downsampled, and interpolating an axis at its own grid
    # points is exact, so the axes that keep their shape come out unchanged.
    resampled = resize(raw, shape, order=1, anti_aliasing=True, preserve_range=True)
    return np.clip(np.round(resampled), 0, 255).astype(np.uint8)


def to_native_resolution(seg, shape):
    """Resize a segmentation to the native grid with nearest-neighbor interpolation.

    Every output voxel takes the input voxel its center falls into. Along an upsampled axis every
    input voxel is hit, so nothing is lost; along a downsampled one, such as z for isotropic
    inference below 25 nm, an object thinner than a native slice can be.
    """
    index = [
        np.minimum(((np.arange(n_out) + 0.5) * n_in / n_out).astype(int), n_in - 1)
        for n_in, n_out in zip(seg.shape, shape)
    ]
    return seg[np.ix_(*index)].astype(np.uint32)


def n_objects(seg):
    ids = np.unique(seg)
    return int(len(ids) - (1 if 0 in ids else 0))
