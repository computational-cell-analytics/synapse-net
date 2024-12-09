import os
from typing import Dict, List, Optional, Tuple, Union

import mrcfile
import numpy as np
import pooch


def get_cache_dir() -> str:
    """Get the cache directory of synapse net.

    Returns:
        The cache directory.
    """
    cache_dir = os.path.expanduser(pooch.os_cache("synapse-net"))
    return cache_dir


def get_data_path(folder: str, n_tomograms: Optional[int] = 1) -> Union[str, List[str]]:
    """Get the path to all tomograms stored as .rec or .mrc files in a folder.

    Args:
        folder: The folder with tomograms.
        n_tomograms: The expected number of tomograms.

    Returns:
        The filepath or list of filepaths of the tomograms in the folder.
    """
    file_names = os.listdir(folder)
    tomograms = []
    for fname in file_names:
        ext = os.path.splitext(fname)[1]
        if ext in (".rec", ".mrc"):
            tomograms.append(os.path.join(folder, fname))

    if n_tomograms is None:
        return tomograms
    assert len(tomograms) == n_tomograms, f"{folder}: {len(tomograms)}, {n_tomograms}"
    return tomograms[0] if n_tomograms == 1 else tomograms


def _parse_voxel_size(voxel_size):
    parsed_voxel_size = None
    try:
        # The voxel sizes are stored in Angsrrom in the MRC header, but we want them
        # in nanometer. Hence we divide by a factor of 10 here.
        parsed_voxel_size = {
            "x": voxel_size.x / 10,
            "y": voxel_size.y / 10,
            "z": voxel_size.z / 10,
        }
    except Exception as e:
        print(f"Failed to read voxel size: {e}")
    return parsed_voxel_size


def read_voxel_size(path: str) -> Dict[str, float] | None:
    """Read voxel size from mrc/rec file.

    The original unit of voxel size is Angstrom and we convert it to nanometers by dividing it by ten.

    Args:
        path: Path to mrc/rec file.

    Returns:
        Mapping from the axis name to voxel size. None if the voxel size could not be read.
    """
    with mrcfile.open(path, permissive=True) as mrc:
        voxel_size = _parse_voxel_size(mrc.voxel_size)
    return voxel_size


def read_mrc(path: str) -> Tuple[np.ndarray, Dict[str, float]]:
    """Read data and voxel size from mrc/rec file.

    Args:
        path: Path to mrc/rec file.

    Returns:
        The data read from the file.
        The voxel size read from the file.
    """
    with mrcfile.open(path, permissive=True) as mrc:
        voxel_size = _parse_voxel_size(mrc.voxel_size)
        data = np.asarray(mrc.data[:])
    assert data.ndim in (2, 3)

    # Transpose the data to match python axis order.
    data = np.flip(data, axis=1) if data.ndim == 3 else np.flip(data, axis=0)
    return data, voxel_size
