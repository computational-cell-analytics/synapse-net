import os
from typing import Dict, List, Optional, Tuple, Union

import mrcfile
import numpy as np
import pooch

try:
    import cryoet_data_portal as cdp
except ImportError:
    cdp = None

try:
    import zarr
except ImportError:
    zarr = None


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


def read_ome_zarr(uri: str, scale_level: int = 0) -> Tuple[np.ndarray, Dict[str, float]]:
    """Read data and voxel size from an ome.zarr file.

    Args:
        uri: Path or url to the ome.zarr file.
        scale_level: The level of the multi-scale image pyramid to load.

    Returns:
        The data read from the file.
        The voxel size read from the file.
    """
    if zarr is None:
        raise RuntimeError("The zarr library is required to read ome.zarr files.")

    # TODO handle URLs / make sure that zarr parses it correctly.
    with zarr.open(uri, "r") as f:
        multiscales = f.attrs["multiscales"][0]
        # TODO double check that the metadata is correct and transform the voxel size to a dict.
        # TODO voxel size is given in Angstrom, divide by 10 to get nanometer
        internal_path = multiscales["dataset"][scale_level]
        data = f[internal_path][:]
        transformation = multiscales["transformation"][scale_level]
        voxel_size = transformation["scale"]

    return data, voxel_size


def read_data_from_cryo_et_portal_run(
    run_id: int, output_path: Optional[str] = None
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Read data and voxel size from a CryoET Data Portal run.

    Args:
        run_id: The ID of the experiment run.
        output_path: The path for saving the data. The data will be streamed if the path is not given.

    Returns:
        The data read from the run.
        The voxel size read from the run
    """
    if cdp is None:
        raise RuntimeError("The CryoET Data portal library is required to read data from the portal.")
