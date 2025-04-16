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

try:
    import s3fs
except ImportError:
    s3fs = None


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


def read_ome_zarr(uri: str, scale_level: int = 0, fs=None) -> Tuple[np.ndarray, Dict[str, float]]:
    """Read data and voxel size from an ome.zarr file.

    Args:
        uri: Path or url to the ome.zarr file.
        scale_level: The level of the multi-scale image pyramid to load.
        fs: S3 filesystem to use for initializing the store.

    Returns:
        The data read from the file.
        The voxel size read from the file.
    """
    if zarr is None:
        raise RuntimeError("The zarr library is required to read ome.zarr files.")

    def parse_s3_uri(uri):
        return uri.lstrip("s3://")

    if uri.startswith("s3"):
        if fs is None:
            fs = s3fs.S3FileSystem(anon=True)
        s3_uri = parse_s3_uri(uri)
        store = s3fs.S3Map(root=s3_uri, s3=fs, check=False)
    elif fs is not None:
        s3_uri = parse_s3_uri(uri)
        store = s3fs.S3Map(root=s3_uri, s3=fs, check=False)
    else:
        if not os.path.exists(uri):
            raise ValueError(f"Cannot find the filepath at {uri}.")
        store = uri

    with zarr.open(store, "r") as f:
        multiscales = f.attrs["multiscales"][0]

        # Read the axis and transformation metadata for this dataset, to determine the voxel size.
        axes = [axis["name"] for axis in multiscales["axes"]]
        assert set(axes) == set("xyz")
        units = [axis.get("unit", "angstrom") for axis in multiscales["axes"]]
        assert all(unit in ("angstrom", "nanometer") for unit in units)

        transformations = multiscales["datasets"][scale_level]["coordinateTransformations"]
        scale_transformation = [trafo["scale"] for trafo in transformations if trafo["type"] == "scale"][0]

        # Convert the given unit size to nanometer.
        # (It is typically given in angstrom, and we have to divide by a factor of 10).
        unit_factor = [10.0 if unit == "angstrom" else 1.0 for unit in units]
        voxel_size = {axis: scale / factor for axis, scale, factor in zip(axes, scale_transformation, unit_factor)}

        # Get the internale path for the given scale and load the data.
        internal_path = multiscales["datasets"][scale_level]["path"]
        data = f[internal_path][:]

    return data, voxel_size


def read_data_from_cryo_et_portal_run(
    run_id: int,
    output_path: Optional[str] = None,
    use_zarr_format: bool = True,
    processing_type: str = "denoised",
    id_field: str = "run_id",
    scale_level: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Read data and voxel size from a CryoET Data Portal run.

    Args:
        run_id: The ID of the experiment run.
        output_path: The path for saving the data. The data will be streamed if the path is not given.
        use_zarr_format: Whether to use the data in zarr format instead of mrc.
        processing_type: The processing type of the tomogram to download.
        id_field: The name of the id field. One of 'id' or 'run_id'.
            The 'id' references specific tomograms, whereas 'run_id' references a collection of experimental data.
        scale_level: The scale level to read from the data. Only valid for zarr data.

    Returns:
        The data read from the run.
        The voxel size read from the run.
    """
    assert id_field in ("id", "run_id")
    if output_path is not None and os.path.exists(output_path):
        return read_ome_zarr(output_path) if use_zarr_format else read_mrc(output_path)

    if cdp is None:
        raise RuntimeError("The CryoET data portal library is required to download data from the portal.")
    if s3fs is None:
        raise RuntimeError("The CryoET data portal download requires s3fs download.")

    client = cdp.Client()

    fs = s3fs.S3FileSystem(anon=True)
    tomograms = cdp.Tomogram.find(
        client, [getattr(cdp.Tomogram, id_field) == run_id, cdp.Tomogram.processing == processing_type]
    )
    if len(tomograms) == 0:
        return None, None
    if len(tomograms) > 1:
        raise NotImplementedError
    tomo = tomograms[0]

    if use_zarr_format:
        if output_path is None:
            scale_level = 0 if scale_level is None else scale_level
            data, voxel_size = read_ome_zarr(tomo.s3_omezarr_dir, fs=fs)
        else:
            # TODO: write the outuput to ome zarr, for all scale levels.
            raise NotImplementedError
    else:
        if scale_level is not None:
            raise ValueError
        if output_path is None:
            raise RuntimeError("You have to pass an output_path to download the data as mrc file.")
        fs.get(tomo.s3_mrc_file, output_path)
        data, voxel_size = read_mrc(output_path)

    return data, voxel_size
