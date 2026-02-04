import multiprocessing
import os
import shutil
import tempfile

from concurrent import futures
from glob import glob
from subprocess import run
from typing import Optional, Tuple, Union

import h5py
import imageio.v3 as imageio
import mrcfile
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries
from tqdm import tqdm


def _load_segmentation(segmentation_path, segmentation_key):
    assert os.path.exists(segmentation_path), segmentation_path
    if segmentation_key is None:
        seg = imageio.imread(segmentation_path)
    else:
        with h5py.File(segmentation_path, "r") as f:
            seg = f[segmentation_key][:]
    return seg


def _separate_instances(segmentation):
    boundaries = find_boundaries(segmentation, mode="outer")
    boundaries &= (segmentation != 0)
    segmentation = segmentation.copy()
    segmentation[boundaries] = 0
    return segmentation


# TODO: this has still some issues with some tomograms that has an offset info.
# For now, this occurs for the inner ear data tomograms; it works for Fidi's STEM tomograms.
# Ben's theory is that this might be due to data form JEOL vs. ThermoFischer microscopes.
# To test this I can check how it works for data from Maus et al. / Imig et al., which were taken on a JEOL.
# Can also check out the mrc documentation here: https://www.ccpem.ac.uk/mrc_format/mrc2014.php
def write_segmentation_to_imod(
    mrc_path: str,
    segmentation: Union[str, np.ndarray],
    output_path: str,
    segmentation_key: Optional[str] = None,
    separate_instances: bool = True,
) -> None:
    """Write a segmentation to a mod file as closed contour object(s).

    Args:
        mrc_path: The filepath to the mrc file from which the segmentation was derived.
        segmentation: The segmentation (either as numpy array or filepath to a .tif file).
        output_path: The output path where the mod file will be saved.
        segmentation_key: The key to the segmentation data in case the segmentation is stored in hdf5 files.
        separate_instances: Whether to seperate touching instances before the export.
    """
    cmd = "imodauto"
    cmd_path = shutil.which(cmd)
    assert cmd_path is not None, f"Could not find the {cmd} imod command."

    # Load the segmentation case a filepath was passed.
    if isinstance(segmentation, str):
        segmentation = _load_segmentation(segmentation, segmentation_key)

    # Binarize the segmentation and flip its axes to match the IMOD axis convention.
    if separate_instances:
        segmentation = _separate_instances(segmentation)
    segmentation = (segmentation > 0).astype("uint8")
    segmentation = np.flip(segmentation, axis=1)

    # Read the voxel size and origin information from the mrc file.
    assert os.path.exists(mrc_path)
    with mrcfile.open(mrc_path, mode="r") as f:
        voxel_size = f.voxel_size
        nx, ny, nz = f.header.nxstart, f.header.nystart, f.header.nzstart
        origin = f.header.origin

    # Write the input for imodauto to a temporary mrc file.
    with tempfile.NamedTemporaryFile(suffix=".mrc") as f:
        tmp_path = f.name

        mrcfile.new(tmp_path, data=segmentation, overwrite=True)
        # Write the voxel_size and origin infomration.
        with mrcfile.open(tmp_path, mode="r+") as f:
            f.voxel_size = voxel_size

            f.header.nxstart = nx
            f.header.nystart = ny
            f.header.nzstart = nz
            f.header.origin = (0.0, 0.0, 0.0) * 3 if origin is None else origin

            f.update_header_from_data()

        # Run the command.
        cmd_list = [cmd, "-E", "1", "-u", tmp_path, output_path]
        run(cmd_list)


def convert_segmentation_to_spheres(
    segmentation: np.ndarray,
    verbose: bool = False,
    num_workers: Optional[int] = None,
    resolution: Optional[Tuple[float, float, float]] = None,
    radius_factor: float = 1.0,
    estimate_radius_2d: bool = True,
    props: Optional[list] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract spheres parameterized by center and radius from a segmentation.

    Args:
        segmentation: The segmentation.
        verbose: Whether to print a progress bar.
        num_workers: Number of workers to use for parallelization.
        resolution: The physical resolution of the data.
        radius_factor: Factor for increasing the radius to account for too small exported spheres.
        estimate_radius_2d: If true the distance to boundary for determining the centroid and computing
            the radius will be computed only in 2d rather than in 3d. This can lead to better results
            in case of deformation across the depth axis.
        props: Optional list of regionprops

    Returns:
        The center coordinates.
        The radii.
    """
    num_workers = multiprocessing.cpu_count() if num_workers is None else num_workers
    if props is None:
        props = regionprops(segmentation)

    def coords_and_rads(prop):
        seg_id = prop.label
        bbox = prop.bbox
        ndim = len(bbox) // 2

        # Handle 2D bounding box
        if len(bbox) == 4:
            bb = np.s_[bbox[0]:bbox[2], bbox[1]:bbox[3]]
            mask = segmentation[bb] == seg_id
            if resolution:
                dists = distance_transform_edt(mask, sampling=resolution)
            else:
                dists = distance_transform_edt(mask)

        # Handle 3D bounding box
        elif len(bbox) == 6:
            bb = np.s_[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]]
            mask = segmentation[bb] == seg_id

            if estimate_radius_2d:
                if resolution:
                    dists = np.array([distance_transform_edt(ma, sampling=resolution[1:]) for ma in mask])
                else:
                    dists = np.array([distance_transform_edt(ma) for ma in mask])
            else:
                dists = distance_transform_edt(mask, sampling=resolution)
        else:
            raise ValueError(f"Unsupported bounding box dimensions: {len(bbox)}")

        max_coord = np.unravel_index(np.argmax(dists), mask.shape)
        radius = dists[max_coord] * radius_factor

        offset = np.array(bbox[:ndim])

        coord = np.array(max_coord) + offset

        return coord, radius

    with futures.ThreadPoolExecutor(num_workers) as tp:
        res = list(tqdm(
            tp.map(coords_and_rads, props), disable=not verbose, total=len(props),
            desc="Compute coordinates and radii"
        ))

    coords = [re[0] for re in res]
    rads = [re[1] for re in res]
    return np.array(coords), np.array(rads)


def write_points_to_imod(
    coordinates: np.ndarray,
    radii: np.ndarray,
    shape: Tuple[int, int, int],
    min_radius: Union[float, int],
    output_path: str,
    color: Optional[Tuple[int, int, int]] = None,
) -> None:
    """Write point annotations to a .mod file for IMOD.

    Args:
        coordinates: Array with the point coordinates. #2D or 3D
        radii: Array with the point radii.
        shape: Shape of the volume to be exported.
        min_radius: Minimum radius for export.
        output_path: Where to save the .mod file.
        color: Optional color for writing out the points.
    """
    cmd = "point2model"
    cmd_path = shutil.which(cmd)
    assert cmd_path is not None, f"Could not find the {cmd} imod command."

    def _pad(inp, n=3):
        inp = int(round(inp))
        plen = 6 + (n - len(str(inp)))
        pw = plen * " "
        return f"{pw}{inp}.00"

    is_3d = coordinates.shape[1] == 3
    if not is_3d and coordinates.shape[1] != 2:
        raise ValueError("Coordinates must have shape (N, 2) for 2D or (N, 3) for 3D.")

    with tempfile.NamedTemporaryFile() as tmp_file:
        fname = tmp_file.name
        with open(fname, "w") as f:
            for coord, radius in zip(coordinates, radii):
                if radius < min_radius:
                    continue

                if is_3d:
                    # (z, y, x) indexing
                    x = _pad(coord[2])
                    y = _pad(shape[1] - coord[1])
                    z = _pad(coord[0])

                else:
                    # (y, x) indexing, single z-plane
                    x = _pad(coord[1])
                    y = _pad(shape[0] - coord[0])
                    z = _pad(0)

                f.write(f"{x}{y}{z}{_pad(radius, 2)}\n")
        cmd = [cmd, "-si", "-scat", fname, output_path]

        if color is not None:
            assert len(color) == 3
            r, g, b = [str(co) for co in color]
            cmd += ["-co", f"{r} {g} {b}"]

        run(cmd)


def write_segmentation_to_imod_as_points(
    mrc_path: str,
    segmentation: Union[str, np.ndarray],
    output_path: str,
    min_radius: Union[int, float],
    radius_factor: float = 1.0,
    estimate_radius_2d: bool = True,
    segmentation_key: Optional[str] = None,
) -> None:
    """Write segmentation results to .mod file with imod point annotations.

    This approximates each segmented object as a sphere.

    Args:
        mrc_path: Filepath to the mrc volume that was segmented.
        segmentation: The segmentation (either as numpy array or filepath to a .tif file).
        output_path: Where to save the .mod file.
        min_radius: Minimum radius for export.
        radius_factor: Factor for increasing the radius to account for too small exported spheres.
        estimate_radius_2d: If true the distance to boundary for determining the centroid and computing
            the radius will be computed only in 2d rather than in 3d. This can lead to better results
            in case of deformation across the depth axis.
        segmentation_key: The key to the segmentation data in case the segmentation is stored in hdf5 files.
    """

    # Extract the center coordinates and radii from the segmentation.
    if isinstance(segmentation, str):
        segmentation = _load_segmentation(segmentation, segmentation_key)
    coordinates, radii = convert_segmentation_to_spheres(
        segmentation, radius_factor=radius_factor, estimate_radius_2d=estimate_radius_2d
    )

    # Write the point annotations to imod.
    write_points_to_imod(coordinates, radii, segmentation.shape, min_radius, output_path)


def _get_file_paths(input_path, ext=(".mrc", ".rec")):
    if not os.path.exists(input_path):
        raise Exception(f"Input path not found {input_path}.")

    if isinstance(ext, str):
        ext = (ext,)

    if os.path.isfile(input_path):
        input_files = [input_path]
        input_root = None
    else:
        input_files = []
        for ex in ext:
            input_files.extend(
                sorted(glob(os.path.join(input_path, "**", f"*{ex}"), recursive=True))
            )
        input_root = input_path

    return input_files, input_root


def export_helper(
    input_path: str,
    segmentation_path: str,
    output_root: str,
    export_function: callable,
    force: bool = False,
    segmentation_key: Optional[str] = None,
) -> None:
    """
    Helper function to run imod export for files in a directory.

    Args:
        input_path: The path to the input data.
            Can either be a folder. In this case all mrc files below the folder will be exported.
            Or can be a single mrc file. In this case only this mrc file will be segmented.
        segmentation_path: Filepath to the segmentation results.
            The filestructure must exactly match `input_path`.
        output_root: The path to the output directory where the export results will be saved.
        export_function: The function performing the export to the desired imod format.
            This function must take the path to the input in a .mrc file,
            the path to the segmentation in a .tif file and the output path as only arguments.
            If you want to pass additional arguments to this function the use 'funtools.partial'
        force: Whether to rerun segmentation for output files that are already present.
        segmentation_key: The key to the segmentation data in case the segmentation is stored in hdf5 files.
    """
    input_files, input_root = _get_file_paths(input_path)
    segmentation_files, _ = _get_file_paths(segmentation_path, ext=".tif" if segmentation_key is None else ".h5")
    assert len(input_files) == len(segmentation_files)

    for input_path, seg_path in tqdm(zip(input_files, segmentation_files), total=len(input_files)):
        # Determine the output file name.
        input_folder, input_name = os.path.split(input_path)
        fname = os.path.splitext(input_name)[0] + ".mod"
        if input_root is None:
            output_path = os.path.join(output_root, fname)
        else:  # If we have nested input folders then we preserve the folder structure in the output.
            rel_folder = os.path.relpath(input_folder, input_root)
            output_path = os.path.join(output_root, rel_folder, fname)

        # Check if the output path is already present.
        # If it is we skip the prediction, unless force was set to true.
        if os.path.exists(output_path) and not force:
            continue

        os.makedirs(os.path.split(output_path)[0], exist_ok=True)
        export_function(input_path, seg_path, output_path, segmentation_key=segmentation_key)
