import shutil
import tempfile
from subprocess import run
from typing import Dict, List, Optional, Tuple

import imageio.v3 as imageio
import numpy as np
from elf.io import open_file
from scipy.ndimage import maximum_filter1d
from skimage.morphology import ball

from tqdm import tqdm
import math

def get_label_names(
    imod_path: str,
    return_types: bool = False,
    return_counts: bool = False,
    filter_empty_labels: bool = True,
) -> Dict[str, str]:
    """Get the ids and names of annotations in an imod file.

    Uses the imodinfo command to parse the contents of the imod file.

    Args:
        imod_path: The filepath to the imod annotation file.
        return_types: Whether to also return the annotation types.
        return_counts: Whether to also return the number of annotations per object.
        filter_empty_labels: Whether to filter out empty objects from the object list.

    Returns:
        Dictionary that maps the object ids to their names.
    """
    cmd = "imodinfo"
    cmd_path = shutil.which(cmd)
    assert cmd_path is not None, f"Could not find the {cmd} imod command."

    label_names, label_types, label_counts = {}, {}, {}

    with tempfile.NamedTemporaryFile() as f:
        tmp_path = f.name
        run([cmd, "-f", tmp_path, imod_path])

        object_id = None
        with open(tmp_path) as f:
            lines = f.readlines()
            for line in lines:

                if line.startswith("OBJECT"):
                    object_id = int(line.rstrip("\n").strip().split()[-1])

                if line.startswith("NAME"):
                    name = line.rstrip("\n").strip().split()[-1]
                    assert object_id is not None
                    label_names[object_id] = name

                if "object uses" in line:
                    type_ = " ".join(line.rstrip("\n").strip().split()[2:]).rstrip(".")
                    label_types[object_id] = type_

                # Parse the number of annotions for this object.
                if "contours" in line:
                    try:
                        count = int(line.rstrip("\n").strip().split()[0])
                        label_counts[object_id] = count
                    # The condition also hits for 'closed countours' / 'open contours'.
                    # We just catch the exception thrown for these cases and continue.
                    except ValueError:
                        pass

                if "CONTOUR" in line and label_types[object_id] == "scattered points":
                    count = int(line.rstrip("\n").strip().split()[2])
                    label_counts[object_id] = count

    if filter_empty_labels:
        empty_labels = [k for k, v in label_counts.items() if v == 0]
        label_names = {k: v for k, v in label_names.items() if k not in empty_labels}
        label_types = {k: v for k, v in label_types.items() if k not in empty_labels}
        label_counts = {k: v for k, v in label_counts.items() if k not in empty_labels}

    if return_types and return_counts:
        return label_names, label_types, label_counts
    elif return_types:
        return label_names, label_types
    elif return_counts:
        return label_names, label_counts

    return label_names


def export_segmentation(
    imod_path: str,
    mrc_path: str,
    object_id: Optional[int] = None,
    output_path: Optional[int] = None,
    require_object: bool = True,
    depth_interpolation: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Export a segmentation mask from IMOD annotations.

    This function uses the imodmop command to extract masks.

    Args:
        imod_path: The filepath to the imod annotation file.
        mrc_path: The filepath to the mrc file with the corresponding tomogram data.
        object_id: The id of the object to be extracted.
            If none is given all objects in the mod file will be extracted.
        output_path: Optional path to a file where to save the extracted mask as tif.
            If not given the segmentation mask will be returned as a numpy array.
        require_object: Whether to throw an error if the object could not be extractd.
        depth_interpolation: Window for interpolating the extracted mask across the depth
            axis. This can be used to close masks which are only annotated in a subset of slices.

    Returns:
        The extracted mask. Will only be returned if `output_path` is not given.
    """
    cmd = "imodmop"
    cmd_path = shutil.which(cmd)
    assert cmd_path is not None, f"Could not find the {cmd} imod command."

    with tempfile.NamedTemporaryFile() as f:
        tmp_path = f.name

        if object_id is None:
            cmd = [cmd, "-ma", "1", imod_path, mrc_path, tmp_path]
        else:
            cmd = [cmd, "-ma", "1", "-o", str(object_id), imod_path, mrc_path, tmp_path]

        run(cmd)
        with open_file(tmp_path, ext=".mrc", mode="r") as f:
            data = f["data"][:]

    segmentation = data == 1
    if require_object and segmentation.sum() == 0:
        id_str = "" if object_id is None else f"for object {object_id}"
        raise RuntimeError(f"Segmentation extracted from {imod_path} {id_str} is empty.")

    if depth_interpolation is not None:
        segmentation = maximum_filter1d(segmentation, size=depth_interpolation, axis=0, mode="constant")

    if output_path is None:
        return segmentation

    imageio.imwrite(output_path, segmentation.astype("uint8"), compression="zlib")

def draw_spheres(
    coordinates: np.ndarray,
    radii: np.ndarray,
    shape: Tuple[int, int, int],
    verbose: bool = True,
) -> np.ndarray:
    """Create a volumetric segmentation by painting spheres around the given coordinates.

    Args:
        coordinates: The center coordinates of the spheres.
        radii: The radii of the spheres.
        shape: The shape of the volume.
        verbose: Whether to print the progress bar.

    Returns:
        The segmentation volume with painted spheres.
    """
    labels = np.zeros(shape, dtype="uint32")

    for label_id, (coord, radius) in tqdm(
        enumerate(zip(coordinates, radii), start=1), total=len(coordinates), disable=not verbose
    ):
        radius = int(radius)
        mask = ball(radius)
        mask_center = np.array([radius] * 3)

        full_slices = []
        mask_slices = []

        for i in range(3):
            start = coord[i] - radius
            end = coord[i] + radius + 1
            vol_start = max(start, 0)
            vol_end = min(end, shape[i])
            full_slices.append(slice(vol_start, vol_end))

            mask_start = vol_start - start
            mask_end = mask_start + (vol_end - vol_start)
            mask_slices.append(slice(mask_start, mask_end))

        full_slice = tuple(full_slices)
        mask_slice = tuple(mask_slices)

        # Skip if sphere is completely out of bounds in any dimension
        if any(s.stop - s.start <= 0 for s in full_slice):
            continue

        full_mask = np.zeros(shape, dtype=bool)
        full_mask[full_slice] = mask[mask_slice]
        labels[full_mask] = label_id

    return labels.astype("uint32")


def load_points_from_imodinfo(
    imod_path: str,
    full_shape: Tuple[int, int, int],
    bb: Optional[Tuple[slice, slice, slice]] = None,
    exclude_labels: Optional[List[int]] = None,
    exclude_label_patterns: Optional[List[str]] = None,
    resolution: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, str]]:
    """Load point coordinates, radii and label information from a .mod file.

    The coordinates and sizes returned will be scaled so that they are in
    the voxel coordinate space if the 'resolution' parameter is passed.
    If it is not passed then the radius will be returned in the physical resolution.

    Args:
        imod_path: The filepath to the .mod file.
        full_shape: The voxel shape of the volume.
        bb: Optional bounding box to limit the extracted points to.
        exclude_labels: Label ids to exclude from the export.
            This can be used to exclude specific labels / classes, specifying them by their id.
        exclude_label_patterns: Label names to exclude from the export.
            This can be used to exclude specific labels / classes, specifying them by their name.
        resolution: The resolution / voxel size of the data. Will be used to scale the radii.

    Returns:
        The center coordinates of the sphere annotations.
        The radii of the spheres.
        The ids of the semantic labels.
        The names of the semantic labels.
    """
    coordinates, sizes, labels = [], [], []
    label_names = {}

    if bb is not None:
        start = [b.start for b in bb]
        shape = [b.stop - sta for b, sta in zip(bb, start)]

    # first round: load the point sizes and labels
    with tempfile.NamedTemporaryFile() as f:
        tmp_path = f.name
        run(["imodinfo", "-p", "-f", tmp_path, imod_path])

        label_id = None
        label_name = None
        with open(tmp_path) as f:
            lines = f.readlines()
            for line in lines:

                if line.startswith("#Object"):
                    line = line.rstrip("\n")
                    label_id = int(line.split(" ")[1])
                    label_name = line.split(",")[1]
                    label_names[label_id] = label_name
                    continue

                if label_id is None:
                    continue
                if exclude_labels is not None and label_id in exclude_labels:
                    continue
                if exclude_label_patterns is not None and any(
                    pattern.lower() in label_name.lower() for pattern in exclude_label_patterns
                ):
                    continue

                try:
                    size = float(line.rstrip("\n"))
                    sizes.append(size)
                    labels.append(label_id)
                except ValueError:
                    continue

    label_names = {lid: label_name for lid, label_name in label_names.items() if lid in labels}

    in_bounds = []
    with tempfile.NamedTemporaryFile() as f:
        tmp_path = f.name
        run(["imodinfo", "-vv", "-f", tmp_path, imod_path])

        label_id = None
        with open(tmp_path) as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("OBJECT"):
                    label_id = int(line.rstrip("\n").split(" ")[-1])
                    continue
                if label_id not in label_names:
                    continue

                values = line.strip().rstrip("\n").split("\t")
                if len(values) != 3:
                    continue

                try:
                    x, y, z = float(values[0]), float(values[1]), float(values[2])
                    coords = [x, y, z]
                    coords = [int(np.round(float(coord))) for coord in coords]
                    # IMOD uses a very weird coordinate system:
                    coords = [coords[2], full_shape[1] - coords[1], coords[0]]

                    in_bound = True
                    if bb is not None:
                        coords = [co - sta for co, sta in zip(coords, start)]
                        if any(co < 0 or co > sh for co, sh in zip(coords, shape)):
                            in_bound = False

                    in_bounds.append(in_bound)
                    coordinates.append(coords)
                except ValueError:
                    continue

    # labelx = {seg_id: int(label_id) for seg_id, label_id in enumerate(labels, 1)}
    # print("Extracted the following labels:", label_names)
    # print("With counts:", {k: v for k, v in zip(*np.unique(list(labelx.values()), return_counts=True))})
    # breakpoint()
    assert len(coordinates) == len(sizes) == len(labels) == len(in_bounds), \
        f"{len(coordinates)}, {len(sizes)}, {len(labels)}, {len(in_bounds)}"

    coordinates, sizes, labels = np.array(coordinates), np.array(sizes), np.array(labels)
    in_bounds = np.array(in_bounds)

    # get rid of empty annotations
    in_bounds = np.logical_and(in_bounds, sizes > 0)

    coordinates, sizes, labels = coordinates[in_bounds], sizes[in_bounds], labels[in_bounds]

    if resolution is not None:
        #resolution = math.ceil((resolution / 2) * 10) / 10
        #print(f"the resolution is {resolution}")
        sizes /= resolution

    if len(coordinates) == 0:
        raise RuntimeError(f"Empty annotations: {imod_path}")
    return coordinates, sizes, labels, label_names


def export_point_annotations(
    imod_path: str,
    shape: Tuple[int, int, int],
    bb: Optional[Tuple[slice, slice, slice]] = None,
    exclude_labels: Optional[List[int]] = None,
    exclude_label_patterns: Optional[List[str]] = None,
    return_coords_and_radii: bool = False,
    resolution: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    """Create a segmentation by drawing spheres corresponding to objects from a .mod file.

    Args:
        imod_path: The filepath to the .mod file.
        shape: The voxel shape of the volume.
        bb: Optional bounding box to limit the extracted points to.
        exclude_labels: Label ids to exclude from the segmentation.
            This can be used to exclude specific labels / classes, specifying them by their id.
        exclude_label_patterns: Label names to exclude from the segmentation.
            This can be used to exclude specific labels / classes, specifying them by their name.
        return_coords_and_radii: Whether to also return the underlying coordinates
            and radii of the exported spheres.
        resolution: The resolution / voxel size of the data. Will be used to scale the radii.

    Returns:
        The exported segmentation.
        The label ids for the instance ids in the segmentation.
        The map of label ids to corresponding obejct names.
    """
    coordinates, radii, labels, label_names = load_points_from_imodinfo(
        imod_path, shape, bb=bb,
        exclude_labels=exclude_labels,
        exclude_label_patterns=exclude_label_patterns,
        resolution=resolution,
    )
    labels = {seg_id: int(label_id) for seg_id, label_id in enumerate(labels, 1)}
    segmentation = draw_spheres(coordinates, radii, shape)
    if return_coords_and_radii:
        return segmentation, labels, label_names, coordinates, radii
    return segmentation, labels, label_names
