import argparse
import os
from functools import partial

import torch
import torch_em
from tqdm import tqdm

from ..cristae_analysis import compute_mito_crista_statistics
from ..file_utils import read_voxel_size
from ..imod.to_imod import (
    _get_file_paths, _load_segmentation, export_helper,
    write_segmentation_to_imod_as_points, write_segmentation_to_imod,
)
from ..inference.inference import _get_model_registry, get_model, get_model_training_resolution, run_segmentation
from ..inference.scalable_segmentation import scalable_segmentation
from ..inference.util import inference_helper, parse_tiling
from .pool_visualization import _visualize_vesicle_pools


def imod_point_cli():
    parser = argparse.ArgumentParser(
        description="Convert a vesicle segmentation to an IMOD point model, "
        "corresponding to a sphere for each vesicle in the segmentation."
    )
    parser.add_argument(
        "--input_path", "-i", required=True,
        help="The filepath to the mrc file or the directory containing the tomogram data."
    )
    parser.add_argument(
        "--segmentation_path", "-s", required=True,
        help="The filepath to the file or the directory containing the segmentations."
    )
    parser.add_argument(
        "--output_path", "-o", required=True,
        help="The filepath to directory where the segmentations will be saved."
    )
    parser.add_argument(
        "--segmentation_key", "-k",
        help="The key in the segmentation files. If not given we assume that the segmentations are stored as tif."
        "If given, we assume they are stored as hdf5 files, and use the key to load the internal dataset."
    )
    parser.add_argument(
        "--min_radius", type=float, default=10.0,
        help="The minimum vesicle radius in nm. Objects that are smaller than this radius will be exclded from the export."  # noqa
    )
    parser.add_argument(
        "--radius_factor", type=float, default=1.0,
        help="A factor for scaling the sphere radius for the export. "
        "This can be used to fit the size of segmented vesicles to the best matching spheres.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Whether to over-write already present export results."
    )
    args = parser.parse_args()

    export_function = partial(
        write_segmentation_to_imod_as_points,
        min_radius=args.min_radius,
        radius_factor=args.radius_factor,
    )

    export_helper(
        input_path=args.input_path,
        segmentation_path=args.segmentation_path,
        output_root=args.output_path,
        export_function=export_function,
        force=args.force,
        segmentation_key=args.segmentation_key,
    )


def imod_object_cli():
    parser = argparse.ArgumentParser(
        description="Convert segmented objects to close contour IMOD models."
    )
    parser.add_argument(
        "--input_path", "-i", required=True,
        help="The filepath to the mrc file or the directory containing the tomogram data."
    )
    parser.add_argument(
        "--segmentation_path", "-s", required=True,
        help="The filepath to the file or the directory containing the segmentations."
    )
    parser.add_argument(
        "--output_path", "-o", required=True,
        help="The filepath to directory where the segmentations will be saved."
    )
    parser.add_argument(
        "--segmentation_key", "-k",
        help="The key in the segmentation files. If not given we assume that the segmentations are stored as tif."
        "If given, we assume they are stored as hdf5 files, and use the key to load the internal dataset."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Whether to over-write already present export results."
    )
    args = parser.parse_args()
    export_helper(
        input_path=args.input_path,
        segmentation_path=args.segmentation_path,
        output_root=args.output_path,
        export_function=write_segmentation_to_imod,
        force=args.force,
        segmentation_key=args.segmentation_key,
    )


def pool_visualization_cli():
    parser = argparse.ArgumentParser(description="Load tomogram data, vesicle pools and additional segmentations for viualization.")  # noqa
    parser.add_argument(
        "--input_path", "-i", required=True,
        help="The filepath to the mrc file containing the tomogram data."
    )
    parser.add_argument(
        "--vesicle_paths", "-v", required=True, nargs="+",
        help="The filepath(s) to the tif file(s) containing the vesicle segmentation."
    )
    parser.add_argument(
        "--table_paths", "-t", required=True, nargs="+",
        help="The filepath(s) to the table(s) with the vesicle pool assignments."
    )
    parser.add_argument(
        "-s", "--segmentation_paths", nargs="+", help="Filepaths for additional segmentations."
    )
    parser.add_argument(
        "--split_pools", action="store_true", help="Whether to split the pools into individual layers.",
    )
    args = parser.parse_args()
    _visualize_vesicle_pools(
        args.input_path, args.vesicle_paths, args.table_paths, args.segmentation_paths, args.split_pools
    )


# TODO: handle kwargs
def segmentation_cli():
    parser = argparse.ArgumentParser(description="Run segmentation.")
    parser.add_argument(
        "--input_path", "-i", required=True,
        help="The filepath to the mrc file or the directory containing the tomogram data."
    )
    parser.add_argument(
        "--output_path", "-o", required=True,
        help="The filepath to directory where the segmentations will be saved."
    )
    model_names = list(_get_model_registry().urls.keys())
    model_names = ", ".join(model_names)
    parser.add_argument(
        "--model", "-m", required=True,
        help=f"The model type. The following models are currently available: {model_names}"
    )
    parser.add_argument(
        "--mask_path", help="The filepath to a tif file with a mask that will be used to restrict the segmentation."
        "Can also be a directory with tifs if the filestructure matches input_path."
    )
    parser.add_argument("--input_key", "-k", required=False)
    parser.add_argument(
        "--force", action="store_true",
        help="Whether to over-write already present segmentation results."
    )
    parser.add_argument(
        "--tile_shape", type=int, nargs=3,
        help="The tile shape for prediction, in ZYX order. Lower the tile shape if GPU memory is insufficient."
    )
    parser.add_argument(
        "--halo", type=int, nargs=3,
        help="The halo for prediction, in ZYX order. Increase the halo to minimize boundary artifacts."
    )
    parser.add_argument(
        "--data_ext", default=".mrc", help="The extension of the tomogram data. By default .mrc."
    )
    parser.add_argument(
        "--checkpoint", "-c", help="Path to a custom model, e.g. from domain adaptation.",
    )
    parser.add_argument(
        "--segmentation_key", "-s",
        help="If given, the outputs will be saved to an hdf5 file with this key. Otherwise they will be saved as tif.",
    )
    parser.add_argument(
        "--scale", type=float,
        help="The factor for rescaling the data before inference. "
        "By default, the scaling factor will be derived from the voxel size of the input data. "
        "If this parameter is given it will over-ride the default behavior. "
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Whether to print verbose information about the segmentation progress."
    )
    parser.add_argument(
        "--scalable", action="store_true", help="Use the scalable segmentation implementation. "
        "Currently this only works for vesicles, mitochondria, or active zones."
    )
    parser.add_argument(
        "--extra_input_path", default=None, help="Filepath to extra inputs, needed for cristae segmentation."
    )
    parser.add_argument(
        "--extra_input_ext", default=".tif", help="File extension for the extra inputs, default is tif."
    )
    args = parser.parse_args()

    if args.checkpoint is None:
        model = get_model(args.model)
    else:
        checkpoint_path = args.checkpoint
        if checkpoint_path.endswith("best.pt"):
            checkpoint_path = os.path.split(checkpoint_path)[0]

        if os.path.isdir(checkpoint_path):  # Load the model from a torch_em checkpoint.
            model = torch_em.util.load_model(checkpoint=checkpoint_path)
        else:
            model = torch.load(checkpoint_path, weights_only=False)
        assert model is not None, f"The model from {args.checkpoint} could not be loaded."

    is_2d = "2d" in args.model
    tiling = parse_tiling(args.tile_shape, args.halo, is_2d=is_2d)

    # If the scale argument is not passed, then we get the average training resolution for the model.
    # The inputs will then be scaled to match this resolution based on the voxel size from the mrc files.
    if args.scale is None:
        model_resolution = get_model_training_resolution(args.model)
        model_resolution = tuple(model_resolution[ax] for ax in ("yx" if is_2d else "zyx"))
        scale = None
    # Otherwise, we set the model resolution to None and use the scaling factor provided by the user.
    else:
        model_resolution = None
        scale = (2 if is_2d else 3) * (args.scale,)

    if args.scalable:
        if not args.model.startswith(("vesicle", "mito", "active")):
            raise ValueError(
                "The scalable segmentation implementation is currently only supported for "
                f"vesicles, mitochondria, or active zones, not for {args.model}."
            )
        segmentation_function = partial(
            scalable_segmentation, model=model, tiling=tiling, verbose=args.verbose
        )
        allocate_output = True

    else:
        segmentation_function = partial(
            run_segmentation, model=model, model_type=args.model, verbose=args.verbose, tiling=tiling,
        )
        allocate_output = False

    inference_helper(
        args.input_path, args.output_path, segmentation_function,
        mask_input_path=args.mask_path, force=args.force, data_ext=args.data_ext,
        output_key=args.segmentation_key, model_resolution=model_resolution, scale=scale,
        allocate_output=allocate_output, extra_input_path=args.extra_input_path,
        extra_input_ext=args.extra_input_ext
    )


def cristae_analysis_helper(
    crista_path, mito_path, output_root,
    crista_key=None, mito_key=None,
    voxel_size=None, tomogram_path=None,
    membrane_thickness_nm=8.0, border_gap_nm=None,
    method="skip", membrane_mode="slice_2d",
    junction_mode="overlap", max_extension_nm=None, terminus_nm=None, min_junction_volume_nm3=None,
    n_jobs=-1, force=False, verbose=False,
):
    """Batch-compute per-mitochondrion cristae statistics and save one CSV per input pair.

    This is the headless equivalent of the napari cristae-analysis widget. It matches crista and
    mitochondria segmentations by sorted order (a single file each, or two directories), computes
    the statistics via :func:`synapse_net.cristae_analysis.compute_mito_crista_statistics`, and
    writes the resulting table next to a mirrored input folder structure.

    Args:
        crista_path: Crista segmentation - a single file or a directory of them.
        mito_path: Mitochondria instance segmentation - a single file or a directory of them.
        output_root: Directory where the ``<stem>_cristae_analysis.csv`` tables are written. A single
            input file writes directly into it; a directory input mirrors the nested folder structure.
        crista_key: Internal dataset key for the crista segmentation. If None the crista files are
            assumed to be tif, otherwise hdf5 with this key.
        mito_key: Internal dataset key for the mitochondria segmentation, analogous to crista_key.
        voxel_size: Voxel size in nm applied to every file. If None it is read per file from the
            raw tomogram given via tomogram_path.
        tomogram_path: Raw tomogram (mrc/rec) - a single file or a directory - used to read the
            voxel size when voxel_size is None.
        membrane_thickness_nm: Membrane shell thickness in nm.
        border_gap_nm: Distance from the volume faces where the membrane is suppressed (nm).
            Defaults to membrane_thickness_nm when None.
        method: How the crista orientation anisotropy is computed ("skip", "fast" or "exact").
        membrane_mode: How the membrane shell is built ("slice_2d" or "shell_3d").
        junction_mode: Which junction detector fills crista_junction_count - "overlap" (the direct
            crista-membrane intersection) or "skeleton" (crista regions reaching close to the inner
            boundary membrane near a crista terminus).
        max_extension_nm: How far in nm a crista may fall short of the inner boundary membrane surface
            and still count ("skeleton" mode only). Defaults to membrane_thickness_nm when None.
        terminus_nm: A near-membrane crista region counts only if it lies within this distance in nm
            of a crista terminus - the free end of the cleaned-up crista skeleton ("skeleton" mode
            only), which rejects a crista running alongside the membrane. Defaults to 20 nm when None.
        min_junction_volume_nm3: Smallest junction volume in nm^3 that counts ("skeleton" mode only).
            Defaults to 50 when None.
        n_jobs: Number of workers for the per-mitochondrion computation (-1 = all cores).
        force: Whether to over-write already present result tables.
        verbose: Whether to show a progress bar over the mitochondria of each file.
    """
    crista_files, crista_root = _get_file_paths(crista_path, ext=".h5" if crista_key else ".tif")
    mito_files, _ = _get_file_paths(mito_path, ext=".h5" if mito_key else ".tif")
    if len(crista_files) != len(mito_files):
        raise ValueError(
            f"The number of crista ({len(crista_files)}) and mitochondria ({len(mito_files)}) "
            "segmentations does not match."
        )

    if voxel_size is not None:
        voxel_sizes = [voxel_size] * len(crista_files)
    elif tomogram_path is not None:
        tomo_files, _ = _get_file_paths(tomogram_path, ext=(".mrc", ".rec"))
        if len(tomo_files) != len(crista_files):
            raise ValueError(
                f"The number of tomograms ({len(tomo_files)}) does not match the number of "
                f"crista segmentations ({len(crista_files)})."
            )
        voxel_sizes = [read_voxel_size(path) for path in tomo_files]
    else:
        raise ValueError("Provide either --voxel_size or --tomogram_path to determine the voxel size.")

    for crista_file, mito_file, this_voxel_size in tqdm(
        zip(crista_files, mito_files, voxel_sizes), total=len(crista_files), desc="Processing files"
    ):
        input_folder, input_name = os.path.split(crista_file)
        fname = os.path.splitext(input_name)[0] + "_cristae_analysis.csv"
        if crista_root is None:
            output_path = os.path.join(output_root, fname)
        else:
            rel_folder = os.path.relpath(input_folder, crista_root)
            output_path = os.path.join(output_root, rel_folder, fname)

        if os.path.exists(output_path) and not force:
            continue

        crista = _load_segmentation(crista_file, crista_key)
        mito = _load_segmentation(mito_file, mito_key)
        stats_df = compute_mito_crista_statistics(
            crista, mito, this_voxel_size,
            membrane_thickness_nm=membrane_thickness_nm, border_gap_nm=border_gap_nm,
            method=method, membrane_mode=membrane_mode,
            junction_mode=junction_mode, max_extension_nm=max_extension_nm,
            terminus_nm=terminus_nm, min_junction_volume_nm3=min_junction_volume_nm3,
            n_jobs=n_jobs, verbose=verbose,
        )

        os.makedirs(os.path.split(output_path)[0], exist_ok=True)
        stats_df.to_csv(output_path, index=False)
        print(f"Saved cristae analysis to {output_path}.")


def cristae_analysis_cli():
    parser = argparse.ArgumentParser(
        description="Compute per-mitochondrion cristae statistics from a crista segmentation and a "
        "mitochondria instance segmentation, and save the results as a CSV table. This is the "
        "command-line equivalent of the napari cristae-analysis widget."
    )
    parser.add_argument(
        "--crista_path", "-c", required=True,
        help="The filepath to the crista segmentation, or a directory containing multiple of them."
    )
    parser.add_argument(
        "--mito_path", "-m", required=True,
        help="The filepath to the mitochondria instance segmentation, or a directory containing multiple of them."
    )
    parser.add_argument(
        "--output_path", "-o", required=True,
        help="The filepath to the directory where the result tables will be saved."
    )
    parser.add_argument(
        "--crista_key",
        help="The key in the crista segmentation file. If not given the crista segmentation is assumed to be tif. "
        "If given, it is assumed to be an hdf5 file and the key is used to load the internal dataset."
    )
    parser.add_argument(
        "--mito_key",
        help="The key in the mitochondria segmentation file, analogous to --crista_key."
    )
    parser.add_argument(
        "--voxel_size", type=float,
        help="The voxel size in nm, applied to all inputs. If not given it is read from the raw tomogram "
        "passed via --tomogram_path."
    )
    parser.add_argument(
        "--tomogram_path",
        help="The filepath to the raw tomogram (mrc/rec), or a directory of them, used to read the voxel size "
        "when --voxel_size is not given."
    )
    parser.add_argument(
        "--membrane_thickness", type=float, default=8.0,
        help="The membrane shell thickness in nm. By default 8.0."
    )
    parser.add_argument(
        "--border_gap", type=float, default=None,
        help="The distance from the volume faces where the membrane is suppressed, in nm. "
        "By default the same as the membrane thickness."
    )
    parser.add_argument(
        "--method", default="skip", choices=["skip", "fast", "exact"],
        help="How the crista orientation anisotropy is computed. 'skip' (default) does not compute it, "
        "'fast' uses a downsampled crop (relative only), 'exact' uses the full-resolution structure tensor."
    )
    parser.add_argument(
        "--membrane_mode", default="slice_2d", choices=["slice_2d", "shell_3d"],
        help="How the membrane shell is built - 'slice_2d' (default, per-Z-slice) or 'shell_3d' (connected 3D shell)."
    )
    parser.add_argument(
        "--junction_mode", default="overlap", choices=["overlap", "skeleton"],
        help="Which junction detector fills crista_junction_count. 'overlap' (default) counts the "
        "connected components of the direct crista-membrane intersection, so a crista that stops "
        "short of the membrane scores no junction. 'skeleton' counts crista regions that come within "
        "--max_extension of the membrane near a crista terminus, so it tolerates a crista segmented "
        "short of the membrane. 'skeleton' requires 3D data."
    )
    parser.add_argument(
        "--max_extension", type=float, default=None,
        help="How far in nm a crista may fall short of the inner boundary membrane surface and still "
        "count as a junction (--junction_mode skeleton only). By default the same as the membrane "
        "thickness."
    )
    parser.add_argument(
        "--terminus_distance", type=float, default=None,
        help="How close in nm a near-membrane crista region must be to a crista terminus - a free end "
        "of the cleaned-up crista skeleton - to count as a junction (--junction_mode skeleton only). "
        "This is what separates a crista ending at the membrane from one running alongside it. By "
        "default 20 nm."
    )
    parser.add_argument(
        "--min_junction_volume", type=float, default=None,
        help="The smallest junction volume in nm^3 that counts (--junction_mode skeleton only). This "
        "only removes specks; it does not address the fact that skeleton mode over-counts on densely "
        "packed cristae - see docs/cristae_analysis.md. By default 50."
    )
    parser.add_argument(
        "--n_jobs", type=int, default=-1,
        help="The number of workers for the per-mitochondrion computation. By default -1 (all cores)."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Whether to over-write already present result tables."
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Whether to show a progress bar over the mitochondria of each file."
    )
    args = parser.parse_args()

    cristae_analysis_helper(
        args.crista_path, args.mito_path, args.output_path,
        crista_key=args.crista_key, mito_key=args.mito_key,
        voxel_size=args.voxel_size, tomogram_path=args.tomogram_path,
        membrane_thickness_nm=args.membrane_thickness, border_gap_nm=args.border_gap,
        method=args.method, membrane_mode=args.membrane_mode,
        junction_mode=args.junction_mode, max_extension_nm=args.max_extension,
        terminus_nm=args.terminus_distance, min_junction_volume_nm3=args.min_junction_volume,
        n_jobs=args.n_jobs, force=args.force, verbose=args.verbose,
    )
