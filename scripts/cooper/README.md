# Segmentation of Synaptic Structures in RT EM Tomography

This folder contains functionality for segmenting and visualizing structures in EM tomograms.
It currently holds the following scripts:
- `run_vesicle_segmentation.py`: Segment vesicles from tomograms stored as mrc.
- `run_mitochondria_segmentation.py`: Segment mitochondria from tomograms stored as mrc.
- `run_cristae_segmentation.py`: Segment cristae from tomograms stored as mrc.
- `visualize_segmentation.py`: Visualize segmentation results with napari.
- `export_vesicles_to_imod.py`: Export the vesicle segmentation to IMOD point annotations.
- `extract_mask_from_imod.py`: Extract a mask from IMOD annotations.

## Usage

Before running any of the scripts you need to activate the python environment by running
```
$ micromamba activate sam
```

The segmentation scripts (`run_..._segmentation.py`) all work similarly and can either run segmentation for a single mrc file or for all mrcs in a folder structure.
For example, you can run vesicle segmentation like this:
```
$ python run_vesicle_segmentation.py -i /path/to/input_folder -o /path/to/output_folder
```
The filepath after `-i` specifices the location of the folder with the mrcs to be segmented and the segmentation results will be stored (as tifs) in the folder following `-o`.
To segment vesicles with an additional mask, you can use the `--mask_path` option.

The segmentation scripts accept additional parameters, e.g. `--force` to overwrite existing segmentations in the output folder (by default these are skipped to avoid unnecessary computation) and `--tile_shape <TILE_Z> <TILE_Y> <TILE_X>` to specify a different tile shape (which may be necessary to avoid running out of GPU memory).
You can check out all parameters by printing the help message of the give string, e.g.
```
$ python run_vesicle_segmentation.py -h
```

To check segmentation results you can use the visualization script like this:
```
$ python visualize_segmentation.py -i /path/to/input_folder -s /path/to/vesicles /path/to/mitos -n vesicles mitos
```
This command will open all tomograms saved under `/path/to/input_folder` and the associated segmentations in the folders specified after `-s` (here vesicle and mito segmentations). The option `-n` is used to set the correct layer names for the segmentations in napari.

The script `export_vesicles_to_imod.py` can be used to export the vesicle segmentations to an imod point model.
You can run it like this:
```
$ python export_vesicles_to_imod.py -i /path/to/input_folder -s /path/to/vesicles -o /path/to/imod_export --min_radius 10 --increase_radius 1.5
```
The parameters have a similar meaning to the other scripts (`-i`: folder with mrc files, `-s`: folder with vesicle segmentation results, `-o`: folder for saving the imod export results); `--min_radius` controls the minimal vesicle radius in nanometer.
The option `--increase_radius` specifies a factor that is used to increase the vesicle diameter for export to better fit the data in IMOD.
In order to run this command, you need to install IMOD on the workstation.
Note: export to imod is not yet implemented for other structures than vesicles.

The script `extract_mask_from_imod.py` can be used to export compartment masks from IMOD. You can run it like this:
```
$ python extract_mask_from_imod.py -i /path/to/tomogram.mrc -m /path/to/imodfile.mod -o /path/to/mask.tif -n object_name
```
Here, `-i` and `-m` are the input paths to mrc file and imod file, `-o` is where the mask extracted as tif will be stored and `-n` is the name
of the object to be extracted from the imod file.


## Data Transfer

We now have storage space to exchange tomogram data etc.
You can copy data over there with the following command:
```
$ rsync -avz --delete -e ssh /path/to/tomogram_folder constantin@sfb1286.ims.bio:data/share/<NAME>
```
Here, `/path/to/tomogram_folder` is the folder with the tomograms (or any other data) you want to copy over.
Replace `<NAME>` with the name for storing the data on the server. This name should not contain any additional `/`s,
otherwise it will lead to problems in the transfer.

You will be prompted to enter a password after running the `rsync` command. You can find the password in the file `pwd.txt` in this folder.
