# SynapseNet: Deep Learning for Automatic Synapse Reconstruction

SynapseNet is a tool for automatic segmentation and analysis of synapses in electron micrographs.
It provides deep neural networks for:
- Synaptic vesicle segmentation in ssTEM (2d data) and (cryo-)electron tomography (3d data)
- Active zone membrane segmentation in electron tomography.
- Mitochondrion segmentation in electron tomography.
- Synaptic compartment segmentation in electron tomography.
- Synaptic ribbon and pre-synaptic density segmentation for ribbon synapses in electron tomography.

It also offers functionality for quantifying synaptic ultrastructure based on segmentation results, for example by measuring vesicle or structure morphology, measuring distances between vesicles and structures, or assigning vesicles into different pools.
SynapseNet mainly targets electron tomography, but can also be appled to other types of electron microscopy,
especially throught the [domain adaptation](domain-adaptation) functionality.

SynapseNet offers a [napari plugin](napari-plugin), [command line interface](command-line-interface), and [python library](python-library).
Please cite our [bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2024.12.02.626387v1) if you use it in your research.


## Requirements & Installation

SynapseNet was developed and tested on Linux. It should be possible to install and use it on Mac or Windows, but we have not tested this.
Furthermore, SynapseNet requires a GPU for segmentation of 3D volumes.

You need a [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) or [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) installation. Follow the instruction at the respective links if you have installed neither. We assume you have `conda` for the rest of the instructions. After installing it, you can use the `conda` command.

To install it you should follow these steps:
- First, download the SynapseNet repository via
```bash
git clone https://github.com/computational-cell-analytics/synapse-net
```
- Then, enter the `synapse-net` folder:
```bash
cd synapse-net
```
- Now you can install the environment for SynapseNet with `conda` from the environment file we provide:
```bash
conda env create -f environment.yaml
```
- You will need to confirm this step. It will take a while. Afterwards you can activate the environment:
```bash
conda activate synapse-net
```
- Finally, install SynapseNet itself into the environment:
```bash
pip install -e .
```

Now you can use all SynapseNet features. From now on, just activate the environment via
```
conda activate synapse-net
```
to use them.

> Note: If you use `mamba` instead of conda just replace `conda` in the commands above with `mamba`.

> Note: We also provide an environment for a CPU version of SynapseNet. You can install it by replacing `environment.yaml` with `environment_cpu.yaml` in the respective command above. This version can be used for 2D vesicle segmentation, but it does not work for 3D segmentation.

> Note: If you have issues with the CUDA version then install a PyTorch that matches your nvidia drivers. See [pytorch.org](https://pytorch.org/) for details.


## Segmentation Models

We currently offer 7 different models for segmenting synaptic structures:
- `vesicles_3d` to segment vesicles in (room-temperature) electron tomograms.
- `vesicles_2d` to segment vesicles in two-dimensional electron micrographs.
- `vesicles_cryo` to segment vesicles in cryogenic electron tomograms.
- `active_zone` to segment active zones in electron tomograms.
- `compartments` to segment synaptic compartments in electron tomograms.
- `mitochondria` to segment mitochondria in electron tomograms.
- `ribbon` to segment structures of the active zones in ribbon synapses (ribbon, presynaptic density and active zone membrane) in electron tomograms.


## Napari Plugin

**The napari plugin will be documented in the next few days!**


## Command Line Functionality

SynapseNet provides a command line interface to segment synaptic structures in mrc files (or other image formats), and to export segmentation results to IMOD.

**Segmentation CLI:** The command `synapse_net.run_segmentation` enables segmentation with all of our [supported models](#segmentation-models). For example, you can call it like this:
```bash
synapse_net.run_segmentation -i /path/to/folder-with-mrc -o /path/to/save-segmentation -m vesicles_3d
```
to segment the synaptic vesicles in all tomograms that are stored as mrc files in the folder `/path/to/folder-with-mrc`. The segmentations will be stored as tif files in the (new) folder `/path/to/save-segmentation`.
You can select a different segmentation model by changing the name after `-m`. For example use `-m mitochondria` to segment mitochondria or `-m vesicles_2d` to segment vesicles in 2D images.
The command offers several other arguments to change the segmentation logic; you can run `synapse_net.run_segmentation -h` for an explanation of these arguments.

**IMOD Export:** We offer two commands to export segmentation results to mod files that can be opened with [3dmod](https://bio3d.colorado.edu/imod/doc/3dmodguide.html), which is part of the [IMOD](https://bio3d.colorado.edu/imod/) software suite:
- `synapse_net.export_to_imod_points` to export a vesicle segmentation to a point model; i.e., representing each vesicle as a sphere.
- `synapse_net.export_to_imod_objects` to export an arbitrary segmentation to a closed contour model.

For example, you can run
```bash
synapse_net.export_to_imod_points -i /path/to/folder-with-mrc -s /path/to/save-segmentation -o /path/to/save-modfiles
```
to export the segmentations saved in `/path/to/save-segmentation` to point models that will be saved in `/path/to/save-modfiles`.

For more options supported by the IMOD exports, please run `synapse_net.export_to_imod_points -h` or `synapse_net.export_to_imod_objects -h`.

> Note: to use these commands you have to install IMOD.


## Python Library

Using the `synapse_net` python library offers the most flexibility for using the SynapseNet functionality.
We offer different functionality for segmenting and analyzing synapses in electron microscopy:
- `synapse_net.inference` for segmenting synaptic structures with [our models](segmentation-models).
- `synapse_net.distance_measurements` for measuring distances between segmented objects.
- `synapse_net.imod` for importing and exporting segmentations from / to IMOD.
- `synapse_net.training` for training U-Nets for synaptic structure segmentation, either via [domain adaptation](#domain-adaptation) or [using data with annotations](network-training).

Please refer to the library documentation below for a full overview of our library's functionality.

### Domain Adaptation

We provide functionality for domain adaptation. It implements a special form of neural network training that can improve segmentation for data from a different condition (e.g. different sample preparation, electron microscopy technique or different specimen), **without requiring additional annotated structures**.
Domain adaptation is implemented in `synapse_net.training.domain_adaptation`. You can find an example script that shows how to use it [here](https://github.com/computational-cell-analytics/synapse-net/blob/main/examples/domain_adaptation.py).

> Note: Domain adaptation only works if the initial model you adapt already finds some of the structures in the data from a new condition. If it does not work you will have to train a network on annotated data.

### Network Training

We also provide functionality for 'regular' neural network training. In this case, you have to provide data **and** manual annotations for the structure(s) you want to segment.
This functionality is implemented in `synapse_net.training.supervised_training`. You can find an example script that shows how to use it [here](https://github.com/computational-cell-analytics/synapse-net/blob/main/examples/network_training.py).
