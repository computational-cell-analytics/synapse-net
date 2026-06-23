import runpy
from setuptools import setup, find_packages

__version__ = runpy.run_path("synapse_net/__version__.py")["__version__"]


setup(
    name="synapse_net",
    packages=find_packages(exclude=["test"]),
    version=__version__,
    author="Constantin Pape; Sarah Muth; Luca Freckmann",
    url="https://github.com/computational-cell-analytics/synapse-net",
    license="MIT",
    install_requires=[
        "bioimage-cpp",
        "python-elf>=0.9.0",
        "torch_em>=0.9.0",
        "numpy",
        "scipy",
        "scikit-image",
        "scikit-learn",
        "h5py",
        "imageio",
        "mrcfile",
        "pandas",
        "pooch",
        "tqdm",
        "trimesh",
    ],
    extras_require={
        # Dependencies for the napari plugin GUI. Install with `pip install synapse_net[napari]`.
        "napari": [
            "napari>=0.5.0,<0.7.0",
            "magicgui",
            "qtpy",
            "superqt",
            "napari-skimage-regionprops",
        ],
    },
    entry_points={
        "console_scripts": [
            "synapse_net.run_segmentation = synapse_net.tools.cli:segmentation_cli",
            "synapse_net.export_to_imod_points = synapse_net.tools.cli:imod_point_cli",
            "synapse_net.export_to_imod_objects = synapse_net.tools.cli:imod_object_cli",
            "synapse_net.run_supervised_training = synapse_net.training.supervised_training:main",
            "synapse_net.run_domain_adaptation = synapse_net.training.domain_adaptation:main",
            "synapse_net.visualize_vesicle_pools = synapse_net.tools.cli:pool_visualization_cli",
        ],
        "napari.manifest": [
            "synapse_net = synapse_net:napari.yaml",
        ],
    },
)
