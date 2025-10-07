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
    entry_points={
        "console_scripts": [
            "synapse_net.run_segmentation = synapse_net.tools.cli:segmentation_cli",
            "synapse_net.export_to_imod_points = synapse_net.tools.cli:imod_point_cli",
            "synapse_net.export_to_imod_objects = synapse_net.tools.cli:imod_object_cli",
            "synapse_net.run_supervised_training = synapse_net.training.supervised_training:main",
            "synapse_net.run_domain_adaptation = synapse_net.training.domain_adaptation:main",
            "synapse_net.visualize_vesicle_tools = synapse_net.tools.cli:pool_visualization_cli",
        ],
        "napari.manifest": [
            "synapse_net = synapse_net:napari.yaml",
        ],
    },
)
