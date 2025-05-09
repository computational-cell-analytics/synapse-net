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
        ],
        "napari.manifest": [
            "synapse_net = synapse_net:napari.yaml",
        ],
    },
)
