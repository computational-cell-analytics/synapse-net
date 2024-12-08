import os
import pooch

from .file_utils import read_mrc, get_cache_dir


def get_sample_data(name: str) -> str:
    """Get the filepath to SynapseNet sample data, stored as mrc file.

    Args:
        name: The name of the sample data. Currently, we only provide 'tem_2d' and 'tem_tomo'.

    Returns:
        The filepath to the downloaded sample data.
    """
    registry = {
        "tem_2d.mrc": "3c6f9ff6d7673d9bf2fd46c09750c3c7dbb8fa1aa59dcdb3363b65cc774dcf28",
        "tem_tomo.mrc": "fe862ce7c22000d4440e3aa717ca9920b42260f691e5b2ab64cd61c928693c99",
    }
    urls = {
        "tem_2d.mrc": "https://owncloud.gwdg.de/index.php/s/5sAQ0U4puAspcHg/download",
        "tem_tomo.mrc": "https://owncloud.gwdg.de/index.php/s/FJDhDfbT4UxhtOn/download",
    }
    key = f"{name}.mrc"

    if key not in registry:
        valid_names = [k[:-4] for k in registry.keys()]
        raise ValueError(f"Invalid sample name {name}, please choose one of {valid_names}.")

    cache_dir = get_cache_dir()
    data_registry = pooch.create(
        path=os.path.join(cache_dir, "sample_data"),
        base_url="",
        registry=registry,
        urls=urls,
    )
    file_path = data_registry.fetch(key)
    return file_path


def _sample_data(name):
    file_path = get_sample_data(name)
    data, voxel_size = read_mrc(file_path)
    metadata = {"file_path": file_path, "voxel_size": voxel_size}
    add_image_kwargs = {"name": name, "metadata": metadata, "colormap": "gray"}
    return [(data, add_image_kwargs)]


def sample_data_tem_2d():
    return _sample_data("tem_2d")


def sample_data_tem_tomo():
    return _sample_data("tem_tomo")
