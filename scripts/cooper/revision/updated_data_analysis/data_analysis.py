from synapse_net.distance_measurements import measure_segmentation_to_object_distances
from synapse_net.imod.to_imod import convert_segmentation_to_spheres


def calc_AZ_SV_distance(vesicles, az, resolution):
    """
    Calculate the distance between synaptic vesicles (SVs) and the active zone (AZ).

    Args:
        vesicles (np.ndarray): Segmentation of synaptic vesicles.
        az (np.ndarray): Segmentation of the active zone.
        resolution (tuple): Voxel resolution in nanometers (z, y, x).

    Returns:
        list of dict: Each dict contains 'seg_id' and 'distance', sorted by seg_id.
    """
    distances, _, _, seg_ids = measure_segmentation_to_object_distances(vesicles, az, resolution=resolution)

    # Exclude seg_id == 0
    dist_list = [
        {"seg_id": sid, "distance": dist}
        for sid, dist in zip(seg_ids, distances)
        if sid != 0
    ]
    dist_list.sort(key=lambda x: x["seg_id"])

    return dist_list


def sort_by_distances(input_list):
    """
    Sort a list of dictionaries by the 'distance' key from smallest to largest.

    Args:
        input_list (list of dict): List containing 'distance' as a key in each dictionary.

    Returns:
        list of dict: Sorted list by ascending distance.
    """
    sorted_list = sorted(input_list, key=lambda x: x["distance"])
    return sorted_list


def combine_lists(list1, list2):
    """
    Combine two lists of dictionaries based on the shared 'seg_id' key.

    Args:
        list1 (list of dict): First list with 'seg_id' key.
        list2 (list of dict): Second list with 'seg_id' key.

    Returns:
        list of dict: Combined dictionaries matching by 'seg_id'. Overlapping keys are merged.
    """
    combined_dict = {}

    for item in list1:
        seg_id = item["seg_id"]
        combined_dict[seg_id] = item.copy()

    for item in list2:
        seg_id = item["seg_id"]
        if seg_id in combined_dict:
            for key, value in item.items():
                if key != "seg_id":
                    combined_dict[seg_id][key] = value
        else:
            combined_dict[seg_id] = item.copy()

    combined_list = list(combined_dict.values())
    return combined_list


def calc_SV_diameters(vesicles, resolution):
    """
    Calculate diameters of synaptic vesicles from segmentation data.

    Args:
        vesicles (np.ndarray): Segmentation of synaptic vesicles.
        resolution (tuple): Voxel resolution in nanometers (z, y, x).

    Returns:
        list of dict: Each dict contains 'seg_id' and 'diameter', sorted by seg_id.
    """
    coordinates, radii = convert_segmentation_to_spheres(
        vesicles, resolution=resolution, radius_factor=0.7, estimate_radius_2d=True
    )

    # Assuming the segment ID is the index of the vesicle (same order as radii)
    seg_ids = list(range(len(radii)))
    radii_nm = radii * resolution[0]
    diameters = radii_nm * 2

    # Exclude seg_id == 0
    diam_list = [
        {"seg_id": sid, "diameter": diam}
        for sid, diam in zip(seg_ids, diameters)
        if sid != 0
    ]
    diam_list.sort(key=lambda x: x["seg_id"])

    return diam_list