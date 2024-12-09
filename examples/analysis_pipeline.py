import napari
import pandas as pd
import numpy as np

from scipy.ndimage import binary_closing
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries

from synapse_net.distance_measurements import measure_segmentation_to_object_distances
from synapse_net.file_utils import read_mrc
from synapse_net.imod.to_imod import convert_segmentation_to_spheres
from synapse_net.inference import compute_scale_from_voxel_size, get_model, run_segmentation
from synapse_net.sample_data import get_sample_data


def segment_structures(tomogram, voxel_size):
    # Segment the synaptic vesicles. The data will automatically be resized
    # to match the average voxel size of the training data.
    model_name = "vesicles_3d"  # This is the name for the vesicle model for EM tomography.
    model = get_model(model_name)  # Load the corresponding model.
    # Compute the scale to match the tomogram voxel size to the training data.
    scale = compute_scale_from_voxel_size(voxel_size, model_name)
    vesicles = run_segmentation(tomogram, model, model_name, scale=scale)

    # Segment the active zone.
    model_name = "active_zone"
    model = get_model(model_name)
    scale = compute_scale_from_voxel_size(voxel_size, model_name)
    active_zone = run_segmentation(tomogram, model, model_name, scale=scale)

    # Segment the synaptic compartments.
    model_name = "compartments"
    model = get_model(model_name)
    scale = compute_scale_from_voxel_size(voxel_size, model_name)
    compartments = run_segmentation(tomogram, model, model_name, scale=scale)

    return {"vesicles": vesicles, "active_zone": active_zone, "compartments": compartments}


def n_vesicles(mask, ves):
    return len(np.unique(ves[mask])) - 1


def postprocess_segmentation(segmentations):
    # We find the compartment corresponding to the presynaptic terminal
    # by selecting the compartment with most vesicles. We filter out all
    # vesicles that do not overlap with this compartment.

    vesicles, compartments = segmentations["vesicles"], segmentations["compartments"]

    # First, we find the compartment with most vesicles.
    props = regionprops(compartments, intensity_image=vesicles, extra_properties=[n_vesicles])
    compartment_ids = [prop.label for prop in props]
    vesicle_counts = [prop.n_vesicles for prop in props]
    compartments = (compartments == compartment_ids[np.argmax(vesicle_counts)]).astype("uint8")

    # Filter all vesicles that are not in the compartment.
    props = regionprops(vesicles, compartments)
    filter_ids = [prop.label for prop in props if prop.max_intensity == 0]
    vesicles[np.isin(vesicles, filter_ids)] = 0

    segmentations["vesicles"], segmentations["compartments"] = vesicles, compartments

    # We also apply closing to the active zone segmentation to avoid gaps and then
    # intersect it with the boundary of the presynaptic compartment.
    active_zone = segmentations["active_zone"]
    active_zone = binary_closing(active_zone, iterations=4)
    boundary = find_boundaries(compartments)
    active_zone = np.logical_and(active_zone, boundary).astype("uint8")
    segmentations["active_zone"] = active_zone

    return segmentations


def measure_distances(segmentations, voxel_size):
    # Here, we measure the distances from each vesicle to the active zone.
    # We use the function 'measure_segmentation_to_object_distances' for this,
    # which uses an euclidean distance transform scaled with the voxel size
    # to determine distances.
    vesicles, active_zone = segmentations["vesicles"], segmentations["active_zone"]
    voxel_size = tuple(voxel_size[ax] for ax in "zyx")
    distances, _, _, vesicle_ids = measure_segmentation_to_object_distances(
        vesicles, active_zone, resolution=voxel_size
    )
    # We convert the result to a pandas data frame.
    return pd.DataFrame({"vesicle_id": vesicle_ids, "distance": distances})


def assign_vesicle_pools(vesicle_attributes):
    # We assign the vesicles to their respective pool, 'docked' and 'non-attached',
    # based on the criterion of being within 2 nm from the active zone.
    # We add the pool assignment as a new column to the dataframe with vesicle attributes.
    docked_vesicle_distance = 2  # nm
    vesicle_attributes["pool"] = vesicle_attributes["distance"].apply(
        lambda x: "docked" if x < docked_vesicle_distance else "non-attached"
    )
    return vesicle_attributes


def visualize_results(tomogram, segmentations, vesicle_attributes):
    # Here, we visualize the segmentation and pool assignment result in napari.

    # Create a segmentation to visualize the vesicle pools.
    docked_ids = vesicle_attributes[vesicle_attributes.pool == "docked"].vesicle_id
    non_attached_ids = vesicle_attributes[vesicle_attributes.pool == "non-attached"].vesicle_id
    vesicles = segmentations["vesicles"]
    vesicle_pools = np.isin(vesicles, docked_ids).astype("uint8")
    vesicle_pools[np.isin(vesicles, non_attached_ids)] = 2

    # Create a napari viewer, add the tomogram data and the segmentation results.
    viewer = napari.Viewer()
    viewer.add_image(tomogram)
    for name, segmentation in segmentations.items():
        viewer.add_labels(segmentation, name=name)
    viewer.add_labels(vesicle_pools)
    napari.run()


def save_analysis(segmentations, vesicle_attributes, save_path):
    # Here, we compute the radii and centroid positions of the vesicles,
    # add them to the vesicle attributes and then save all vesicle attributes to
    # an excel table. You can use this table for evaluation of the analysis.
    vesicles = segmentations["vesicles"]
    coordinates, radii = convert_segmentation_to_spheres(vesicles, radius_factor=0.7)
    vesicle_attributes["radius"] = radii
    for ax_id, ax_name in enumerate("zyx"):
        vesicle_attributes[f"center-{ax_name}"] = coordinates[:, ax_id]
    vesicle_attributes.to_excel(save_path, index=False)


def main():
    """This script implements an example analysis pipeline with SynapseNet and applies it to a tomogram.
    Here, we analyze docked and non-attached vesicles in a sample tomogram."""

    # Load the tomogram for our sample data.
    mrc_path = get_sample_data("tem_tomo")
    tomogram, voxel_size = read_mrc(mrc_path)

    # Segment synaptic vesicles, the active zone, and the synaptic compartment.
    segmentations = segment_structures(tomogram, voxel_size)

    # Post-process the segmentations, to find the presynaptic terminal,
    # filter out vesicles not in the terminal, and to 'snape' the AZ to the presynaptic boundary.
    segmentations = postprocess_segmentation(segmentations)

    # Measure the distances between the AZ and vesicles.
    vesicle_attributes = measure_distances(segmentations, voxel_size)

    # Assign the vesicle pools, 'docked' and 'non-attached' vesicles, based on the distances.
    vesicle_attributes = assign_vesicle_pools(vesicle_attributes)

    # Visualize the results.
    visualize_results(tomogram, segmentations, vesicle_attributes)

    # Compute the vesicle radii and combine and save all measurements.
    save_path = "analysis_results.xlsx"
    save_analysis(segmentations, vesicle_attributes, save_path)


if __name__ == "__main__":
    main()
