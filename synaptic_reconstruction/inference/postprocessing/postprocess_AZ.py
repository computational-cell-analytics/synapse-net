import numpy as np
from skimage.segmentation import find_boundaries

def find_intersection_boundary(segmented_AZ, segmented_compartment):
    """
    Find the cumulative intersection of the boundary of each label in segmented_compartment with segmented_AZ.

    Parameters:
    segmented_AZ (numpy.ndarray): 3D array representing the active zone (AZ).
    segmented_compartment (numpy.ndarray): 3D array representing the compartment, with multiple labels.

    Returns:
    numpy.ndarray: 3D array with the cumulative intersection of all boundaries of segmented_compartment labels with segmented_AZ.
    """
    # Step 0: Initialize an empty array to accumulate intersections
    cumulative_intersection = np.zeros_like(segmented_AZ, dtype=bool)
    
    # Step 1: Loop through each unique label in segmented_compartment (excluding 0 if it represents background)
    labels = np.unique(segmented_compartment)
    labels = labels[labels != 0]  # Exclude background label (0) if necessary

    for label in labels:
        # Step 2: Create a binary mask for the current label
        label_mask = (segmented_compartment == label)
        
        # Step 3: Find the boundary of the current label's compartment
        boundary_compartment = find_boundaries(label_mask, mode='outer')
        
        # Step 4: Find the intersection with the AZ for this label's boundary
        intersection = np.logical_and(boundary_compartment, segmented_AZ)
        
        # Step 5: Accumulate intersections for each label
        cumulative_intersection = np.logical_or(cumulative_intersection, intersection)
    
    return cumulative_intersection.astype(int)  # Convert boolean array to int (1 for intersecting points, 0 elsewhere)
