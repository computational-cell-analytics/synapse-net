import numpy as np
from scipy.ndimage import binary_erosion

def find_intersection_boundary(segmented_AZ, segmented_compartment):
    """
    Find the intersection of the boundary of segmented_compartment with segmented_AZ.

    Parameters:
    segmented_AZ (numpy.ndarray): 3D array representing the active zone (AZ).
    segmented_compartment (numpy.ndarray): 3D array representing the compartment.

    Returns:
    numpy.ndarray: 3D array with the intersection of the boundary of segmented_compartment and segmented_AZ.
    """
    # Step 0: Binarize the segmented_compartment
    binarized_compartment = (segmented_compartment > 0).astype(int)
    
    # Step 1: Create a binary mask of the compartment boundary
    eroded_compartment = binary_erosion(binarized_compartment)
    boundary_compartment = binarized_compartment - eroded_compartment
    
    # Step 2: Find the intersection with the AZ
    intersection = np.logical_and(boundary_compartment, segmented_AZ)
    
    return intersection.astype(int)  # Convert boolean array to int (1 for intersecting points, 0 elsewhere)
