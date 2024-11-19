import h5py
import numpy as np
import os
import csv
from scipy.ndimage import binary_opening, median_filter,zoom, binary_closing
from skimage.measure import label, regionprops
from synaptic_reconstruction.morphology import compute_object_morphology
from skimage.morphology import ball
from scipy.spatial import ConvexHull
from skimage.draw import polygon

def calculate_AZ_area_per_slice(AZ_slice, pixel_size_nm=1.554):
    """
    Calculate the area of the AZ in a single 2D slice after applying error-reducing processing.
    
    Parameters:
    - AZ_slice (numpy array): 2D array representing a single slice of the AZ segmentation.
    - pixel_size_nm (float): Size of a pixel in nanometers.
    
    Returns:
    - slice_area_nm2 (float): The area of the AZ in the slice in square nanometers.
    """
    # Apply binary opening or median filter to reduce small segmentation errors
    AZ_slice_filtered = binary_opening(AZ_slice, structure=np.ones((3, 3))).astype(int)
    
    # Calculate area in this slice
    num_AZ_pixels = np.sum(AZ_slice_filtered == 1)
    slice_area_nm2 = num_AZ_pixels * (pixel_size_nm ** 2)
    
    return slice_area_nm2

def calculate_total_AZ_area(tomo_path, pixel_size_nm=1.554):
    """
    Calculate the total area of the AZ across all slices in a 3D tomogram file.
    
    Parameters:
    - tomo_path (str): Path to the tomogram file (HDF5 format).
    - pixel_size_nm (float): Size of a pixel in nanometers.
    
    Returns:
    - total_AZ_area_nm2 (float): The total area of the AZ in square nanometers.
    """
    with h5py.File(tomo_path, "r") as f:
        AZ_intersect_seg = f["/AZ/compartment_AZ_intersection_manComp"][:]

    # Calculate the AZ area for each slice along the z-axis
    total_AZ_area_nm2 = 0
    for z_slice in AZ_intersect_seg:
        slice_area_nm2 = calculate_AZ_area_per_slice(z_slice, pixel_size_nm)
        total_AZ_area_nm2 += slice_area_nm2

    return total_AZ_area_nm2

def calculate_AZ_area_simple(tomo_path, pixel_size_nm=1.554):
    """
    Calculate the volume of the AZ (active zone) in a 3D tomogram file.
    
    Parameters:
    - tomo_path (str): Path to the tomogram file (HDF5 format).
    - pixel_size_nm (float): Size of a pixel in nanometers (default is 1.554 nm).
    
    Returns:
    - AZ_volume_nm3 (float): The volume of the AZ in cubic nanometers.
    """
    # Open the file and read the AZ intersection segmentation data
    with h5py.File(tomo_path, "r") as f:
        AZ_intersect_seg = f["/AZ/compartment_AZ_intersection_manComp"][:]

    # Count voxels with label = 1
    num_AZ_voxels = np.sum(AZ_intersect_seg == 1)

    # Calculate the volume in cubic nanometers
    AZ_area_nm2 = num_AZ_voxels * (pixel_size_nm ** 2)

    return AZ_area_nm2

def calculate_AZ_surface(tomo_path, pixel_size_nm=1.554):
    with h5py.File(tomo_path, "r") as f:
        #AZ_seg = f["/AZ/segment_from_AZmodel_v3"][:]
        AZ_seg = f["/filtered_az"][:]
    
    # Apply binary closing to smooth the segmented regions
    struct_elem = ball(1)  # Use a small 3D structuring element
    AZ_seg_smoothed = binary_closing(AZ_seg > 0, structure=struct_elem, iterations=20)

    labeled_seg = label(AZ_seg_smoothed)

    regions = regionprops(labeled_seg)
    if regions:
        # Sort regions by area and get the label of the largest region
        largest_region = max(regions, key=lambda r: r.area)
        largest_label = largest_region.label

        largest_component_mask = (labeled_seg == largest_label)
        AZ_seg_filtered = largest_component_mask.astype(np.uint8)

    else:
        # If no regions found, return an empty array
        AZ_seg_filtered = np.zeros_like(AZ_seg_interp, dtype=np.uint8)
    
    morphology_data = compute_object_morphology(AZ_seg_filtered, "AZ Structure", resolution=(pixel_size_nm, pixel_size_nm, pixel_size_nm))
    surface_column = "surface [nm^2]" #if resolution is not None else "surface [pixel^2]"
    surface_area = morphology_data[surface_column].iloc[0]

    return surface_area

def calculate_AZ_surface_simple(tomo_path, pixel_size_nm=1.554):
    with h5py.File(tomo_path, "r") as f:
        AZ_seg = f["/labels/AZ"][:]
    
    morphology_data = compute_object_morphology(AZ_seg, "AZ Structure", resolution=(pixel_size_nm, pixel_size_nm, pixel_size_nm))
    surface_column = "surface [nm^2]" #if resolution is not None else "surface [pixel^2]"
    surface_area = morphology_data[surface_column].iloc[0]

    return surface_area

def calculate_AZ_surface_convexHull(tomo_path, pixel_size_nm=1.554):
    with h5py.File(tomo_path, "r") as f:
        AZ_seg = f["/AZ/segment_from_AZmodel_v3"][:]

    # Apply binary closing to smooth the segmented regions
    struct_elem = ball(1)  # Use a small 3D structuring element
    AZ_seg_smoothed = binary_closing(AZ_seg > 0, structure=struct_elem, iterations=10)

    labeled_seg = label(AZ_seg_smoothed)

    regions = regionprops(labeled_seg)
    if regions:
        # Sort regions by area and get the label of the largest region
        largest_region = max(regions, key=lambda r: r.area)
        largest_label = largest_region.label

        largest_component_mask = (labeled_seg == largest_label)
        AZ_seg_filtered = largest_component_mask.astype(np.uint8)
    AZ_seg = AZ_seg_filtered
    # Extract coordinates of non-zero points
    points = np.argwhere(AZ_seg > 0)  # Get the coordinates of non-zero (foreground) pixels

    if points.shape[0] < 4:
        # ConvexHull requires at least 4 points in 3D to form a valid hull
        AZ_seg_filtered = np.zeros_like(AZ_seg, dtype=np.uint8)
    else:
        # Apply ConvexHull to the points
        hull = ConvexHull(points)

        # Create a binary mask for the convex hull
        convex_hull_mask = np.zeros_like(AZ_seg, dtype=bool)

        # Iterate over each simplex (facet) of the convex hull and fill in the polygon
        for simplex in hull.simplices:
            # For each face of the convex hull, extract the vertices and convert to a 2D polygon
            polygon_coords = points[simplex]
            rr, cc = polygon(polygon_coords[:, 0], polygon_coords[:, 1])
            convex_hull_mask[rr, cc] = True
        
        # Optional: Label the convex hull mask
        labeled_seg = label(convex_hull_mask)
        regions = regionprops(labeled_seg)

        if regions:
            # Sort regions by area and get the label of the largest region
            largest_region = max(regions, key=lambda r: r.area)
            largest_label = largest_region.label

            largest_component_mask = (labeled_seg == largest_label)
            AZ_seg_filtered = largest_component_mask.astype(np.uint8)

        else:
            AZ_seg_filtered = np.zeros_like(AZ_seg, dtype=np.uint8)

    # Calculate surface area
    morphology_data = compute_object_morphology(AZ_seg_filtered, "AZ Structure", resolution=(pixel_size_nm, pixel_size_nm, pixel_size_nm))
    surface_column = "surface [nm^2]"
    surface_area = morphology_data[surface_column].iloc[0]

    return surface_area

def process_datasets(folder_path, output_csv="AZ_areas.csv", pixel_size_nm=1.554):
    """
    Process all tomograms in multiple datasets within a folder and save results to a CSV.
    
    Parameters:
    - folder_path (str): Path to the folder containing dataset folders with tomograms.
    - output_csv (str): Filename for the output CSV file.
    - pixel_size_nm (float): Size of a pixel in nanometers.
    """
    results = []

    # Iterate over each dataset folder
    for dataset_name in os.listdir(folder_path):
        dataset_path = os.path.join(folder_path, dataset_name)
        
        # Check if it's a directory (skip files in the main folder)
        if not os.path.isdir(dataset_path):
            continue
        
        # Iterate over each tomogram file in the dataset folder
        for tomo_file in os.listdir(dataset_path):
            tomo_path = os.path.join(dataset_path, tomo_file)
            
            # Check if the file is an HDF5 file (optional)
            if tomo_file.endswith(".h5") or tomo_file.endswith(".hdf5"):
                try:
                    # Calculate AZ area
                    #AZ_area = calculate_total_AZ_area(tomo_path, pixel_size_nm)
                    #AZ_area = calculate_AZ_area_simple(tomo_path, pixel_size_nm)
                    #AZ_surface_area = calculate_AZ_surface(tomo_path, pixel_size_nm)
                    #AZ_surface_area = calculate_AZ_surface_convexHull(tomo_path, pixel_size_nm)
                    AZ_surface_area = calculate_AZ_surface_simple(tomo_path, pixel_size_nm)
                    # Append results to list
                    results.append({
                        "Dataset": dataset_name,
                        "Tomogram": tomo_file,
                        "AZ_surface_area": AZ_surface_area
                    })
                except Exception as e:
                    print(f"Error processing {tomo_file} in {dataset_name}: {e}")
    
    # Write results to a CSV file
    with open(output_csv, mode="w", newline="") as csvfile:
        fieldnames = ["Dataset", "Tomogram", "AZ_surface_area"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"Results saved to {output_csv}")

def main():
    # Define the path to the folder containing dataset folders
    folder_path = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/20241102_TOMO_DATA_Imig2014/exported/"
    output_csv = "./analysis_results/manual_AZ_exported/AZ_surface_area.csv"
    # Call the function to process datasets and save results
    process_datasets(folder_path, output_csv = output_csv)

# Call main
if __name__ == "__main__":
    main()
