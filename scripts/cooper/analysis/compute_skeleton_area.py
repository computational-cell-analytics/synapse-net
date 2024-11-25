import numpy as np


def calculate_surface_area(skeleton, voxel_size=(1.0, 1.0, 1.0)):
    """
    Calculate the surface area of a 3D skeletonized object.

    Parameters:
        skeleton (3D array): Binary 3D skeletonized array.
        voxel_size (tuple): Physical size of voxels (z, y, x).

    Returns:
        float: Approximate surface area of the skeleton.
    """
    # Define the voxel dimensions
    voxel_area = (
        voxel_size[1] * voxel_size[2],  # yz-face area
        voxel_size[0] * voxel_size[2],  # xz-face area
        voxel_size[0] * voxel_size[1],  # xy-face area
    )

    # Compute the number of exposed faces for each voxel
    exposed_faces = 0
    directions = [
        (1, 0, 0), (-1, 0, 0),  # x-axis neighbors
        (0, 1, 0), (0, -1, 0),  # y-axis neighbors
        (0, 0, 1), (0, 0, -1),  # z-axis neighbors
    ]

    # Iterate over all voxels in the skeleton
    for z, y, x in np.argwhere(skeleton):
        for i, (dz, dy, dx) in enumerate(directions):
            neighbor = (z + dz, y + dy, x + dx)
            # Check if the neighbor is outside the volume or not part of the skeleton
            if (
                0 <= neighbor[0] < skeleton.shape[0] and
                0 <= neighbor[1] < skeleton.shape[1] and
                0 <= neighbor[2] < skeleton.shape[2] and
                skeleton[neighbor] == 1
            ):
                continue
            exposed_faces += voxel_area[i // 2]

    return exposed_faces
