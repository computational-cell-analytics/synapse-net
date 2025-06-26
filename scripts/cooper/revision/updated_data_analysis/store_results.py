import os
import pandas as pd

def get_group(input_path):
    """
    Determines whether a tomogram belongs to 'CTRL' or 'KO' group.

    Parameters:
        input_path (str): Path to the input .h5 file.

    Returns:
        str: 'CTRL' if input_path contains 'CTRL', else 'KO'.
    """
    return 'CTRL' if 'CTRL' in input_path else 'KO'


def get_tomogram_name(input_path):
    """
    Extracts the tomogram name from the input path (without extension).

    Parameters:
        input_path (str): Path to the input .h5 file.

    Returns:
        str: Tomogram base name without extension.
    """
    return os.path.splitext(os.path.basename(input_path))[0]


def prepare_output_directory(base_output, group):
    """
    Ensures that the group-specific output directory exists.

    Parameters:
        base_output (str): Base output directory.
        group (str): Group name ('CTRL' or 'KO').

    Returns:
        str: Full path to the group-specific directory.
    """
    group_dir = os.path.join(base_output, group)
    os.makedirs(group_dir, exist_ok=True)
    return group_dir

def write_or_append_csv(file_path, new_data):
    """
    Writes a new DataFrame to CSV, or appends a new column(s) to an existing one.

    Parameters:
        file_path (str): Path to the target CSV file.
        new_data (pd.DataFrame): DataFrame to write or append.
    """
    print(f"saving {file_path}")
    if os.path.exists(file_path):
        existing = pd.read_csv(file_path, index_col=0)
        combined = pd.concat([existing, new_data], axis=1)
    else:
        combined = new_data

    combined.to_csv(file_path)

def save_filtered_dataframes(output_dir, tomogram_name, df):
    """
    Saves the sorted segment data into multiple filtered CSV files.

    Parameters:
        output_dir (str): Directory where CSVs will be saved.
        tomogram_name (str): Name of the tomogram (used as column header).
        df (pd.DataFrame): DataFrame containing 'seg_id', 'distance', and 'diameter'.
    """
    thresholds = {
        'AZ_distances': None,
        'AZ_distances_within_200': 200,
        'AZ_distances_within_100': 100,
        'AZ_distances_within_40': 40,
        'AZ_distances_within_40_with_diameters': 40,
    }

    for filename, max_dist in thresholds.items():
        file_path = os.path.join(output_dir, f"{filename}.csv")
        filtered_df = df if max_dist is None else df[df['distance'] <= max_dist]

        if filename == 'AZ_distances_within_40_with_diameters':
            data = pd.DataFrame({
                f"{tomogram_name}_distance": filtered_df['distance'].values,
                f"{tomogram_name}_diameter": filtered_df['diameter'].values
            })
        else:
            data = pd.DataFrame({tomogram_name: filtered_df['distance'].values})

        write_or_append_csv(file_path, data)

def save_filtered_dataframes_with_seg_id(output_dir, tomogram_name, df):
    """
    Saves segment data including seg_id into separate CSV files.

    Parameters:
        output_dir (str): Directory to save files.
        tomogram_name (str): Base name of the tomogram.
        df (pd.DataFrame): DataFrame with 'seg_id', 'distance', 'diameter'.
    """
    thresholds = {
        'AZ_distances_with_seg_id': None,
        'AZ_distances_within_200_with_seg_id': 200,
        'AZ_distances_within_100_with_seg_id': 100,
        'AZ_distances_within_40_with_seg_id': 40,
        'AZ_distances_within_40_with_diameters_and_seg_id': 40,
    }

    for filename, max_dist in thresholds.items():
        #storing with seg ID data in subfolder
        with_segID_dir = os.path.join(output_dir, "with_segID")
        os.makedirs(with_segID_dir, exist_ok=True)
        file_path = os.path.join(with_segID_dir, f"{filename}.csv")
        
        filtered_df = df if max_dist is None else df[df['distance'] <= max_dist]

        if filename == 'AZ_distances_within_40_with_diameters_and_seg_id':
            data = pd.DataFrame({
                f"{tomogram_name}_seg_id": filtered_df['seg_id'].values,
                f"{tomogram_name}_distance": filtered_df['distance'].values,
                f"{tomogram_name}_diameter": filtered_df['diameter'].values
            })
        else:
            data = pd.DataFrame({
                f"{tomogram_name}_seg_id": filtered_df['seg_id'].values,
                f"{tomogram_name}_distance": filtered_df['distance'].values
            })

        write_or_append_csv(file_path, data)

def run_store_results(input_path, analysis_output, sorted_list):
    """
    Processes a single tomogram's sorted segment data and stores results into categorized CSV files.

    Parameters:
        input_path (str): Path to the input .h5 file.
        analysis_output (str): Directory where results should be saved.
        sorted_list (list of dict): List of dicts with 'seg_id', 'distance', and 'diameter',
                                    sorted by distance ascendingly.
    """
    group = get_group(input_path)
    tomogram_name = get_tomogram_name(input_path)
    group_dir = prepare_output_directory(analysis_output, group)
    df = pd.DataFrame(sorted_list)

    # First run: distances only 
    save_filtered_dataframes(group_dir, tomogram_name, df)

    # Second run: include seg_id in the filenames and output
    save_filtered_dataframes_with_seg_id(group_dir, tomogram_name, df)
