# from torch_em.data.datasets.util import download_from_cryo_et_portal
#
# path = "/scratch-grete/projects/nim00007/cryo-et/from_portal"
#
# # TODO this is the stuff to extract later
# ids = [
#  "RN-16498",
#  "RN-16514",
#  "RN-16581",
#  "RN-16641",
# ]
#
# # "24sep24a_Position_102"
# # "24sep24a_Position_113_3"
# # "24sep24a_Position_84"
# # "24sep24a_Position_38"
#
# did = "10443"
# download_from_cryo_et_portal(path, did, download=True)

import cryoet_data_portal as cdp
import s3fs
import os

# S3 filesystem instance
fs = s3fs.S3FileSystem(anon=True)

# Client instance
client = cdp.Client()

# Run IDs (integers)
runs = [16498, 16514, 16581, 16641]

root = "/scratch-grete/projects/nim00007/cryo-et/from_portal"

# Loop over run IDs
for run_id in runs:
    # Query denoised tomograms
    tomograms = cdp.Tomogram.find(
        client,
        [
            cdp.Tomogram.run_id == run_id,
            cdp.Tomogram.processing == "denoised",
        ]
    )

    # Select the first tomogram (there should only be one in this case)
    tomo = tomograms[0]

    # Download the denoised tomogram
    output_folder = os.path.join(root, str(run_id))
    os.makedirs(output_folder, exist_ok=True)
    fname = f"{tomo.id}_{tomo.processing}.mrc"
    output_path = os.path.join(output_folder, fname)
    fs.get(tomo.s3_mrc_file, output_path)
