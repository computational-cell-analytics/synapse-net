import os
from glob import glob
import h5py
from tqdm import tqdm


INPUT_ROOT = "/mnt/ceph-hdd/cold_store/projects/nim00007/new_AZ_train_data"

files = glob(os.path.join(INPUT_ROOT, "**/*.h5"), recursive=True)

key = "labels/az_merged"
for ff in tqdm(files):
    with h5py.File(ff, "a") as f:
        az = f[key][:]
        az = az.squeeze()
        del f[key]
        f.create_dataset(key, data=az, compression="lzf")
