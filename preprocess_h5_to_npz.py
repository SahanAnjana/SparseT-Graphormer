import pandas as pd
import numpy as np
import os

# Change these paths as needed
h5_paths = ["/content/metr-la.h5","/content/pems-bay.h5",]
out_dirs = ["/content/SparseT-Graphormer/src/data/traffic/metar-la","/content/SparseT-Graphormer/src/data/traffic/pems-bay"]
n_hist = 12
n_pred = 12
split_ratio = [0.7, 0.2, 0.1]  # train, val, test

for h5_path, out_dir in zip(h5_paths, out_dirs):
    df = pd.read_hdf(h5_path)
    data = df.values  # shape: (timesteps, num_sensors)

    num_samples = data.shape[0] - n_hist - n_pred + 1
    indices = np.arange(num_samples)

    # Shuffle indices for splitting
    np.random.seed(42)
    np.random.shuffle(indices)

    n_train = int(split_ratio[0] * num_samples)
    n_val = int(split_ratio[1] * num_samples)
    n_test = num_samples - n_train - n_val

    splits = {
        "train": indices[:n_train],
        "val": indices[n_train:n_train+n_val],
        "test": indices[n_train+n_val:]
    }

    for split, idxs in splits.items():
        features = []
        targets = []
        for i in idxs:
            features.append(data[i:i+n_hist])
            targets.append(data[i+n_hist:i+n_hist+n_pred])
        features = np.array(features)  # shape: (samples, n_hist, num_sensors)
        targets = np.array(targets)    # shape: (samples, n_pred, num_sensors)
        np.savez_compressed(
            os.path.join(out_dir, f"{split}_hist{n_hist}_pred{n_pred}.npz"),
            features=features,
            targets=targets
        )
        print(f"Saved {split} split: {features.shape}, {targets.shape}")