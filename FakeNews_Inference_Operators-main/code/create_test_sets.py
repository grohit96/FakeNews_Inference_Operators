import numpy as np
import json
import os

# Path to splits.json
splits_json_path = '/scratch/rganesh5/PaperGraph_release_connected/News-Media-Reliability/data/acl2020/splits.json'
# Directory to save the test set files
output_dir = '/scratch/rganesh5/PaperGraph_release_connected'

with open(splits_json_path, 'r') as f:
    splits = json.load(f)

for split_key, split_data in splits.items():
    test_sources = split_data.get("test", [])
    if test_sources:
        test_set_path = os.path.join(output_dir, f'test_set_{split_key}.npy')
        np.save(test_set_path, np.array(test_sources))
        print(f"Generated: {test_set_path}")
