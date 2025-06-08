import os

structure = {
    "data": ["raw", "patches", "masks", "heatmaps"],
    "models": ["cnn_roi_extractor.py", "vit_classifier.py", "dynamic_pruning.py"],
    "utils": ["preprocessing.py", "wsi_reader.py", "patch_extractor.py", "visualization.py"],
    "notebooks": ["eda.ipynb", "training_logs.ipynb"],
    "experiments": {
        "baseline_vit": ["README.md"],
        "cnn_guided_vit": ["README.md"]
    },
    "results": ["figures", "logs"],
    "scripts": ["train.py", "eval.py", "extract_patches.py", "generate_heatmaps.py"],
}

def create_structure(base_path, struct):
    for name, content in struct.items():
        path = os.path.join(base_path, name)
        if isinstance(content, list):
            # create folder
            os.makedirs(path, exist_ok=True)
            print(f"Created folder: {path}")
            # create empty files
            for file_name in content:
                file_path = os.path.join(path, file_name)
                with open(file_path, 'w') as f:
                    pass
                print(f"Created file: {file_path}")
        elif isinstance(content, dict):
            # create folder
            os.makedirs(path, exist_ok=True)
            print(f"Created folder: {path}")
            # recursive call for nested structure
            create_structure(path, content)

if __name__ == "__main__":
    create_structure(".", structure)