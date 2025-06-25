import os

# Define the sub-directory structure and files
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
    "scripts": ["train.py", "eval.py", "extract_patches.py", "generate_heatmaps.py"]
}

def create_structure(base_path, struct):
    for name, content in struct.items():
        folder_path = os.path.join(base_path, name)
        os.makedirs(folder_path, exist_ok=True)
        print(f"📁 Created folder: {folder_path}")

        if isinstance(content, list):
            # create empty placeholder files
            for fname in content:
                file_path = os.path.join(folder_path, fname)
                # if it's meant to be a folder (no extension), make it
                if not os.path.splitext(fname)[1]:
                    os.makedirs(file_path, exist_ok=True)
                    print(f"📁 Created sub-folder: {file_path}")
                else:
                    with open(file_path, 'w') as f:
                        pass
                    print(f"📄 Created file:   {file_path}")
        elif isinstance(content, dict):
            # nested structure
            create_structure(folder_path, content)

if __name__ == "__main__":
    create_structure(".", structure)