import os
import shutil
from glob import glob

# Paths to training and validation datasets
training_path = "./brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
validation_path = "./brats20-dataset-training-validation/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData"

# Modalities and file pattern
modalities = ['flair', 't1', 't1ce', 't2', 'seg']
file_structure = "{folder}/{folder_name}_{modality}.nii"

# Destination structure
dataset_structure = {
    'train': ['flair', 't1', 't1ce', 't2', 'mask'],
    'val': ['flair', 't1', 't1ce', 't2']
}


# Helper function to create directories
def create_directories(base_path, subdirs):
    for subdir in subdirs:
        os.makedirs(os.path.join(base_path, subdir), exist_ok=True)


# Function to organize data
def organize_data(src_path, dest_base_path, dataset_type, modalities_map):
    folders = [os.path.basename(folder) for folder in glob(os.path.join(src_path, "*")) if os.path.isdir(folder)]
    create_directories(dest_base_path, [f"{dataset_type}/{modality}" for modality in modalities_map])

    for folder_name in folders:
        for modality in modalities:
            modality_files = glob(os.path.join(src_path, folder_name, f"{folder_name}_{modality}.nii"))
            if modality_files:
                for modality_file in modality_files:
                    if modality == 'seg' and 'mask' in modalities_map:
                        dest_modality = 'mask'
                    else:
                        dest_modality = modality
                    dest_path = os.path.join(dest_base_path, dataset_type, dest_modality)
                    shutil.copy(modality_file, dest_path)


# Organize training data
organize_data(training_path, "./Dataset", "train", dataset_structure['train'])

# Organize validation data
organize_data(validation_path, "./Dataset", "val", dataset_structure['val'])

print("Data organized successfully.")

# Output Result
# Dataset/
# ├── train/
# │   ├── flair/
# │       ├── BraTS20_Training_001_flair.nii
# │       ├── ...
# │   ├── t1/
# │       ├── BraTS20_Training_001_t1.nii
# │       ├── ...
# │   ├── t1ce/
# │       ├── BraTS20_Training_001_t1ce.nii
# │       ├── ...
# │   ├── t2/
# │       ├── BraTS20_Training_001_t2.nii
# │       ├── ...
# │   ├── mask/
# │       ├── BraTS20_Training_001_seg.nii
# │       ├── ...
# └── val/
#     ├── flair/
#         ├── BraTS20_Validation_001_flair.nii
#         ├── ...
#     ├── t1/
#         ├── BraTS20_Validation_001_t1.nii
#         ├── ...
#     ├── t1ce/
#         ├── BraTS20_Validation_001_t1ce.nii
#         ├── ...
#     ├── t2/
#         ├── BraTS20_Validation_001_t2.nii
#         ├── ...
