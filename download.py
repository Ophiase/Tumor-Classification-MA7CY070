import kagglehub
import os
import shutil


def extract(
        target_folder: str, 
        kaggle_data: str, 
        forced: bool = False) -> str:
    original_folder = kagglehub.dataset_download(kaggle_data)

    if os.path.exists(target_folder) and not forced:
        return target_folder
    
    os.makedirs(target_folder, exist_ok=True)

    for item in os.listdir(original_folder):
        s = os.path.join(original_folder, item)
        d = os.path.join(target_folder, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)

    return target_folder


def main() -> None:
    DATA_DIR = "data"
    os.makedirs(DATA_DIR, exist_ok=True)
    for target_folder, kaggle_data in [
        ("tumor_mri", "masoudnickparvar/brain-tumor-mri-dataset"),
        ("tumor_mri_2", "sartajbhuvaji/brain-tumor-classification-mri"),
        ("tumor_mri_3", "preetviradiya/brian-tumor-dataset")
    ]:
        extract(os.path.join(DATA_DIR, target_folder), kaggle_data)


if __name__ == "__main__":
    main()

