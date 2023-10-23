'''Augmentation'''
import augmentation
import cv2
from pathlib import Path
from tqdm import tqdm


def aug(path, savepath):
    try:
        occluders = augmentation.load_occluders(pascal_voc_root_path="/home/cash/Thesis/githubog/CFR-GAN/lfw_dataset/synthetic-occlusion/VOCdevkit/VOC2012")
        img = cv2.resize(cv2.imread(path), (256, 256))
        occ_img = augmentation.occlude_with_objects(img, occluders)
        filename = str(Path(path).name)
        outputdir = str(savepath) + "/" + filename
        cv2.imwrite(outputdir, occ_img)
    except Exception as e:
        print(f"An error occurred: {e}")



if __name__ == "__main__":
    import os
    import subprocess

    lfw_dir = Path("/home/cash/Thesis/githubog/CFR-GAN/lfw_augmented/lfw-deepfunneled")


    for subdir in lfw_dir.iterdir():
        if subdir.is_dir():
            outputdir = str(subdir) + "/" + "output"
            inputdir = str(subdir) + "/" + "input"

            if not os.path.exists(outputdir):
                os.makedirs(outputdir)
            
            if not os.path.exists(inputdir):
                os.makedirs(inputdir)

            # Create a progress bar for the current subdirectory
            subdir_progress_bar = tqdm(subdir.iterdir(), desc=f"Processing {subdir.name}", dynamic_ncols=True)
            
            '''Augmentation'''
            for path in subdir_progress_bar:
                if path.suffix == ".jpg":
                    aug(str(path), savepath=inputdir)

    






