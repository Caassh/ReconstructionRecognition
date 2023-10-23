'''Augmentation'''
import augmentation
import cv2
from pathlib import Path
from tqdm import tqdm


'''Helper Functions'''
def count_files_in_directory(directory):
    directory_path = Path(directory)
    file_count = sum(1 for item in directory_path.iterdir() if item.is_file())
    return file_count

if __name__ == "__main__":
    import os
    import subprocess



    lfw_dir = Path("/home/cash/Thesis/githubog/CFR-GAN/lfw_augmented/lfw-deepfunneled")

    # Initialize a total progress bar
    total_progress_bar = tqdm(desc="Processing Images", dynamic_ncols=True)

    for subdir in lfw_dir.iterdir():
        if subdir.is_dir():
            input_lst = []
            output_lst = []

            for path in subdir.iterdir():
                if path.is_dir():
                    if "input" in path.name:
                        input_lst.append(path)
                    if "output" in path.name:
                        output_lst.append(path)

            for i in range(len(input_lst)):
                input_folder = input_lst[i]
                output_folder = output_lst[i]
                outputnum = count_files_in_directory(output_folder)
                inputnum = count_files_in_directory(output_folder)



                generator_path = "/home/cash/Thesis/githubog/CFR-GAN/saved_models/CFRNet_G_ep55_vgg.pth"
                estimator_path = "/home/cash/Thesis/githubog/CFR-GAN/saved_models/trained_weights_occ_3d.pth"

                inference_command = [
                    "python",
                    "/home/cash/Thesis/githubog/CFR-GAN/inference.py",
                    "--img_path", input_folder,
                    "--save_path", output_folder,
                    "--generator_path", generator_path,
                    "--estimator_path", estimator_path
                ]

                subprocess.run(inference_command)

                # Update the total progress bar
                total_progress_bar.update(1)


    # Close the total progress bar
    total_progress_bar.close()




