import cv2 as cv
import numpy as np



'''---Face Reconstruction---'''
import os
import subprocess


'''---Face Reconstruction---'''

'''helper functions'''
def cleanmask_output(lst):
    lst = lst[:-2]
    name = ""
    for i in lst:
        name = name + i + "-"
    return name+"output"

def cleanmask_input(lst):
    lst = lst[:-2]
    name = ""
    for i in lst:
        name = name + i + "-"
    return name+"input"

if __name__ == "__main__":

    from pathlib import Path

    glasses_dir = Path("/home/cash/Thesis/githubog/CFR-GAN-Copy/RealWorldOccludedFaces/images/sunglasses")



    '''Glasses'''
    for subdir in glasses_dir.iterdir():
        if subdir.is_dir():

            foldernamedirty = subdir.name.split("_")
            foldername = cleanmask_output(foldernamedirty)
            foldername_input = cleanmask_input(foldernamedirty)
            inputdir = str(subdir)+"/" + foldername_input
            outputdir = str(subdir) + "/" + foldername

            if not os.path.exists(outputdir):
                os.makedirs(outputdir)
            if not os.path.exists(inputdir):
                os.makedirs(inputdir)


            '''Reconstruction'''
            input_lst = []
            output_lst = []
            for file_path in subdir.iterdir():
                if file_path.is_dir():
                    if "-input" in file_path.name:
                        input_lst.append(file_path)
                    if "-output" in file_path.name:
                        output_lst.append(file_path)
                    
            for i in range(len(input_lst)):

                input_folder = input_lst[i]
                output_folder = output_lst[i]

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
            '''Reconstruction'''

        

            
