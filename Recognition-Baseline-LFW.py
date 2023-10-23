from deepface import DeepFace
from pathlib import Path
from IPython.display import display, HTML
from tqdm import tqdm 
import csv
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

# Define the directory paths
lfw_recon_dir = Path("lfw_reconstructed\lfw-deepfunneled")


#Path to csv
results_csv = "baseline_lfw_facial_recognition_results.csv"



'''Helper functions'''
def clean(lst):
    lst = lst[:-2]
    name = ""
    for i in lst:
        name = name + i + "-"
    return name

def combine(lst):
    name = ""
    for i in lst:
        name = name + i + "-"
    return name

'''Recognition'''
def recog(img1, img2, name):
    try:
        # Define the models to test
        models = [
            "VGG-Face",
            "DeepFace",

        ]

        # Perform facial recognition using DeepFace with the current model

        for model in models:

            output = DeepFace.verify(img1_path=img1, img2_path=img2, model_name=model, enforce_detection=False, distance_metric="cosine")

            # Append the result to the csv file
            results_lst = [name, output['verified'], output['distance'], output['threshold'], output['model'], output['similarity_metric']]

            with open(results_csv, mode = "a", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(results_lst)
                print("written to csv")

    except Exception as e:
        print(f"An error occurred, Skipping...")

        


'''baseline Results'''
for ident in lfw_recon_dir.iterdir():
    if ident.iterdir():
        for neut in ident.iterdir():
            for dir in ident.iterdir():
                if dir.is_dir():
                    if str(dir.name) == "input":
                        for input_img in dir.iterdir():
                            if neut.name == input_img.name:
                                ident_name = str(neut.name)[:-9]
                                recog(img1 = str(input_img), img2 = str(neut), name = ident_name)





