from deepface import DeepFace
from pathlib import Path
from IPython.display import display, HTML
from tqdm import tqdm 
import csv
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

# Define the directory paths
sunglasses_dir = Path(r"D:\Documents\ProjectCode\RealWorldOccludedFaces-reconstructed\images\sunglasses")
masked_dir = Path(r"D:\Documents\ProjectCode\RealWorldOccludedFaces-reconstructed\images\masked")
neutral_dir = Path(r"D:\Documents\ProjectCode\RealWorldOccludedFaces-reconstructed\images\neutral")


#Path to csv
results_masked_csv = "masked_facial_recognition_results.csv"
results_glasses_csv = "glasses_facial_recognition_results.csv"


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

            with open(results_glasses_csv, mode = "a", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(results_lst)
                print("written to csv")

    except Exception as e:
        print(f"An error occurred, Skipping...")

# '''Masked'''
# for masks in masked_dir.iterdir():
#     for neut in neutral_dir.iterdir():
#         masks_name = clean(masks.name.split("_"))
#         neut_name = combine(neut.name.split("_"))
#         if masks_name == neut_name:
#             for dir in masks.iterdir():
#                 if dir.is_dir():
#                     dirname = str(dir).split("-")
#                     if dirname[len(dirname)-1] == "output":
#                        # Use tqdm to create a progress bar
#                         for img_output in tqdm(list(dir.iterdir()), desc=f"Processing {neut_name[:-1]}"):
#                             for img_neut in neut.iterdir():
#                                 # print("img_output", img_output)
#                                 # print("img_neut", img_neut)
#                                 recog(img1=str(img_output), img2=str(img_neut), name=neut_name[:-1])



'''Glasses'''
for glasses in sunglasses_dir.iterdir():
    for neut in neutral_dir.iterdir():
        glasses_name = clean(glasses.name.split("_"))
        neut_name = combine(neut.name.split("_"))
        if glasses_name == neut_name:
            for dir in glasses.iterdir():
                if dir.is_dir():
                    dirname = str(dir).split("-")
                    if dirname[len(dirname)-1] == "output":
                        for img_output in dir.iterdir():
                            for img_neut in neut.iterdir():

                                recog(img1 = str(img_output), img2 = str(img_neut), name = neut_name[:-1])






