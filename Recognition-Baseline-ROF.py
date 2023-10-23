from deepface import DeepFace
from pathlib import Path
from IPython.display import display, HTML
from tqdm import tqdm 
import csv
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

# Define the directory paths
sunglasses_dir = Path(r"D:\Documents\ProjectCode\RealWorldOccludedFaces-v1\images\sunglasses")
masked_dir = Path(r"D:\Documents\ProjectCode\RealWorldOccludedFaces-v1\images\masked")
neutral_dir = Path(r"D:\Documents\ProjectCode\RealWorldOccludedFaces-v1\images\neutral")


#Path to csv
results_masked_csv = "baseline_masked_facial_recognition_results.csv"
# results_glasses_csv = "/home/cash/Thesis/githubog/CFR-GAN/baseline_glasses_facial_recognition_results.csv"


'''Helper functions'''
def clean(lst):
    lst = lst[:-2]
    name = ""
    for i in lst:
        name = name + i + "_"
    return name[:-1]

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

            with open(results_masked_csv, mode = "a", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(results_lst)
                print("written to csv")

    except Exception as e:
        print(f"An error occurred, Skipping...")

def is_string_in_csv(csv_file_path, search_string):
    try:
        with open(csv_file_path, "r", newline="") as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                if search_string in row:
                    return True  # String found in the CSV
        return False  # String not found in the CSV
    except FileNotFoundError:
        print("CSV file not found.")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

'''Masked'''
for neutfolder in neutral_dir.iterdir():
    for maskfolder in masked_dir.iterdir():
        maskname = clean(str(maskfolder.name).split("_"))
        if maskname == neutfolder.name:
            with tqdm(total=len(list(neutfolder.iterdir())), desc=f"Processing {maskname}") as pbar:
                for img_neut in neutfolder.iterdir():
                    for img_mask in maskfolder.iterdir():
                        if img_mask.is_file():
                            if is_string_in_csv(results_masked_csv, maskname):
                                continue
                            else:
                                recog(img1=str(img_mask), img2=str(img_neut), name=maskname)
                            pbar.update(1)
                pbar.close()





'''Glasses'''
for neutfolder in neutral_dir.iterdir():
    for glassesfolder in sunglasses_dir.iterdir():
        glassesname = clean(str(glassesfolder.name).split("_"))

        if glassesname == neutfolder.name:

            for img_neut in neutfolder.iterdir():
                for img_glass in glassesfolder.iterdir():

                    if img_glass.is_file():

                        recog(img1 = str(img_glass), img2 = str(img_neut), name = glassesname)









