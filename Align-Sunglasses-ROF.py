import cv2 as cv
import numpy as np
import os


'''---Face Alignment---'''
from align_faces import warp_and_crop_face, get_reference_facial_points
from mtcnn.detector import MtcnnDetector
def align(img_path, output_size, inputdir):
    try:

        lastslash = img_path.rfind('/')
        filename = img_path[lastslash + 1:]
        name, ext = os.path.splitext(filename)
        aligned_name = f"{name}-aligned{ext}"
        folderpath = img_path[:lastslash] + "/"
        # print("Folderpath", folderpath)
        img = cv.imread(img_path)

        detector = MtcnnDetector()
        _, facial5points = detector.detect_faces(img)

        if len(facial5points) == 0:
            print(f"No facial landmarks found in {img_path}. Skipping...")
            return None

        facial5points = np.reshape(facial5points[0], (2, 5))

        default_square = True
        inner_padding_factor = 0.25
        outer_padding = (0, 0)

        # get the reference 5 landmarks position in the crop settings
        reference_5pts = get_reference_facial_points(
            output_size, inner_padding_factor, outer_padding, default_square)

        dst_img = warp_and_crop_face(img, facial5points, reference_pts=reference_5pts, crop_size=output_size)
        aligned_path = os.path.join(inputdir, aligned_name)
        cv.imwrite(aligned_path, dst_img)

    except Exception as e:
        print(f"An error has occurred: {e}")
        return None
'''---Face Alignment---'''



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

    '''Sunglasses'''
    for subdir in glasses_dir.iterdir():
        if subdir.is_dir():

            '''Aligning Faces'''
            foldernamedirty = subdir.name.split("_")
            foldername = cleanmask_output(foldernamedirty)
            foldername_input = cleanmask_input(foldernamedirty)
            inputdir = str(subdir)+"/" + foldername_input
            outputdir = str(subdir) + "/" + foldername

            if not os.path.exists(outputdir):
                os.makedirs(outputdir)
            if not os.path.exists(inputdir):
                os.makedirs(inputdir)

            for file_path in subdir.iterdir():
                if file_path.suffix == ".jpg":
                    align(str(file_path),output_size=(500,500),inputdir=inputdir+"/")
            '''Aligning Faces'''








        

            
