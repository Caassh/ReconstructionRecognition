# Reconstruction of real-world and synthetic occlusions for human face recognition systems

## Step 1: Environment
1. Setup a WSL 18.04 environment
2. Install CONDA for managing pythong environments
3. Clone and follow installation steps of this repo: https://github.com/yeongjoonJu/CFR-GAN in the WSL environment
4. Download the required datsets here: 
ROF: https://github.com/ekremerakin/RealWorldOccludedFaces LFW: https://www.kaggle.com/datasets/jessicali9530/lfw-dataset
5. Adjust all file path paths in the python files so that the correct directories are being processed. 


## Step 2: Cropping LFW
Run the ``LFW-crop.py`` file, make sure the correct directory is in the ``root_dir`` variable.

## Step 3: Align images in ROF
1. Clone this repo into the working dir: https://github.com/foamliu/Face-Alignment
2. Run ``Align-Masked-ROF.py`` for the masked subdirectory images
2. Run ``Align-Sunglasses-ROF.py`` for the sunglasses subdirectory images

## Step 4: Reconstruct ROF
1. Run ``Reconstruct-Masked-ROF.py`` for the masked images.
2. Run ``Reconstruct-Sunglasses-ROF.py`` for the sunglasses images.

## Step 5: Augment LFW
1. Clone this repo in the working dir: https://github.com/isarandi/synthetic-occlusion
2. Run ``augment-LFW.py`` to add synthetic occlusions.


## Step 6: Reconstruct LFW
1. Run ``Reconstruct-LFW.py`` 

## Step 7: Baseline Face Recognition results for ROF
0. ``pip install deepface``
1. Create a csv with these headings ``Identity,Verified,Distance_Metric,Threshold,Model,Similarity_Metric``
2. In the ``Recognition-Baseline-ROF.py``
3. Uncomment the Masked or Sunglasses sections of code to get the baseline results respectively and change the where the info is being stored in the ``recog`` function by changing which csv its being saved to. 
4. Run the ``Recognition-Baseline-ROF.py``

## Step 8: Baseline Face Recognition results for LFW
0. Recomenned to Run all recognition results in windows as the WSL environment can be a bottleneck. 
1. Create a csv with the same heading as in Step 7.
2. Run ``Recognition-LFW.py``

## Step 9: Get Reconstruction Recognition Results for ROF
1. In ``Recognition-ROF.py``: Uncomment which sets of results you want to collect and change the path of the csv being saved to. Similar to Step 7.
2. Run ``Recognition-ROF.py``

## Step 10: Get Reconstruction Recognition Results for LFW
1. Run ``Recognition-LFW.py``
