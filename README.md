# Project BubblesDNN
Neural network for analysing data obtained with the Bubbles technique.

## Data storage
Data and scripts reside on Crunchy: ```diff 
/loctmp2/data/vog20246/BubblesDNN/

via smb: \\crunchy.ur.de\data2\vog20246
 
Raw data for all participants was saved in one large file, 
- ../BKH_Bubbles/raw/BubblesRawData.mat.

From that file, the individual bubbelized images as shown in the experiments are reconstructed with calling
- get_dset_FullFaceCrunchy2.m

The call creates a folder
- dsetComposite 

with subfolders for each condition, as well as a log file 
- BubblesProtocolComposite.txt 

with filenames, participant names, condition number, group, original trial number and number of Bubbles in that trial.

Update: Not necessary any more to sort images into subfolders. *py-files for model training and test select images from the BubblesProtocol-File.

## Analysis
### For 2023 Poster (May 2023)
- control and NSSI group were split into test and training set (i.e., test and training individuals) and image classifier was trained on emotion classification (happy vs neutral; sad vs neutral)
- py-file ./ResNet50_training_twoGroups_crossVal.py produces four models, res50_happyCorrectVShappyNeutralCorrect_2G_control.h5; res50_happyCorrectVShappyNeutralCorrect_2G_experimental.h5; res50_sadCorrectVSsadNeutralCorrect_2G_control.h5; res50_sadCorrectVSsadNeutralCorrect_2G_experimental.h5
- py-files ResNet50_predictionG2_crossvalidation.py uses the four models for predicition (validation). All Images that were not used for training are included. Writes results to './results/probs_for_G2_crossvalidation.csv'
- py-file ResNet50_prediction_sad__LRP_stats_perVP.py takes all images "sad" and "sad neutral" with correct responses, per subject, and computes the relevance. Files are saved as './results/Sxxx.ime.csv' for raw relevance,  './results/Sxxx.imze.csv' for z-scored relevance,  './results/Sxxx.imzp.csv' for relevance proportion.
- m-file ./persubject_LRPs.m reads csv-files, averages across participants and plots

ToDo:
- run model with z- transformed image: (correct - mean (incorrect)) / sd(incorrect)
- mask image face area?
- multi-level classifaction (class hierarchy)
- finetune classification, e. g. further hidden layer before fc-layer [OK, 2023-03-20 with 46 Ekman& Friesen Action units]
- LRP
- models with random 1/2 selection of groups members, then predict (cross-validate) other half of participants on trained models control vs experimental


## Results

![Fig1](https://user-images.githubusercontent.com/66525570/194876846-9c60f03f-7d21-496e-b93a-a052aeaa0589.png)

Figure 1. [example figure]

