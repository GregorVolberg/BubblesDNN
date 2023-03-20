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

## Analysis



## Results

![Fig1](https://user-images.githubusercontent.com/66525570/194876846-9c60f03f-7d21-496e-b93a-a052aeaa0589.png)

Figure 1. [example figure]

