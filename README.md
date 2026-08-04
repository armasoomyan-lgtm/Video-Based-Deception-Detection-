# vision-machine
Video-Based Deception Detection via Local-Global Facial Motion and Emotional Feature Fusion

This repository contains the implementation of the proposed hybrid deep learning framework for video-based deception detection presented in our research.
The proposed method combines local facial motion analysis, global facial dynamics, and facial emotion features to improve deception detection performance.
# Environment
The project was developed and tested using Google Colab. 
# Dataset
The experiments were conducted using the RLT Deception Detection Dataset:
Pérez-Rosas, V., Abouelenien, M., Mihalcea, R., & Burzo, M. (2015).
________________________________________
# Project Workflow
The implementation consists of the following stages:
1.	Extract facial regions from the 121 videos in the RLT dataset. 
2.	Generate 15-frame Groups of Pictures (GOPs). 
3.	Extract and save:

a.	Video frame sequences 
b.	Emotion labels 
c.	Micro-expression labels 
4.	Generate homography-based feature labels. 
5.	Build the proposed HME model. 
6.	Load training and testing data. 
7.	Perform model verification before training. 
8.	Configure the training parameters (epochs, batch size, training and validation sets). 
9.	Train the proposed HME model. 
10.	Save model checkpoints and training logs.

    
Figure 1 shows the training and validation loss curve for the proposed architecture.

________________________________________
# Python Codes files for the proposed architecture:

1.	Hybrid_ME_EmotionACC.py  (Training Hybrid Emodel Code:)
  •	Extract Face from 121 video dataset
  •	Extracting packking = 15 Of all_video_np,Emotion_label_np,micro_exp_labels
  •	Extracting homograph_label_H1_np , homograph_label_mesh_np
  •	defining  model 
  •	loading inputs && outputs
  •	Pre_Testing model
  •	define epochs ,batch_size ,train_in_data train_out_data val_in_data val_out_data
  •	Training HMEmodel
  •	Logging

2.	Test_Hybrid_ME_EmotionACC.py  (Testing & Evaluating Hybrid Emodel code:)
  The performance of the proposed hybrid model for deception detection was evaluated using the RLT dataset. We assessed the performance of the proposed hybrid model at two evaluation levels: (1) the GOP level, in which performance metrics are calculated using 15-frame GOPs, and (2) the video level, in which performance metrics are measured for complete test videos. 

The accuracy and AUC of the proposed hybrid architecture in GOP and video level:

  	Accuracy (%) --->	GOP=91.66	  Video=100

  	AUC          ---> GOP=0.963   Video=1.0

________________________________________
