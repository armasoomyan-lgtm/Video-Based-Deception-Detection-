### BY GOD 
##################################### pips
!pip install cvlib
!pip install tensorflow
!pip install keras
!pip install tf_keras
#!pip uninstall opencv-python 
#!pip install opencv-python

########################################## imports
import tf_keras
from PIL import Image
from keras.models import load_model 
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
print(tf.__version__)
print(tf.version.VERSION)
#####################################################################
import cv2 as cv
import cv2
import glob
#import cvlib as cv
import cvlib as cvlib
from google.colab.patches import cv2_imshow
import pandas as pd
from keras import Model
import keras #Need to be solved
from keras.layers import (Activation, Conv3D, Dense, Dropout, Flatten, 
                          MaxPooling3D, Lambda)
#from keras.layers.advanced_activations import LeakyReLU
from keras.layers import ELU, PReLU, LeakyReLU
from keras.losses import categorical_crossentropy
#from keras.layers.merge import concatenate
from keras.layers import concatenate
from keras.models import Sequential

from tensorflow.keras.optimizers import Adam, SGD, RMSprop
#from keras.optimizers import Adam
#from keras.utils import np_utils
#from keras.utils.vis_utils import plot_model

import os
import numpy as np
import scipy
from scipy import ndimage
import random
from random import shuffle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
######################################### 2
import keras
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense, Conv2D , Conv3D, MaxPool3D , Flatten, Activation , MaxPooling2D
#from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tqdm import tqdm
import matplotlib.pyplot as plt
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
#################################################
from models import H_estimator
from utils import DataLoader, load, save
import constant
import skimage
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import keras.backend as K

###################### mountinggggggggg gdrive  #########################
from google.colab import drive
drive.mount('/content/gdrive')
#################### import zipfile 
import zipfile
from zipfile import *
with zipfile.ZipFile("/content/gdrive/My Drive/Clips30+clipper.zip") as existing_zip:
    existing_zip.extractall("/content/")

# with zipfile.ZipFile("/content/gdrive/My Drive/Deceptive.zip") as existing_zip:
    # existing_zip.extractall("/content/")
# with zipfile.ZipFile("/content/gdrive/My Drive/Truthful.zip") as existing_zip:
    # existing_zip.extractall("/content/")

#!unzip -u "/content/gdrive/My Drive/Deceptive.zip"
#!unzip -u "/content/gdrive/My Drive/Truthful.zip"
#################### import zipfile
import zipfile
from zipfile import *
with zipfile.ZipFile("/content/gdrive/My Drive/model & check points 8-10-1403.zip") as existing_zip:
    existing_zip.extractall("/content/checkpoints/")
#################### import rarfile    
!pip install patool pyunpack
import pyunpack
if not os.path.exists("/content/training"):
    pyunpack.Archive("/content/gdrive/My Drive/training2.rar").extractall("/content/") 
############################# cerate directory 
if not os.path.exists("/content/summary"):
    os.mkdir("/content/summary")   
if not os.path.exists("/content/checkpoints"):
    os.mkdir("/content/checkpoints")
if not os.path.exists("/content/fusion"):
    os.mkdir("/content/fusion")  
if not os.path.exists("/content/imageshow"):
    os.mkdir("/content/imageshow")    

##########################################################################
!ls "/content/gdrive/MyDrive/"
!ls "/content/"
########################## parameters ##################################


#############################################################################
#############################################################################
###############################  part 1 ######################################
#############################################################################
##################  Extract Face from 121 video dataset  ####################

#####################   Defining Functions
#################################normalize_image denormalize 
def normalize_image(img):
    return np.array(img) / 255.0

def denormalize(img):
    #print('type(img)===',type(img))
    return (np.array(img) * 255).astype('uint8')
#################################
def read_labels(what):
    global dataset_path
    if what == 'all_video':
        path = dataset_path + '/all_video_labels.npy'
    elif what == 'train':
        path = dataset_path + '/train_labels.npy'
    else:
        path = dataset_path + '/test_labels.npy'
    return np.load(path, allow_pickle=True)

###  save video as np array by cv2 framework ......
###############################################################  Label Extraction
###  based on the videos names we assign a label, 0 for deceptive and 1 for truthful
def video_labeling(video_paths: list) -> list:
    video_label = []
    for video_path in video_paths:
        if 'lie' in video_path:
            video_label.append([0,1]) ## 0,1 == lie
        else:
            video_label.append([1,0]) ### 1,0 == truth
    return video_label
###############################################################  Label Extraction
###  based on the videos names we assign a label, 0 for deceptive and 1 for truthful
def all_video_labeling(video_paths: list) -> list:
    global all_video_labels
    video_label = []
    inner_names = [os.path.basename(video_path) for video_path in video_paths]
    print('inner_names == ',inner_names )
    for video_name in inner_names:
        if 'lie' in video_name:
            video_num = int(video_name[10:13]) - 1
            video_label.append(all_video_labels[video_num])
        else:
            video_num = int(video_name[12:15]) - 1
            video_label.append(all_video_labels[video_num+61])
    return video_label
#####################################################
#############################  Loading Videos  This functions loads the videos and returns them in a list
def read_videos(video_paths: list) -> list:
    videos = []
    videos_frames=[]
    for video_path in tqdm(video_paths):
      capp=cv2.VideoCapture(video_path)
      videos.append(capp)
      frameCount = int(capp.get(cv2.CAP_PROP_FRAME_COUNT))
      #frameCount1 = len(capp)
      videos_frames.append(frameCount-1) ## len video is ok 
    cv2.destroyAllWindows()
    print('videos_frames=',videos_frames)
    return videos,videos_frames
###################################################################  Converting RGB videos to gray-scale
##  this function takes a list of videos and returns a list of gray-scale video
def rgb2list2face(rgb_videos: list, resize_shape: tuple=(64, 64)) -> list:  #tuple=(112, 112) tuple=(256, 256))
    face_orgin__videos = []
    num_face_videos = []
    facecasc = cv2.CascadeClassifier(pathgd+'haarcascade_frontalface_default.xml') ## Extracting FACE
    
    ##for index,u in enumerate(num_face_videos) :
    ##for video_path in tqdm(video_paths):
    for index,video in enumerate(rgb_videos):
        print('video ====',video,index)
        # create a new video to act as the output gray video.
        
        # Read the first frame to find the original video attributes.
        ret, first_frame = video.read()
        #print('ret, first_frame',ret, first_frame.shape)
        cur_video_frames = []
        
        cropped_orgin_img = np.expand_dims(np.expand_dims(first_frame, -1), 0) ## the first run if can not detect face
        
        #???count00=0
        while(video.isOpened()):
            # Reading the next frame.
            ret, cur_frame = video.read()
                        
            #???count00=count00+1
            #????print('count00=======================',count00) 
            if not ret: 
                video.release()
                cv2.destroyAllWindows()
                break   # End of video or error
                
            #print('ret, cur_frame',ret, cur_frame.shape)
            # ########################## face detection 
                        
            #facecasc = cv2.CascadeClassifier(pathgd+'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
            orgin_img = cur_frame
             
            #print('gray ====',gray.shape)
            #print('orgin_img ====',orgin_img.shape)  ##gray ==== (480, 854) orgin_img ==== (480, 854, 3)
            
            #cv2_imshow(gray)
            ##???cv2_imshow(orgin_img)
            
            
            #faces = facecasc.detectMultiScale(orgin_img,scaleFactor=1.3, minNeighbors=3) ### gray == orgin_img  are ok
            faces = facecasc.detectMultiScale(gray,scaleFactor=1.08, minNeighbors=15) ### gray == orgin_img  are ok
            #faces = facecasc.detectMultiScale(gray,scaleFactor=1.03, minNeighbors=15) ### for debug gray == orgin_img  are ok
            ## scaleFactor from 1.01 to 1.5. For face detection, 1.1 to 1.3 are often good starting points.
            ## minNeighbors from 3 to 6. A value of 5 is a very common and often effective choice for face detection, striking a good balance between recall and precision.
            
            # face0 = faces                   ## for debug 
            # if len(faces) != 0:              ## for debug 
                # face0 =[faces[0]]
            # else :
            # #    face0 =[]
                 # face0 =[(1,2,3,4)]    
            
            # if count00<283:
              # x, y, w, h = 260 , 70 , 120 , 120 #???? x, y, w, h = 190 , 20 , 300 , 300 ## asigning manual position
              # face0 =[(x, y, w, h)]
            # elif (383<count00<835): 
              # x, y, w, h = 330 , 30 , 120 , 120 #???? x, y, w, h = 190 , 20 , 300 , 300 ## asigning manual position
              # face0 =[(x, y, w, h)]
            # else:
              # face0 =[]             
                        
            ##???print('faces ====',faces)   ## for debug 
            #face0 = faces                   ## for debug 
            if len(faces) != 0:              ## for debug 
                face0 =[faces[0]]
            else :
                face0 =[]
                
            
            #for (x, y, w, h) in faces:    ## poltic for lessing not facecs
            for (x, y, w, h) in face0:
                #????print('x, y, w, h ====',x, y, w, h) ## for debug
                #???? x, y, w, h = 190 , 20 , 300 , 300 ## asigning manual position
                if x == x :
                #if (200<x<270)&(50<y<90) : #if (360<x<515)& (100<h<195) : #if(x<70) : #if (100<y<250) & (270<x<580)  :  ##  if(x>100) & (h>105):   y<100  (y>30) & (y<60)  if(h>100)
                    cv2.rectangle(cur_frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
                    roi_gray = gray[y:y + h, x:x + w]
                    roi_orgin = orgin_img[y:y + h, x:x + w,:]
                
                    cropped_img = np.expand_dims(np.expand_dims(roi_gray, -1), 0)
                    #print('cropped_img ====',cropped_img.shape) ## cropped_img ==== (1, 121, 121, 1)

                    cropped_orgin_img = np.expand_dims(np.expand_dims(roi_orgin, -1), 0)
                    #print('cropped_orgin_img ====',cropped_orgin_img.shape) ## cropped_orgin_img ==== (1, 117, 117, 3, 1)                

                    cv2.putText(cur_frame, "FACE DETECT", (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    #cv2_imshow(cv2.resize(cur_frame,(128,64),interpolation = cv2.INTER_CUBIC))
                
                    #cv2_imshow(cv2.resize(cropped_img[0],(512,256),interpolation = cv2.INTER_CUBIC))
                    face_pic = cropped_orgin_img[0,:,:,:,0]

                    ######### resizing the frame to (60 * 40)
                    face_pic = cv2.resize(face_pic, resize_shape,interpolation = cv2.INTER_CUBIC)
                    ###???cv2_imshow(face_pic)  ## for debug 
                
                
                    # user_response = input("save the changes? (y/n):")
                    # if user_response.lower() == 'y':
                    # cur_video_frames.append(face_pic)  ##VERY DANGER .tolist()
                
                    cur_video_frames.append(face_pic)  ##VERY DANGER .tolist()
                    #print('face_pic ====',face_pic.shape)  ## face_pic ==== (64, 64, 3) 
                
                    
            
            #cur_frame = cv2.resize(cur_frame, resize_shape,interpolation = cv2.INTER_CUBIC)
            #cv2_imshow(cur_frame)
           
        ##### append the converted video to the list of videos to be returned.
        face_orgin__videos.append(cur_video_frames)
        num_face_videos.append(len(cur_video_frames)) ## number faces that recognized
        ####### closing the opened video. delete all opened frames.
        video.release()
        cv2.destroyAllWindows()
        

    return face_orgin__videos,num_face_videos

#######################################################################
#################################  dataset_path train_path test_path
current_dir = os.getcwd()
print('current_dir====',current_dir)
dataset_path ='/content/gdrive/My Drive'
pathgd="/content/gdrive/My Drive/"

#train_path = '/content/gdrive/My Drive'
#test_path = '/content/gdrive/My Drive'

# train_labels = read_labels('train')
# print('train_labels.shape',train_labels.shape,type(train_labels.shape),len(train_labels.shape))

all_video_labels =read_labels('all_video')
print('all_video_labels.shape',all_video_labels.shape,type(all_video_labels.shape),len(all_video_labels.shape))

# Deceptive and truthful folder's pathes
deceptive_path = current_dir + "/Clips30+clipper/Deceptive/train/"
truthful_path  = current_dir + "/Clips30+clipper/Truthful/train/"

# deceptive_path = current_dir + "/Clips30+clipper/Deceptive/val/"
# truthful_path  = current_dir + "/Clips30+clipper/Truthful/val/"

# deceptive_path = current_dir + "/Clips30+clipper/Deceptive/test/"
# truthful_path  = current_dir + "/Clips30+clipper/Truthful/test/"

print(current_dir,deceptive_path,truthful_path)

#################################  Loading GAN model...  ##################################################################
# print("Loading ganmodel...")
# ganmodel =  tf_keras.models.load_model("/content/gdrive/My Drive/r_p2p_gen_t2_2*10.model")
# #ganmodel = tf.keras.models.load_model("/content/gdrive/My Drive/r_p2p_gen_t2_2*10.model", compile=False)  # Load model with compile=False if custom objects were used
# print("Loaded ganmodel")
# print("ganmodel Summary ==== ::: ")
# ganmodel.summary()
#################################  Loading Emotion model...  ################

#EEmodel =  tf_keras.models.load_model(pathgd+'emotion model.h5')
# Create the Emodel
input_1_0 =  keras.layers.Input(shape=(64,64,3))  ## input should be Alone  and uint8

def normalize2(x):
    return (tf.cast(x, tf.float32) / 255.0) - 0.0
input_1 = keras.layers.Lambda(normalize2)(input_1_0)
output_1 = input_1
#output_1 = keras.layers.Conv2D(64, (3, 3), activation='relu')(output_1)
output_1 = keras.layers.Conv2D(64, (3, 3), padding='valid')(output_1)
output_1 = keras.layers.BatchNormalization()(output_1)  ## ++++
output_1 = keras.activations.relu(output_1)  #++++
print('1111111 output_1.shape ',output_1.shape)
#output_1 = keras.layers.Conv2D(64, (3, 3), activation='relu')(output_1)
output_1 = keras.layers.Conv2D(64, (3, 3), padding='valid')(output_1)
output_1 = keras.layers.BatchNormalization()(output_1)  ## ++++
output_1 = keras.activations.relu(output_1)  #+++
print('1111111 output_1.shape ',output_1.shape)
output_1 = keras.layers.MaxPool2D(pool_size=(2, 2))(output_1)
print('1111111 output_1.shape ',output_1.shape)
output_1 = keras.layers.Dropout(0.25)(output_1)

#output_1 = keras.layers.Conv2D(128, (3, 3), activation='relu')(output_1)
output_1 = keras.layers.Conv2D(128, (3, 3), padding='valid')(output_1)
output_1 = keras.layers.BatchNormalization()(output_1)  ## ++++
output_1 = keras.activations.relu(output_1)  #+++
print('1111111 output_1.shape ',output_1.shape)
#output_1 = keras.layers.Conv2D(128, (3, 3), activation='relu')(output_1)
output_1 = keras.layers.Conv2D(128, (3, 3), padding='valid')(output_1)
output_1 = keras.layers.BatchNormalization()(output_1)  ## ++++
output_1 = keras.activations.relu(output_1)  #+++
print('1111111 output_1.shape ',output_1.shape)
output_1 = keras.layers.MaxPool2D(pool_size=(2, 2))(output_1)
print('1111111 output_1.shape ',output_1.shape)
output_1 = keras.layers.Dropout(0.25)(output_1)

#output_1 = keras.layers.Conv2D(256, (3, 3), activation='relu')(output_1)
output_1 = keras.layers.Conv2D(256, (3, 3), padding='valid')(output_1)
output_1 = keras.layers.BatchNormalization()(output_1)  ## ++++
output_1 = keras.activations.relu(output_1)  #+++
print('1111111 output_1.shape ',output_1.shape)
#output_1 = keras.layers.Conv2D(256, (3, 3), activation='relu')(output_1)
output_1 = keras.layers.Conv2D(256, (3, 3), padding='valid')(output_1)
output_1 = keras.layers.BatchNormalization()(output_1)  ## ++++
output_1 = keras.activations.relu(output_1)  #+++
print('1111111 output_1.shape ',output_1.shape)
output_1 = keras.layers.MaxPool2D(pool_size=(2, 2))(output_1)
print('1111111 output_1.shape ',output_1.shape)
output_1 = keras.layers.Dropout(0.4)(output_1)

output_1 = keras.layers.Flatten()(output_1)

#output_1 = keras.layers.Dense(1025, activation='relu')(output_1) ##
output_1 = keras.layers.Dense(1025)(output_1)  
output_1 = keras.layers.BatchNormalization()(output_1)  ## ++++
output_1 = keras.activations.relu(output_1)  ## ++++
output_1 = keras.layers.Dropout(0.5)(output_1) 
#output_1 = keras.layers.Dense(7, activation='softmax')(output_1) ##
output_1 = keras.layers.Dense(7)(output_1)  
output_1 = keras.layers.BatchNormalization()(output_1)  ## ++++
output_1 = keras.activations.softmax(output_1)  ## ++++
 
print('1111111 output_1.shape ',output_1.shape)

Emodel = keras.models.Model([input_1_0],[output_1])

Emodel.summary()
print('len(Emodel.layers)===',len(Emodel.layers))

#Emodel.load_weights(pathgd+'emotion model1.weights.h5')
Emodel.load_weights(pathgd+'Emodel_64_64_RGB_BN_Epoach_09_val_loss_0.9781.weights.h5')

# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

# dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

############# test Emodel
f00=cv2.imread("/content/gdrive/My Drive/train/happy/im7.png")
cv2_imshow(f00)
#gray_pic = cv2.cvtColor(f00, cv2.COLOR_BGR2GRAY)
#expand_dims_img = np.expand_dims(np.expand_dims(cv2.resize(gray_pic, (64, 64)), -1), 0)
expand_dims_img = np.expand_dims(cv2.resize(f00, (64, 64)), 0) ##  shape = 1,64,64,3
##### Convert the grayscale image to RGB
#rgb_image = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2RGB)
    
print('expand_dims_img ====',expand_dims_img.shape) ## expand_dims_img ==== (1, 64, 64, 3)    
cv2_imshow(expand_dims_img[0])
prediction = Emodel.predict(expand_dims_img)## !!! DELATE Emodel.predict IN model shuld divide 255 /255.0
print('prediction.shape ,prediction ========== ',prediction.shape,prediction) ## (1, 7) [[0. 0. 1. 0. 0. 0. 0.]]  
maxindex = int(np.argmax(prediction))
cv2.putText(f00, emotion_dict[maxindex], (5, 5), cv2.FONT_HERSHEY_SIMPLEX, .25, (255, 0, 0), 1, cv2.LINE_AA)
cv2_imshow(cv2.resize(f00,(128,128),interpolation = cv2.INTER_CUBIC))

#######################################################################
######################### Loading the videos names with their extension from the drive.
#total_videos_batch =sorted(glob.glob(deceptive_path + '*[0][0-6][0-9].mp4') + glob.glob(truthful_path + '*[0][0-6][0-9].mp4'))  ## 121 video loaded
#total_videos_batch =sorted(glob.glob(deceptive_path + '*[0][0-0][0-1].mp4') + glob.glob(truthful_path + '*[0][0-0][0-1].mp4'))   ## ONLY 2 VIDEO LOADED
#total_videos_batch =sorted(glob.glob(truthful_path + '*[0][2-2][0-0].mp4'))
# total_videos_batch =sorted(glob.glob(deceptive_path + '*[0][0-6][0-9].mp4')
                        # + glob.glob(truthful_path + '*[0][0-6][0-9].mp4')) 
      
###################  For detecting BAD video does not Faces ###########################
# total_videos_batch = sorted(
                        # #+ glob.glob(deceptive_path + '*[0][2][4].mp4')
                        # #+ glob.glob(deceptive_path + '*[0][2][5].mp4')
                        # #+ glob.glob(deceptive_path + '*[0][2][6].mp4')
                        # #glob.glob(deceptive_path + '*[0][3][6].mp4')
                        # #+ glob.glob(deceptive_path + '*[0][4][1].mp4')
                        # #+ glob.glob(deceptive_path + '*[0][4][4].mp4')
                        # #+ glob.glob(deceptive_path + '*[0][5][0].mp4')
                        # #+ glob.glob(deceptive_path + '*[0][5][1].mp4')
                        # #+ glob.glob(deceptive_path + '*[0][5][3].mp4')
                        # #+ glob.glob(deceptive_path + '*[0][6][0].mp4')
                        # #+ glob.glob(deceptive_path + '*[0][6][1].mp4')
                        
                        # #+ glob.glob(truthful_path + '*[0][0][1].mp4')
                        # glob.glob(truthful_path + '*[0][1][6].mp4')
                        # glob.glob(truthful_path + '*[0][1][7].mp4')
                        # glob.glob(truthful_path + '*[0][2][3].mp4')
                        # glob.glob(truthful_path + '*[0][2][4].mp4')
                        # glob.glob(truthful_path + '*[0][2][5].mp4')
                        # #+ glob.glob(truthful_path + '*[0][4][5].mp4')
                        # + glob.glob(truthful_path + '*[0][4][9].mp4')
                        # + glob.glob(truthful_path + '*[0][5][0].mp4')
                        # #+ glob.glob(truthful_path + '*[0][5][2].mp4')
                        # ) 
#############################################################################################
# # Define the list of videos to exclude
# exclude_videos = [
    # deceptive_path + 'trial_lie_036.mp4',
    # deceptive_path + 'trial_lie_050.mp4',
    # deceptive_path + 'trial_lie_051.mp4',
    # deceptive_path + 'trial_lie_053.mp4',
    # deceptive_path + 'trial_lie_060.mp4',
    # deceptive_path + 'trial_lie_061.mp4',
    
    # truthful_path + 'trial_truth_049.mp4'
# ]
# # Get all videos and filter out the excluded ones
# al_videos = glob.glob(deceptive_path + '*[0][0-6][0-9].mp4') + glob.glob(truthful_path + '*[0][0-6][0-9].mp4')
# total_videos_batch = sorted([video for video in al_videos if video not in exclude_videos])
###################################################################################################

total_videos_batch =sorted(glob.glob(deceptive_path + '*[0][0-6][0-9].mp4')
                        + glob.glob(truthful_path + '*[0][0-6][0-9].mp4'))
 
#total_videos_batch =sorted(glob.glob(truthful_path + '*[0][5][3].mp4'))
#total_videos_batch =sorted(glob.glob(deceptive_path + '*[0][5][3].mp4'))
 
kk=len(total_videos_batch)
print("total_videos_batch:",total_videos_batch)
print('kk==len(total_videos_batch)==',len(total_videos_batch))

# Getting the label 0 1  , micro_exps column for the read videos.
video_label = video_labeling(total_videos_batch)
print("video_label:",video_label)

all_video_labels_micro_exps=all_video_labeling(total_videos_batch)
print("all_video_labels_micro_exps:",len(all_video_labels_micro_exps))

# Loading the actual vidiosret, first_frame = video.read()ret, first_frame = video.reret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()reret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()reret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()reret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()reret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()reret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()reret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()reret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()reret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()reret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()reret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()reret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()reret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()ret, first_frame = video.read()
input_videos3,num_frames = read_videos(total_videos_batch)
print('videos_frames======',len(input_videos3),num_frames)

#inner_names ==  ['trial_lie_001.mp4', 'trial_lie_002.mp4', 'trial_lie_003.mp4', 'trial_lie_004.mp4', 'trial_lie_005.mp4', 'trial_lie_007.mp4', 'trial_lie_008.mp4', 'trial_lie_009.mp4', 'trial_lie_010.mp4', 'trial_lie_011.mp4', 'trial_lie_013.mp4', 'trial_lie_014.mp4', 'trial_lie_015.mp4', 'trial_lie_016.mp4', 'trial_lie_017.mp4', 'trial_lie_018.mp4', 'trial_lie_019.mp4', 'trial_lie_020.mp4', 'trial_lie_021.mp4', 'trial_lie_025.mp4', 'trial_lie_026.mp4', 'trial_lie_027.mp4', 'trial_lie_028.mp4', 'trial_lie_030.mp4', 'trial_lie_031.mp4', 'trial_lie_032.mp4', 'trial_lie_033.mp4', 'trial_lie_034.mp4', 'trial_lie_035.mp4', 'trial_lie_036.mp4', 'trial_lie_037.mp4', 'trial_lie_038.mp4', 'trial_lie_039.mp4', 'trial_lie_040.mp4', 'trial_lie_041.mp4', 'trial_lie_042.mp4', 'trial_lie_043.mp4', 'trial_lie_044.mp4', 'trial_lie_046.mp4', 'trial_lie_047.mp4', 'trial_lie_048.mp4', 'trial_lie_049.mp4', 'trial_lie_051.mp4', 'trial_lie_052.mp4', 'trial_lie_053.mp4', 'trial_lie_054.mp4', 'trial_lie_055.mp4', 'trial_lie_056.mp4', 'trial_lie_057.mp4', 'trial_lie_058.mp4', 'trial_lie_059.mp4', 'trial_lie_060.mp4', 'trial_lie_061.mp4', 'trial_truth_001.mp4', 'trial_truth_002.mp4', 'trial_truth_003.mp4', 'trial_truth_004.mp4', 'trial_truth_005.mp4', 'trial_truth_006.mp4', 'trial_truth_007.mp4', 'trial_truth_008.mp4', 'trial_truth_009.mp4', 'trial_truth_010.mp4', 'trial_truth_011.mp4', 'trial_truth_012.mp4', 'trial_truth_013.mp4', 'trial_truth_014.mp4', 'trial_truth_015.mp4', 'trial_truth_016.mp4', 'trial_truth_017.mp4', 'trial_truth_018.mp4', 'trial_truth_020.mp4', 'trial_truth_021.mp4', 'trial_truth_023.mp4', 'trial_truth_025.mp4', 'trial_truth_026.mp4', 'trial_truth_027.mp4', 'trial_truth_028.mp4', 'trial_truth_030.mp4', 'trial_truth_031.mp4', 'trial_truth_032.mp4', 'trial_truth_033.mp4', 'trial_truth_034.mp4', 'trial_truth_035.mp4', 'trial_truth_036.mp4', 'trial_truth_037.mp4', 'trial_truth_038.mp4', 'trial_truth_039.mp4', 'trial_truth_040.mp4', 'trial_truth_041.mp4', 'trial_truth_042.mp4', 'trial_truth_043.mp4', 'trial_truth_044.mp4', 'trial_truth_045.mp4', 'trial_truth_046.mp4', 'trial_truth_047.mp4', 'trial_truth_048.mp4', 'trial_truth_049.mp4', 'trial_truth_050.mp4', 'trial_truth_051.mp4', 'trial_truth_052.mp4', 'trial_truth_053.mp4', 'trial_truth_054.mp4', 'trial_truth_055.mp4', 'trial_truth_056.mp4', 'trial_truth_057.mp4', 'trial_truth_058.mp4', 'trial_truth_059.mp4', 'trial_truth_060.mp4']
#all_video_labels_micro_exps: 109
#videos_frames====== 109 [509, 1873, 211, 348, 1601, 1404, 218, 618, 789, 1009, 570, 449, 1051, 1126, 998, 1043, 1111, 362, 524, 902, 866, 825, 754, 1132, 800, 698, 1124, 749, 1043, 1168, 701, 620, 809, 647, 712, 719, 317, 224, 989, 479, 1438, 689, 209, 1015, 496, 411, 959, 733, 569, 659, 1198, 413, 869, 426, 617, 382, 2445, 1044, 831, 2056, 1077, 651, 2187, 1277, 855, 881, 408, 1108, 187, 112, 173, 172, 327, 699, 1064, 958, 746, 449, 1374, 449, 1109, 424, 774, 749, 924, 548, 473, 799, 773, 1013, 893, 653, 623, 983, 1103, 773, 773, 893, 683, 868, 863, 809, 719, 869, 1018, 1138, 629, 809, 539]
# videos_frames====== 5 [236, 1266, 640, 324, 574]
# videos_frames====== 7 [545, 1195, 761, 569, 749, 854, 722]

#####################################  JUMPING  JUMPING JUMPING Somtimes from here  #############################################
##################################### when making face videes and saving in _input_videos2_pkl_30clip #########
# converting the original RGB input videos to face detect  & resizing 112*112*3.
#input_videos2,num_face_videos = rgb2list2face(input_videos3[0:5])  #  chenge dimention= 121*n*64*64*3 , face detection
input_videos2,num_face_videos = rgb2list2face(input_videos3[:])  #  chenge dimention= 121*n*64*64*3 , face detection

print('videos_frames======',len(input_videos3),num_frames)
print('num_face_videos===',len(input_videos2),num_face_videos)

# video ==== < cv2.VideoCapture 0x7fae4c31d4b0> 108
## train videos_frames====== 109 [509, 1873, 211, 348, 1601, 1404, 218, 618, 789, 1009, 570, 449, 1051, 1126, 998, 1043, 1111, 362, 524, 902, 866, 825, 754, 1132, 800, 698, 1124, 749, 1043, 1168, 701, 620, 809, 647, 712, 719, 317, 224, 989, 479, 1438, 689, 209, 1015, 496, 411, 959, 733, 569, 659, 1198, 413, 869, 426, 617, 382, 2445, 1044, 831, 2056, 1077, 651, 2187, 1277, 855, 881, 408, 1108, 187, 112, 173, 172, 327, 699, 1064, 958, 746, 449, 1374, 449, 1109, 424, 774, 749, 924, 548, 473, 799, 773, 1013, 893, 653, 623, 983, 1103, 773, 773, 893, 683, 868, 863, 809, 719, 869, 1018, 1138, 629, 809, 539]
## val videos_frames====== 5 [236, 1266, 640, 324, 574]
##  val faces num_face_videos=== 5 [236, 929, 589, 324, 295]
## test  videos_frames====== 7 [545, 1195, 761, 569, 749, 854, 722]
# test num_face_videos=== 7 [475, 894, 475, 381, 35, 841, 516]


######################### saving & reading input_videos2 by pickle  + num_face_videos
import pickle
filename = "/content/gdrive/My Drive/_input_videos2_pkl_30clip.pkl"
with open(filename, 'wb') as f: # 'wb' for write binary
    pickle.dump(input_videos2, f)
###########################################################################################
##########################################################################################
################################## Upgrate faces in input_videos2  #####################################
############################################################################################
import pickle
filename = "/content/gdrive/My Drive/_input_videos2_pkl_30clip.pkl"
input_videos2_ver2 = []
with open(filename, 'rb') as f: # 'rb' for read binary
    input_videos2_ver2 = pickle.load(f)

num_face_videos = [len(vid) for vid in input_videos2_ver2]
print('num_face_videos train===',len(input_videos2_ver2),num_face_videos) #num_face_videos train=== 109 [465, 1870, 182, 348, 1343, 1403, 218, 618, 789, 1009, 569, 360, 294, 736, 854, 461, 1004, 175, 326, 526, 520, 717, 646, 850, 689, 698, 464, 718, 1036, 2, 681, 609, 765, 647, 10, 637, 311, 154, 989, 479, 1379, 689, 58, 989, 0, 403, 959, 578, 498, 517, 1123, 0, 144, 139, 586, 382, 2445, 1038, 831, 2054, 1077, 650, 2118, 1224, 850, 833, 379, 1066, 130, 64, 173, 172, 223, 0, 218, 630, 733, 309, 1374, 425, 1109, 424, 774, 749, 924, 525, 469, 786, 750, 111, 677, 648, 623, 19, 1091, 773, 617, 0, 8, 867, 13, 337, 719, 869, 1018, 1137, 629, 809, 513]
total_sum = sum(num_face_videos)
print("The sum of the elements in the list is:", total_sum) ## elements in the list is: 76198 == Excel

print(len(input_videos2_ver2[100]))
print(len(input_videos2[0]))
input_videos2_ver2[100]= input_videos2[0]
print(len(input_videos2_ver2[100]))

#  ?
#  ?
#  ?
import pickle
filename = "/content/gdrive/My Drive/_input_videos2_pkl_30clip.pkl"
with open(filename, 'wb') as f: # 'wb' for write binary
    pickle.dump(input_videos2_ver2, f)

############################################################################################
####################################### PART 2 #############################################
############################################################################################
##### Extracting packking = 15 Of all_video_np,Emotion_label_np,micro_exp_labels,..  #######
############################################################################################
import pickle
filename = "/content/gdrive/My Drive/train_input_videos2_pkl_30clip.pkl"
#filename = "/content/gdrive/My Drive/val_input_videos2_pkl_30clip.pkl"
#filename = "/content/gdrive/My Drive/test_input_videos2_pkl_30clip.pkl"
input_videos2_ver2 = []
with open(filename, 'rb') as f: # 'rb' for read binary
    input_videos2_ver2 = pickle.load(f)

num_face_videos = [len(vid) for vid in input_videos2_ver2]
print('num_face_videos train ===',len(input_videos2_ver2),num_face_videos)

# num_face_videos train === 109 [465, 1870, 182, 348, 1554, 1403, 218, 618, 789, 1009, 570, 413, 823, 1073, 996, 1034, 1099, 329, 524, 880, 815, 806, 724, 1119, 776, 698, 698, 718, 1028, 1064, 681, 609, 765, 647, 618, 627, 255, 226, 989, 479, 1379, 689, 28, 989, 0, 403, 640, 514, 498, 659, 1187, 56, 733, 396, 586, 382, 2445, 1038, 831, 2054, 1077, 650, 2133, 1224, 879, 836, 379, 1103, 153, 109, 171, 172, 327, 514, 751, 661, 733, 309, 1374, 425, 1109, 424, 774, 749, 924, 525, 469, 786, 750, 0, 625, 648, 623, 941, 1091, 773, 617, 181, 602, 867, 350, 764, 719, 869, 1018, 1137, 629, 809, 539]
# The sum of the elements in the list is: 80338total_sum = sum(num_face_videos)

# num_face_videos train === 5 [236, 1263, 589, 324, 568]
# The sum of the elements in the list is: 2980

# num_face_videos train === 7 [515, 1173, 717, 0, 0, 854, 575]
# The sum of the elements in the list is: 3834

total_sum = sum(num_face_videos)
print("The sum of the elements in the list is:", total_sum) ## elements in the list is: 76198 == Excel


############################################################################################
############################################################################################
##################################  START  ##########################################
################################### IN 3 STAGE 
####### 1--- TRAIN  DATA 2----VAL DATA  3----- TEST DATA 
############################################################################################

###########################################################################
pack=15  #20  ## 144

input_videos2 = input_videos2_ver2
video_label = video_label
all_video_labels_micro_exps = all_video_labels_micro_exps
num_face_videos = [len(vid) for vid in input_videos2]
##?????print(video_label,all_video_labels_micro_exps,num_face_videos)
## [465, 1910, 182, 348, 1349, 475, 1459, 218, 618, 1147, 1009, 238, 569, 407, 306, 819, 644, 522, 1149, 192, 357, 1137, 1078, 584, 635, 684, 829, 746, 595, 932, 812, 698, 464, 718, 476, 2, 681, 609, 765, 647, 8, 637, 337, 147, 0, 1033, 488, 139, 586, 382, 2451, 1036, 1146, 2449, 1200, 652, 2119, 1224, 850, 834, 379, 1066, 130, 64, 173, 324, 172, 176, 841, 638, 516, 218, 795, 418, 309, 296, 1374, 425, 924, 424, 774, 749, 924, 525, 469, 868, 750, 53, 232, 215, 207, 7, 366]

################# slicing 15 pack from videos
all_total_videos_batch=[]
all_video=[]
all_video_label_01=[]
all_video_label_micro_exp=[]
homograph_pic=[]
homograph_label_H1=[]
homograph_label_MESH=[]
homograph_pic0=[]
homograph_label_H10=[]
homograph_label_MESH0=[]
Emotion_label=[]
Emotion_label0=[]
################################################

print('NUM vedeos ==',len(input_videos2))  ## 12+16+93=121
#for index,u in enumerate(num_frames) :  ## conut no of  videos in input_videos2 , index 0,1,2,...
for index,u in enumerate(num_face_videos) :  ## conut no of  videos in input_videos2 , index 0,1,2,...
    print('u , index==',u , index)
    k1=int(u/pack)
    #batch_number = k1
    #print('batch_number== k1==u/pack== ',k1,batch_number)
    for k2 in range (k1):  ## count k2  [0  u/20=174/20=8] 0,1,2,3,4,5,6,7
        #print('k2==',k2)
        #print('all_total_videos_batch==',os.path.basename(total_videos_batch[index]).split('.')[0]+'_'+str(k2)+'.mp4')
        #new_mame=os.path.basename(total_videos_batch[index]).split('.')[0]+'_'+str(k2)+'.mp4'
        #all_total_videos_batch.append(new_mame)
        all_video.append(input_videos2[index][k2*pack:pack+pack*k2])
        all_video_label_01.append(video_label[index])
        all_video_label_micro_exp.append(all_video_labels_micro_exps[index])
        
        ##### Extrect homographic pic  , face Emotion  here .....
        homograph_pic0=[]  ## RESETINGGG... notice that homograph_pic0 should reset
        Emotion_label0=[]  ## RESETINGGG... notice that Emotion_label0 should reset
        for k3 in range (pack):  ## count k3 [0  to pack=15] 0,1,2,3, ... ,15 
            #frame1 = input_videos2[index][k3*5+k2*pack] ## select frame 0,5,10 from 20=pack
            #frame1 = np.array(frame1, dtype=np.float32)
            #frame1 = (frame1 / 127.5) - 1.0
            #frame1 = cv2.resize(frame1,(512,512),interpolation = cv2.INTER_CUBIC)  # image resize 112 112  to 512 512 
        
            #frame2 = input_videos2[index][5+k3*5+pack*k2]  ## select frame 5,10,15 from 20=pack
            #frame2 = np.array(frame2, dtype=np.float32)
            #frame2 = (frame2 / 127.5) - 1.0
            #frame2 = cv2.resize(frame2,(512,512),interpolation = cv2.INTER_CUBIC)   # image resize 112 112  to 512 512
        
            #concatenated_frames = np.concatenate([frame1, frame2], axis=2)
            #homograph_pic0.append(concatenated_frames)
            ##############
            
            frame3 = input_videos2[index][k3+k2*pack] ## For  Emotion  select frame 0,1,2,3, ... ,15  from 15=pack
            frame3 = np.array(frame3, dtype=np.float32)
            #print('frame3.shape ========== ',frame3.shape)  ##(112, 112, 3)
            #gray_pic = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)
            #print('gray_pic.shape ========== ',gray_pic.shape)   ##(112, 112)
            #expand_dims_img = np.expand_dims(np.expand_dims(cv2.resize(gray_pic, (64, 64)), -1), 0)  
            expand_dims_img = np.expand_dims(cv2.resize(frame3, (64, 64)), 0) ##  shape = 1,64,64,3
            ###??? print('expand_dims_img ====',expand_dims_img.shape) ## expand_dims_img ==== (1, 48, 48, 1)    
            ####??? cv2_imshow(expand_dims_img[0])
            prediction = Emodel(expand_dims_img)  ## !!! DELATE Emodel.predict IN model shuld divide 255 /255.0
            ###??? print('prediction.shape ,prediction ========== ',prediction.shape,prediction) ## (1, 7) [[0. 0. 1. 0. 0. 0. 0.]]
            maxindex = int(np.argmax(prediction))
            #print('maxindex , emotion_dict[maxindex] ========== ',maxindex ,emotion_dict[maxindex])  ## 2  happy
            cv2.putText(frame3, emotion_dict[maxindex], (20, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
            ###??? cv2_imshow(cv2.resize(frame3,(128,128),interpolation = cv2.INTER_CUBIC))
            
            Emotion_label0.append(prediction)
            ##############
 
        #homograph_pic.append(homograph_pic0)  ##homograph_pic.shape (b, 3, 64, 64, 6) <class 'tuple'> 5 ,
                                              ## normalize_image(frame2 / 127.5) - 1.0
        Emotion_label.append(Emotion_label0)  ##Emotion_label_np.shape (b, 3, 1, 7) <class 'tuple'> 4
        #print('k2==',k2)
       
#print('all_total_videos_batch ==',all_total_videos_batch)
print('len(all_video)== ',len(all_video))
print('all_video_label_01 == ',type(all_video_label_01),len(all_video_label_01))
print('all_video_label_micro_exp == ',type(all_video_label_micro_exp),len(all_video_label_micro_exp))
#print('homograph_pic == ',type(homograph_pic),len(homograph_pic))
print('Emotion_label == ',type(Emotion_label),len(Emotion_label))


########################  clear RAM 1 
input_videos2_ver2 = []
input_videos2 = []
video_label = []
all_video_labels_micro_exps = []
total_videos_batch = []
############################
# input_videos = np.asanyarray(all_video,dtype=object )

# print("videos", type(input_videos), len(input_videos),input_videos.shape)
# print("frames", type(input_videos[0]), len(input_videos[0]),input_videos[0].shape)
# print("rows", type(input_videos[0][0]), len(input_videos[0][0]),input_videos[0][0].shape)
# print("cols", type(input_videos[0][0][0]), len(input_videos[0][0][0]),input_videos[0][0][0].shape)
# print('one pixel== ',type(input_videos[0][0][0][0]),input_videos[0][0][0][0])

######## LIst to Array  Label and  pic packs
all_video_np =  np.asarray(all_video).astype(np.uint8)
print('all_video_np.shape',all_video_np.shape,type(all_video_np.shape),len(all_video_np.shape))  

micro_exp_labels = np.asarray(all_video_label_micro_exp).astype(np.float32) ### LIST TO ARRAY FLOAT32
print('micro_exp_labels.shape',micro_exp_labels.shape,type(micro_exp_labels.shape),len(micro_exp_labels.shape))

Emotion_label_np =  np.asarray(Emotion_label).astype(np.float32)
print('Emotion_label_np.shape',Emotion_label_np.shape,type(Emotion_label_np.shape),len(Emotion_label_np.shape))

all_video_label_01_np = np.asarray(all_video_label_01).astype(np.float32)
print('all_video_label_01_np.shape',all_video_label_01_np.shape,type(all_video_label_01_np.shape),len(all_video_label_01_np.shape))

# all_video_label_01 ==  <class 'list'> 5300
# all_video_label_micro_exp ==  <class 'list'> 5300
# Emotion_label ==  <class 'list'> 5300
# all_video_np.shape (5300, 15, 64, 64, 3) <class 'tuple'> 5
# micro_exp_labels.shape (5300, 107) <class 'tuple'> 2
# Emotion_label_np.shape (5300, 15, 1, 7) <class 'tuple'> 4
# all_video_label_01_np.shape (5300, 2) <class 'tuple'> 2

# len(all_video)==  196
# all_video_label_01 ==  <class 'list'> 196
# all_video_label_micro_exp ==  <class 'list'> 196
# Emotion_label ==  <class 'list'> 196
# all_video_np.shape (196, 15, 64, 64, 3) <class 'tuple'> 5
# micro_exp_labels.shape (196, 107) <class 'tuple'> 2
# Emotion_label_np.shape (196, 15, 1, 7) <class 'tuple'> 4
# all_video_label_01_np.shape (196, 2) <class 'tuple'> 

# len(all_video)==  253
# all_video_label_01 ==  <class 'list'> 253
# all_video_label_micro_exp ==  <class 'list'> 253
# Emotion_label ==  <class 'list'> 253
# all_video_np.shape (253, 15, 64, 64, 3) <class 'tuple'> 5
# micro_exp_labels.shape (253, 107) <class 'tuple'> 2
# Emotion_label_np.shape (253, 15, 1, 7) <class 'tuple'> 4
# all_video_label_01_np.shape (253, 2) <class 'tuple'> 2

###########################################################################

np.save('/content/gdrive/My Drive/_30clip_label_01_np.npy', all_video_label_01_np)
np.save('/content/gdrive/My Drive/_30clip_micro_exp_labels.npy', micro_exp_labels)

np.save('/content/gdrive/My Drive/_30clip_video_np.npy', all_video_np)
np.save('/content/gdrive/My Drive/_30clip_Emotion_label_np.npy', Emotion_label_np)
#np.save('/content/gdrive/My Drive/_homograph_pic_np.npy', homograph_pic)



#############################################################################
#############################################################################
###############################  part 3 ######################################
#############################################################################
### Extracting homograph_label_H1_np , homograph_label_mesh_np  #############

###################### Disable eager execution for homograph_label_

all_video_np_org = np.load('/content/gdrive/My Drive/train_30clip_video_np.npy') ##all_video_np.shape (3486, 20, 64, 64, 3)
#all_video_np_org = np.load('/content/gdrive/My Drive/val_30clip_video_np.npy')
#all_video_np_org = np.load('/content/gdrive/My Drive/test_30clip_video_np.npy')

all_video_np = all_video_np_org
####all_video_np.shape (5300, 15, 64, 64, 3) 3486+0=3486

# all_video_np = all_video_np_org[1000:]  
# print('all_video_np.shape',all_video_np.shape,len(all_video_np))

# all_video_np = all_video_np_org[1000+200:]  
# print('all_video_np.shape',all_video_np.shape,len(all_video_np))


# all_video_np = all_video_np_org[1000+200+300:]  
# print('all_video_np.shape',all_video_np.shape,len(all_video_np))

#all_video_np = all_video_np_org[1000+200+300+700+100+350+1200+250+350+850:]  
############################### 1000+200+300+700+100+350+1200+250+350+850==== 5300
print('all_video_np.shape',all_video_np.shape,len(all_video_np))



#all_video_np = all_video_np_org[(1450+850):]  
#print('all_video_np.shape',all_video_np.shape,len(all_video_np))
####all_video_np.shape (3002, 20, 64, 64, 3) 1239+1397+850=3486



homograph_label_H1=[]
homograph_label_MESH=[]

tf.compat.v1.disable_eager_execution()

os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = constant.GPU
test_folder = constant.TEST_FOLDER
snapshot_dir =  constant.SNAPSHOT_DIR + '/model & check points 8-10-1403/model.ckpt-500000'
batch_size = constant.TEST_BATCH_SIZE

height, width = 512, 512

###################  define dataset
with tf.name_scope('dataset'):
    ##########testing###############
    
    test_inputs_clips_tensor = tf.compat.v1.placeholder(shape=[batch_size, height, width, 3 * 2], dtype=tf.float32)
    test_inputs = test_inputs_clips_tensor
    print('test inputs = {}'.format(test_inputs))
    

##########  depth is not needed in the inference process, 
#we assign "test_depth" arbitrary values such as an all-one map
test_depth = tf.ones_like(test_inputs[...,0:1])
print("test_depth.shape")
print(test_depth.shape)
with tf.compat.v1.variable_scope('generator', reuse=tf.compat.v1.AUTO_REUSE):
    print('testing = {}'.format(tf.compat.v1.get_variable_scope().name))
    H1_motion, H2_motion, mesh_motion,test_warp2_depth, test_mesh, test_warp2_H1, test_warp2_H2, test_warp2_H3, test_one_warp_H1, test_one_warp_H2, test_one_warp_H3 = H_estimator(test_inputs, test_inputs, test_depth)
    
#####  create parametrs once by H_estimator

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True      
with tf.compat.v1.Session(config=config) as sess:
    # dataset
    data_loader = DataLoader(test_folder, height, width)
    print('data_loader shapeeeeee==={}'.format(data_loader))   ## <utils.DataLoader object at 0x7911c1ba19d0>

    # initialize weights
    sess.run(tf.compat.v1.global_variables_initializer())
    print('Init global successfully!')

    # tf saver
    saver = tf.compat.v1.train.Saver(var_list=tf.compat.v1.global_variables(), max_to_keep=None)

    restore_var = [v for v in tf.compat.v1.global_variables()]
    loader = tf.compat.v1.train.Saver(var_list=restore_var)

    def inference_func(ckpt):
        print("============")
        print("Checking for checkpoint file at:", ckpt)
        if not os.path.exists(ckpt + ".meta"):
            print("Checkpoint file not found! Please check the path and ensure it exists.")
            return  # or raise an exception
        
        
        try:
            print(ckpt)
            load(loader, sess, ckpt)
            print("============")
            # ... (rest of the inference code)
        
        except tf.errors.OutOfRangeError as e:
            print("Error loading checkpoint:", e)
            print("Possible causes: corrupted checkpoint file, model mismatch, or storage issues.")
            # Handle the error appropriately, e.g., exit the program or try a different checkpoint
        
        print(ckpt)
        load(loader, sess, ckpt)
        print("============")
        length = 1106
        psnr_list = []
        ssim_list = []


        #for i in range(0, length):
        ######  #homograph_pic.shape (b, 4, 64, 64, 6)
        #####     all_video_np=      (b ,20, 64, 64, 3)
        for i in range(0, len(all_video_np)):  ## 
            if i % 50 == 0:  # This is true if x is a multiple of 50 (e.g., 0, 50, 100, 150, ...)
                    print("Checkpoint loop ", i,'/',len(all_video_np))
                    homograph_label_H1_np =  np.asarray(homograph_label_H1).astype(np.float32)
                    homograph_label_MESH_np =  np.asarray(homograph_label_MESH).astype(np.float32)
                    np.save('/content/gdrive/My Drive/homograph_label_H1_np.npy', homograph_label_H1_np)
                    np.save('/content/gdrive/My Drive/homograph_label_MESH_np.npy', homograph_label_MESH_np)
                    print('homograph_label_H1_np.shape',homograph_label_H1_np.shape)
                    print('homograph_label_MESH_np.shape',homograph_label_MESH_np.shape)

                    
            homograph_label_H10=[]   ## RESETTING..... variable
            homograph_label_MESH0=[]  ## RESETTING..... variable
            for j in range(0, 6):  ## 4 ==  pack/5 == 0,1,2,3  == 4 duble frames
                
                if j==0 :
                    xx0 = all_video_np[i][2][:,:,:] ## select frame 2,6,10,14 from 20=pack  (64,64,3)
                    yy0 = all_video_np[i][6][:,:,:]
                elif j==1:
                    xx0 = all_video_np[i][2][:,:,:] ## select frame 2,6,10,14 from 20=pack  (64,64,3)
                    yy0 = all_video_np[i][10][:,:,:]
                elif j==2:
                    xx0 = all_video_np[i][2][:,:,:] ## select frame 2,6,10,14 from 20=pack  (64,64,3)
                    yy0 = all_video_np[i][14][:,:,:]
                elif j==3:
                    xx0 = all_video_np[i][6][:,:,:] ## select frame 2,6,10,14 from 20=pack  (64,64,3)
                    yy0 = all_video_np[i][10][:,:,:]
                elif j==4:
                    xx0 = all_video_np[i][6][:,:,:] ## select frame 2,6,10,14 from 20=pack  (64,64,3)
                    yy0 = all_video_np[i][14][:,:,:]
                elif j==5:
                    xx0 = all_video_np[i][10][:,:,:] ## select frame 2,6,10,14 from 20=pack  (64,64,3)
                    yy0 = all_video_np[i][14][:,:,:]
                    
                ################    view & test image ##########################
                #xx0 = all_video_np[i][2+j*4][:,:,:] ## select frame 2,6,10,14 from 20=pack  (64,64,3)
                #yy0 = all_video_np[i][2+4+j*4][:,:,:] ## select frame 6,10,14,18 from 20=pack (64,64,3)
                    
                #xx0 = homograph_pic[i][j][:,:,0:3]
                #yy0 = homograph_pic[i][j][:,:,3:6]
                
                xx0 = np.array(xx0, dtype=np.float32)
                xx0 = (xx0 / 127.5) - 1.0
                xx0 = cv2.resize(xx0,(512,512),interpolation = cv2.INTER_CUBIC)  # image resize 512, 512, 3 

                yy0 = np.array(yy0, dtype=np.float32)
                yy0 = (yy0 / 127.5) - 1.0
                yy0 = cv2.resize(yy0,(512,512),interpolation = cv2.INTER_CUBIC)   # image resize 512, 512, 3 
        
                concate_frames = np.concatenate([xx0, yy0], axis=2)   # image resize 512, 512, 6
                #input_clip = np.expand_dims(data_loader.get_data_clips(i), axis=0)
                input_clip = np.expand_dims(concate_frames, axis=0)
                ####??? print('input_clip==={}',input_clip.shape)  # input_clip==={} (1, 512, 512, 6)
                #####################   view & test image  ##########
                xx = (input_clip[0][:,:,0:3]+1)*127.5
                yy = (input_clip[0][:,:,3:6]+1)*127.5
                
                #path2 = "/content/imageshow/" + str(j).zfill(6)+ str(i).zfill(6) + "input_clip1" + ".jpg"
                #cv.imwrite(path2, xx)
                #path5 = "/content/imageshow/" + str(i).zfill(6)+ str(i).zfill(6) + "input_clip2" + ".jpg"
                #cv.imwrite(path5, yy)
            
                ###??? cv2_imshow(cv2.resize(xx,(64,64),interpolation = cv2.INTER_CUBIC))
                ####??? cv2_imshow(cv2.resize(yy,(64,64),interpolation = cv2.INTER_CUBIC))
            
                #Attention: both inputs and outpus are the types of numpy, that is :(preH, warp_gt) and (input_clip,h_clip)
                H1motion, H2motion, meshmotion,_, mesh, warp_H1, warp_H2, warp_H3, warp_one_H1, warp_one_H2, warp_one_H3 = sess.run([H1_motion, H2_motion, mesh_motion,test_warp2_depth, test_mesh, test_warp2_H1, test_warp2_H2, test_warp2_H3, test_one_warp_H1, test_one_warp_H2, test_one_warp_H3], 
                        feed_dict={test_inputs_clips_tensor: input_clip})
            
                #print('mesh, warp_H1, warp_H2, warp_H3, warp_one_H1, warp_one_H2, warp_one_H3 =====  ',mesh, warp_H1, warp_H2, warp_H3, warp_one_H1, warp_one_H2, warp_one_H3 )
                #print('mesh==={}'.format(mesh))
                #
                # mesh==={} (1, 9, 9, 2)
                # warp_H1==={} (1, 512, 512, 3)
                # warp_H2==={} (1, 512, 512, 3)
                #warp_H3==={} (1, 512, 512, 3)
                # warp_one_H1==={} (1, 512, 512, 3)
                # warp_one_H2==={} (1, 512, 512, 3)
                # warp_one_H3==={} (1, 512, 512, 3)
                
                # print('mesh==={}',mesh.shape)
                # print('warp_H1==={}',warp_H1.shape)
                # print('warp_H2==={}',warp_H2.shape)
                # print('warp_H3==={}',warp_H3.shape)
                # print('warp_one_H1==={}',warp_one_H1.shape)
                # print('warp_one_H2==={}',warp_one_H2.shape)
                # print('warp_one_H3==={}',warp_one_H3.shape)
            
                # print('H1motion==={}',H1motion.shape,H1motion)  ##(1, 8, 1)
                # print('H2motion==={}',H2motion.shape,H2motion)  ##(1, 8, 1)
                # print('meshmotion==={}',meshmotion.shape,meshmotion)  ##(1, 9, 9, 2)
            
            
                homograph_label_H10.append(H1motion[0,:,:])  ##  append (8,1) instand (1, 8, 1)
                homograph_label_MESH0.append(meshmotion[0,:,:,:]) ##  append (9, 9, 2) instand (1, 9, 9, 2)
                      
                # warp  = warp_H3
                final_warp = (warp_H3+1) * 127.5    
                final_warp = final_warp[0] 
                # warp_one  = warp_one_H3
                final_warp_one = warp_one_H3[0]
                # input1
                input1 = (input_clip[...,0:3]+1) * 127.5    
                input1 = input1[0]
            
                # calculate psnr/ssim
                #psnr = skimage.measure.compare_psnr(input1*final_warp_one, final_warp*final_warp_one, 255)
                #ssim = skimage.measure.compare_ssim(input1*final_warp_one, final_warp*final_warp_one, data_range=255, multichannel=True)
            
                psnr = peak_signal_noise_ratio(input1*final_warp_one, final_warp*final_warp_one, data_range=255)
                ssim = structural_similarity(input1*final_warp_one, final_warp*final_warp_one, data_range=255, multichannel=True, win_size=3) #win_size is set to 3

                # image fusion
                img1 = input1
                img2 = final_warp*final_warp_one
            
                fusion = np.zeros((512,512,3), np.uint8)
                fusion[...,0] = img2[...,0] 
                fusion[...,1] = img1[...,1]*0.5 +  img2[...,1]*0.5
                fusion[...,2] = img1[...,2]
            
                ################    view & test image ##########################
                #path3 = "/content/imageshow/" + str(j).zfill(6)+ str(i).zfill(6) + "img2" + ".jpg"
                #cv.imwrite(path3, img2)
                #path = "../fusion/" + str(i+1).zfill(6) + ".jpg"
                #path = "/content/fusion/" + str(j).zfill(6)+ str(i).zfill(6) + "fusion" + ".jpg"
                #cv.imwrite(path, fusion)
                ####??? cv2_imshow(cv2.resize(fusion,(64,64),interpolation = cv2.INTER_CUBIC))
            
                ####??? print('i = {}/ {} ,j = {}, psnr = {:.6f}'.format( i+1, len(all_video_np),j+1, psnr))
                ####??? print('i = {}/ {} ,j = {}, ssim = {:.6f}'.format( i+1, len(all_video_np),j+1, ssim))
            
                psnr_list.append(psnr)
                ssim_list.append(ssim)
        
            homograph_label_H1.append(homograph_label_H10)  ##  append (j,8,1) 
            homograph_label_MESH.append(homograph_label_MESH0) ##  append (j,9, 9, 2) instand
            

        
    inference_func(snapshot_dir)

##
homograph_label_H1_np =  np.asarray(homograph_label_H1).astype(np.float32)
print('homograph_label_H1_np.shape',homograph_label_H1_np.shape,type(homograph_label_H1_np.shape),len(homograph_label_H1_np.shape))

homograph_label_MESH_np =  np.asarray(homograph_label_MESH).astype(np.float32)
print('homograph_label_MESH_np.shape',homograph_label_MESH_np.shape,type(homograph_label_MESH_np.shape),len(homograph_label_MESH_np.shape))

#################  saving input  & outputs
np.save('/content/gdrive/My Drive/homograph_label_H1_np.npy', homograph_label_H1_np)
np.save('/content/gdrive/My Drive/homograph_label_MESH_np.npy', homograph_label_MESH_np)



#################################  Stage 2  ############################
##############################################################################
######################  concatenate  H1,MESH  for large dataset  ########################

homograph_label_H1_np9 = np.load('/content/gdrive/My Drive/homograph_label_H1_np_850_9.npy') 
homograph_label_MESH_np9 = np.load('/content/gdrive/My Drive/homograph_label_MESH_np_850_9.npy') 
print('homograph_label_H1_np9.shape',homograph_label_H1_np9.shape,len(homograph_label_H1_np9))
print('homograph_label_MESH_np9.shape',homograph_label_MESH_np9.shape,len(homograph_label_MESH_np9))


homograph_label_H1_np8 = np.load('/content/gdrive/My Drive/homograph_label_H1_np_350_8.npy') 
homograph_label_MESH_np8 = np.load('/content/gdrive/My Drive/homograph_label_MESH_np_350_8.npy') 
print('homograph_label_H1_np8.shape',homograph_label_H1_np8.shape,len(homograph_label_H1_np8))
print('homograph_label_MESH_np8.shape',homograph_label_MESH_np8.shape,len(homograph_label_MESH_np8))

homograph_label_H1_np7 = np.load('/content/gdrive/My Drive/homograph_label_H1_np_250_7.npy') 
homograph_label_MESH_np7 = np.load('/content/gdrive/My Drive/homograph_label_MESH_np_250_7.npy') 
print('homograph_label_H1_np7.shape',homograph_label_H1_np7.shape,len(homograph_label_H1_np7))
print('homograph_label_MESH_np7.shape',homograph_label_MESH_np7.shape,len(homograph_label_MESH_np7))

homograph_label_H1_np6 = np.load('/content/gdrive/My Drive/homograph_label_H1_np_1200_6.npy') 
homograph_label_MESH_np6 = np.load('/content/gdrive/My Drive/homograph_label_MESH_np_1200_6.npy') 
print('homograph_label_H1_np6.shape',homograph_label_H1_np6.shape,len(homograph_label_H1_np6))
print('homograph_label_MESH_np6.shape',homograph_label_MESH_np6.shape,len(homograph_label_MESH_np6))


homograph_label_H1_np562 = np.load('/content/gdrive/My Drive/homograph_label_H1_np562.npy') 
homograph_label_MESH_np562 = np.load('/content/gdrive/My Drive/homograph_label_MESH_np562.npy') 
print('homograph_label_H1_np562.shape',homograph_label_H1_np562.shape,len(homograph_label_H1_np562))
print('homograph_label_MESH_np562.shape',homograph_label_MESH_np562.shape,len(homograph_label_MESH_np562))

homograph_label_H1_np561 = np.load('/content/gdrive/My Drive/homograph_label_H1_np561.npy') 
homograph_label_MESH_np561 = np.load('/content/gdrive/My Drive/homograph_label_MESH_np561.npy') 
print('homograph_label_H1_np561.shape',homograph_label_H1_np561.shape,len(homograph_label_H1_np561))
print('homograph_label_MESH_np561.shape',homograph_label_MESH_np561.shape,len(homograph_label_MESH_np561))


homograph_label_H1_np5 = np.load('/content/gdrive/My Drive/homograph_label_H1_np_100_5.npy') 
homograph_label_MESH_np5 = np.load('/content/gdrive/My Drive/homograph_label_MESH_np_100_5.npy') 
print('homograph_label_H1_np5.shape',homograph_label_H1_np5.shape,len(homograph_label_H1_np5))
print('homograph_label_MESH_np5.shape',homograph_label_MESH_np5.shape,len(homograph_label_MESH_np5))

homograph_label_H1_np4 = np.load('/content/gdrive/My Drive/homograph_label_H1_np_700_4.npy') 
homograph_label_MESH_np4 = np.load('/content/gdrive/My Drive/homograph_label_MESH_np_700_4.npy') 
print('homograph_label_H1_np4.shape',homograph_label_H1_np4.shape,len(homograph_label_H1_np4))
print('homograph_label_MESH_np4.shape',homograph_label_MESH_np4.shape,len(homograph_label_MESH_np4))

homograph_label_H1_np3 = np.load('/content/gdrive/My Drive/homograph_label_H1_np_300_3.npy') 
homograph_label_MESH_np3 = np.load('/content/gdrive/My Drive/homograph_label_MESH_np_300_3.npy') 
print('homograph_label_H1_np3.shape',homograph_label_H1_np3.shape,len(homograph_label_H1_np3))
print('homograph_label_MESH_np3.shape',homograph_label_MESH_np3.shape,len(homograph_label_MESH_np3))

homograph_label_H1_np2 = np.load('/content/gdrive/My Drive/homograph_label_H1_np_200_2.npy') 
homograph_label_MESH_np2 = np.load('/content/gdrive/My Drive/homograph_label_MESH_np_200_2.npy') 
print('homograph_label_H1_np2.shape',homograph_label_H1_np2.shape,len(homograph_label_H1_np2))
print('homograph_label_MESH_np2.shape',homograph_label_MESH_np2.shape,len(homograph_label_MESH_np2))

homograph_label_H1_np1 = np.load('/content/gdrive/My Drive/homograph_label_H1_np_1000_1.npy') 
homograph_label_MESH_np1 = np.load('/content/gdrive/My Drive/homograph_label_MESH_np_1000_1.npy') 
print('homograph_label_H1_np1.shape',homograph_label_H1_np1.shape,len(homograph_label_H1_np1))
print('homograph_label_MESH_np1.shape',homograph_label_MESH_np1.shape,len(homograph_label_MESH_np1))

# homograph_label_H1_np9.shape (850, 6, 8, 1) 850
# homograph_label_MESH_np9.shape (850, 6, 9, 9, 2) 850
# homograph_label_H1_np8.shape (350, 6, 8, 1) 350
# homograph_label_MESH_np8.shape (350, 6, 9, 9, 2) 350
# homograph_label_H1_np7.shape (250, 6, 8, 1) 250
# homograph_label_MESH_np7.shape (250, 6, 9, 9, 2) 250
# homograph_label_H1_np6.shape (1200, 6, 8, 1) 1200
# homograph_label_MESH_np6.shape (1200, 6, 9, 9, 2) 1200
# homograph_label_H1_np562.shape (150, 6, 8, 1) 150
# homograph_label_MESH_np562.shape (150, 6, 9, 9, 2) 150
# homograph_label_H1_np561.shape (200, 6, 8, 1) 200
# homograph_label_MESH_np561.shape (200, 6, 9, 9, 2) 200
# homograph_label_H1_np5.shape (100, 6, 8, 1) 100
# homograph_label_MESH_np5.shape (100, 6, 9, 9, 2) 100
# homograph_label_H1_np4.shape (700, 6, 8, 1) 700
# homograph_label_MESH_np4.shape (700, 6, 9, 9, 2) 700
# homograph_label_H1_np3.shape (300, 6, 8, 1) 300
# homograph_label_MESH_np3.shape (300, 6, 9, 9, 2) 300
# homograph_label_H1_np2.shape (200, 6, 8, 1) 200
# homograph_label_MESH_np2.shape (200, 6, 9, 9, 2) 200
# homograph_label_H1_np1.shape (1000, 6, 8, 1) 1000
# homograph_label_MESH_np1.shape (1000, 6, 9, 9, 2) 1000
############################### 1000+200+300+700+100+350+1200+250+350+850==== 5300
####################


homograph_label_H1_np = keras.layers.Concatenate(axis=0)([homograph_label_H1_np1,
            homograph_label_H1_np2,homograph_label_H1_np3,homograph_label_H1_np4,
            homograph_label_H1_np5,homograph_label_H1_np561,homograph_label_H1_np562,
            homograph_label_H1_np6,homograph_label_H1_np7,homograph_label_H1_np8,homograph_label_H1_np9])
homograph_label_MESH_np = keras.layers.Concatenate(axis=0)([homograph_label_MESH_np1,
            homograph_label_MESH_np2,homograph_label_MESH_np3,homograph_label_MESH_np4,
            homograph_label_MESH_np5,homograph_label_MESH_np561,homograph_label_MESH_np562,
            homograph_label_MESH_np6,homograph_label_MESH_np7,homograph_label_MESH_np8,homograph_label_MESH_np9])
print('homograph_label_H1_np.shape',homograph_label_H1_np.shape,len(homograph_label_H1_np))
print('homograph_label_MESH_np.shape',homograph_label_MESH_np.shape,len(homograph_label_MESH_np)) 


           
np.save('/content/gdrive/My Drive/_30clip_homograph_label_H1_np.npy', homograph_label_H1_np)
np.save('/content/gdrive/My Drive/_30clip_homograph_label_MESH_np.npy', homograph_label_MESH_np) 

os._exit(0)  # restart

#### train homo mesh ,H1 ##################################
# homograph_label_H1_np.shape (5300, 6, 8, 1) 5300
# homograph_label_MESH_np.shape (5300, 6, 9, 9, 2) 5300

#### val homo mesh ,H1 ##################################

# homograph_label_H1_np.shape (393, 4, 8, 1) <class 'tuple'> 4
# homograph_label_MESH_np.shape (393, 4, 9, 9, 2) <class 'tuple'> 5

#### test homo mesh ,H1 ##################################
# homograph_label_H1_np.shape (253, 6, 8, 1) <class 'tuple'> 4
# homograph_label_MESH_np.shape (253, 6, 9, 9, 2) <class 'tuple'> 5


           

#############################################################################
#############################################################################
###############################  part 4 ######################################
#############################################################################
################### defining  model   ########################################
################### defin model
import os
#os.kill(os.getpid(), 9)
#os._exit(0)  # restart
#quit()   ### after run code crashed 
import os
import numpy as np
import keras
from keras import layers
from tensorflow import data as tf_data
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Embedding, Input, Flatten, Layer
from keras.layers import Dense, Conv2D , Conv3D, MaxPool3D , Flatten, Activation , MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate, Dense
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import tensorflow as tf
import random
from random import shuffle
from google.colab.patches import cv2_imshow
import cv2
from tensorflow.keras import backend as K
from tensorflow.keras import layers

from models import disjoint_augment_image_pair
from loss_functions import intensity_loss, depth_consistency_loss3
from utils import load, save, DataLoader
import skimage
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
###################################
from tensorflow.keras.layers import Conv2D
import tensorflow as tf
import tf_slim as slim
from tensorDLT import solve_DLT
from tf_spatial_transform import transform
from tensorflow.python.keras.layers import Conv2D
import tf_spatial_transform_local
import constant
import keras.backend as K

grid_w = constant.GRID_W
grid_h = constant.GRID_H

###################### mountinggggggggg gdrive  #########################
from google.colab import drive
drive.mount('/content/gdrive')
#################### import zipfile 
import zipfile
from zipfile import *
with zipfile.ZipFile("/content/gdrive/My Drive/Clips30+clipper.zip") as existing_zip:
    existing_zip.extractall("/content/")

# with zipfile.ZipFile("/content/gdrive/My Drive/Deceptive.zip") as existing_zip:
    # existing_zip.extractall("/content/")
# with zipfile.ZipFile("/content/gdrive/My Drive/Truthful.zip") as existing_zip:
    # existing_zip.extractall("/content/")

#################################################
max_val_MESH,min_val_MESH,max_val_H1,min_val_H1= (10,1,10,1)
################################################################################
pathgd="/content/gdrive/My Drive/"
print('eager mode is enabel??????????',tf.executing_eagerly())
###################################################################################
############################################################
# plots accuracy and loss curves
def plot_model_history(model_history):
    """
    Plot Accuracy and Loss curves given the model_history
    """
    fig, axs = plt.subplots(1,4,figsize=(30,7))

    # summarize history for accuracy Decept_Detect
    axs[0].plot(range(1,len(model_history.history['Decept_Detect_categorical_accuracy'])+1),model_history.history['Decept_Detect_categorical_accuracy'])
    axs[0].plot(range(1,len(model_history.history['val_Decept_Detect_categorical_accuracy'])+1),model_history.history['val_Decept_Detect_categorical_accuracy'])
    axs[0].set_title('Model Decept_Detect_categorical_accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    #axs[0].set_xticks(np.arange(1,len(model_history.history['accuracy'])+1),len(model_history.history['accuracy'])/10)
    axs[0].set_xticks(np.arange(1,len(model_history.history['Decept_Detect_categorical_accuracy'])+1))
    axs[0].legend(['train', 'val'], loc='best')

    # summarize history for Decept_Detect_loss
    axs[1].plot(range(1,len(model_history.history['Decept_Detect_loss'])+1),model_history.history['Decept_Detect_loss'])
    axs[1].plot(range(1,len(model_history.history['val_Decept_Detect_loss'])+1),model_history.history['val_Decept_Detect_loss'])
    axs[1].set_title('Model Decept_Detect_loss')
    axs[1].set_ylabel('Decept_Detect_loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['Decept_Detect_loss'])+1))
    axs[1].legend(['train', 'val'], loc='best') 

    # summarize history for M_EXP_loss
    axs[2].plot(range(1,len(model_history.history['M_EXP_loss'])+1),model_history.history['M_EXP_loss'])
    axs[2].plot(range(1,len(model_history.history['val_M_EXP_loss'])+1),model_history.history['val_M_EXP_loss'])
    axs[2].set_title('Model M_EXP_loss')
    axs[2].set_ylabel('M_EXP_loss')
    axs[2].set_xlabel('Epoch')
    axs[2].set_xticks(np.arange(1,len(model_history.history['M_EXP_loss'])+1))
    axs[2].legend(['train', 'val'], loc='best')
    
    # summarize history for loss
    axs[3].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[3].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[3].set_title('Model Loss')
    axs[3].set_ylabel('Loss')
    axs[3].set_xlabel('Epoch')
    #axs[3].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[3].set_xticks(np.arange(1,len(model_history.history['loss'])+1))
    axs[3].legend(['train', 'val'], loc='best')
       
    fig.savefig('plot.png')
    plt.show()
############################

def H_estimator(train_inputs_aug, train_inputs, train_depth):
    return H_model(train_inputs_aug, train_inputs, train_depth)
    


#Covert global homo into mesh
def H2Mesh(H2, patch_size):
    batch_size = tf.shape(H2)[0]
    h = patch_size/grid_h
    w = patch_size/grid_w
    ori_pt = []
    for i in range(grid_h + 1):
        for j in range(grid_w + 1):
            ww = j * w
            hh = i * h
            p = tf.constant([ww, hh, 1], shape=[3], dtype=tf.float32)
            ori_pt.append(tf.expand_dims(tf.expand_dims(p, 0),2))
    ori_pt = tf.concat(ori_pt, axis=2)
    ori_pt = tf.tile(ori_pt,[batch_size, 1, 1])
    tar_pt = tf.matmul(H2, ori_pt)
    
    #print('tar_pt , H2, ori_pt shape ==',tar_pt.shape, H2.shape,ori_pt.shape)
    ## = (4, 3, 81) (4, 3, 3) (4, 3, 81)
    x_s = tf.slice(tar_pt, [0, 0, 0], [-1, 1, -1])
    y_s = tf.slice(tar_pt, [0, 1, 0], [-1, 1, -1])
    z_s = tf.slice(tar_pt, [0, 2, 0], [-1, 1, -1])

    #print('x_s , y_s, z_s shape ==',x_s.shape, y_s.shape,z_s.shape)
    ## = (4, 1, 81) (4, 1, 81) (4, 1, 81)
    H2_local = tf.concat([x_s/z_s, y_s/z_s], axis=1)
    H2_local = tf.transpose(H2_local, perm=[0, 2, 1])
    H2_local = tf.reshape(H2_local, [batch_size, grid_h+1, grid_w+1, 2])
    print('H2_local shape ==',H2_local.shape)  ##  == (4, 9, 9, 2)  
    
    return H2_local


def H_model(train_inputs_aug, train_inputs, train_depth, patch_size=512.):

    train_inputs_aug = Input(shape=( 512, 512, 6),batch_size=3)  ## or (None,None, 3)
    train_inputs = Input(shape=( 512, 512, 6),batch_size=3)  ## or (None,None, 3)
    train_depth = Input(shape=( 512, 512, 1),batch_size=3)  ## or (None,None, 3)
    batch_size = train_inputs_aug.shape[0]
    
    ############################## build_model(train_inputs_aug) #, feature2_warp_gt, feature3_warp_gt
    input1 = train_inputs_aug[...,0:3]
    input2 = train_inputs_aug[...,3:6]
    #H1_motion, H2_motion, mesh_motion = build_model(train_inputs_aug) #, feature2_warp_gt, feature3_warp_gt
    _vgg_1 = _vgg(input1, input2)  
    [H1_motion, H2_motion, mesh_motion] = _vgg_1([input1, input2])
    
    print('H1_motion, H2_motion, mesh_motion ========',H1_motion.shape, H2_motion.shape, mesh_motion.shape)  
    ### ======== (1, 8, 1) (1, 8, 1) (1, 9, 9, 2)
    ############################################
    
    #scale transformation matrix
    M = np.array([[patch_size / 2.0, 0., patch_size / 2.0],
                  [0., patch_size / 2.0, patch_size / 2.0],
                  [0., 0., 1.]]).astype(np.float32)
    M_tensor = tf.constant(M, tf.float32)
    M_tile = tf.tile(tf.expand_dims(M_tensor, [0]), [batch_size, 1, 1])
    # Inverse of M
    M_inv = np.linalg.inv(M)
    M_tensor_inv = tf.constant(M_inv, tf.float32)
    M_tile_inv = tf.tile(tf.expand_dims(M_tensor_inv, [0]), [batch_size, 1, 1])
    
    ################################################### solve global homo H1/H2
    #H1 = solve_DLT(H1_motion, patch_size)
    solve_DLT_Layer_512_1 = solve_DLT_Layer_512()
    H1 = solve_DLT_Layer_512_1(H1_motion)  
    print ('H1.shape ==',H1.shape)## H1.shape ==  (64, 3, 3)
    
    #H2 = solve_DLT(H1_motion+H2_motion, patch_size)
    solve_DLT_Layer_512_2 = solve_DLT_Layer_512()
    H2 = solve_DLT_Layer_512_1(H1_motion+H2_motion)  ## H1.shape ==  (64, 3, 3)
    print ('H2.shape ==',H2.shape)## H1.shape ==  (64, 3, 3)
    
    #H1_mat = tf.matmul(tf.matmul(M_tile_inv, H1), M_tile)
    MatMulLayer_10 = MatMulLayer()
    H1_mat = MatMulLayer_10(M_tile_inv,H1,M_tile)
    print(' H1_mat shape after matmul ===',H1_mat.shape ) ##(64, 3, 3)
    
    #H2_mat = tf.matmul(tf.matmul(M_tile_inv, H2), M_tile)
    MatMulLayer_20 = MatMulLayer()
    H2_mat = MatMulLayer_20(M_tile_inv,H2,M_tile)
    print(' H2_mat shape after matmul ===',H2_mat.shape ) ##(64, 3, 3)
    
    ################################################prepare for calculating loss###
    ## warp images using H1/H2
    image2_tensor = train_inputs[..., 3:6]
    
    #warp2_H1 = transform(image2_tensor, H1_mat)
    transformLayer_3 = transformLayer()
    warp2_H1 = transformLayer_3(image2_tensor,H1_mat)
    print(' warp2_H1 shape  ===',warp2_H1.shape) ##(None, 512, 512, 3) 
    
    #warp2_H2 = transform(image2_tensor, H2_mat)
    transformLayer_4 = transformLayer()
    warp2_H2 = transformLayer_4(image2_tensor,H2_mat)
    print(' warp2_H2 shape  ===',warp2_H2.shape) ##(None, 512, 512, 3) 
    
    ######################################## warp all-one images using H1/H2
    #one = tf.ones_like(image2_tensor, dtype=tf.float32)
    tf_ones_like_layer = tf_ones_like()
    one = tf_ones_like_layer(image2_tensor)
    
    #one_warp_H1 = transform(one, H1_mat)
    transformLayer_5 = transformLayer()
    one_warp_H1 = transformLayer_5(one,H1_mat)
    print(' one_warp_H1 shape  ===',one_warp_H1.shape) ##(None, 512, 512, 3)
    
    #one_warp_H2 = transform(one, H2_mat)
    transformLayer_6 = transformLayer()
    one_warp_H2 = transformLayer_6(one,H2_mat)
    print(' one_warp_H2 shape  ===',one_warp_H2.shape) ##(None, 512, 512, 3)
    
    ################################################### initialize the mesh using H2
    #ini_mesh = H2Mesh(H2, patch_size)
    H2_Mesh_layer_1 = H2_Mesh_layer_512()
    ini_mesh = H2_Mesh_layer_1(H2)
    print('ini_mesh shape  ===',ini_mesh.shape) #=== (4, 9, 9, 2)
    
    ############################################ calculate the final predicted mesh
    mesh = ini_mesh + mesh_motion
    #depth = tf.concat([train_depth, train_depth, train_depth], 3)
    depth = layers.Concatenate(axis=3)([train_depth, train_depth, train_depth])
    print('depth.shape==',depth.shape) ##(None, 512, 512, 3)
    ############################################# warp the image/all-one image/ depth map using mesh
    #warp2_H3, one_warp_H3, warp2_depth = tf_spatial_transform_local.transformer(image2_tensor, one, depth, mesh)
    tf_local_transformer_1 = tf_local_transformer()
    warp2_H3, one_warp_H3, warp2_depth = tf_local_transformer_1(image2_tensor, one, depth, mesh)
    
    #warp2_depth = tf.expand_dims(tf.reduce_mean(warp2_depth, 3),3)
    tf_expand_reduce_mean_1 = tf_expand_reduce_mean()
    warp2_depth = tf_expand_reduce_mean_1(warp2_depth)
    
    #warp2_depth = tf.clip_by_value(warp2_depth,  0, 1)
    tf_clip_by_value_1 = tf_clip_by_value()
    warp2_depth =tf_clip_by_value_1(warp2_depth)

    print('warp2_depth ========',(warp2_depth.shape))
    print('mesh ========',(mesh.shape))
    print('warp2_H1 ========',(warp2_H1.shape))
    print('warp2_H2 ========',(warp2_H2.shape))
    print('warp2_H3 ========',(warp2_H3.shape))
    print('one_warp_H1 ========',(one_warp_H1.shape))
    print('one_warp_H2 ========',(one_warp_H2.shape))
    print('one_warp_H3 ========',(one_warp_H3.shape))
    
    ### for training 
    #return warp2_depth, mesh, warp2_H1, warp2_H2, warp2_H3, one_warp_H1, one_warp_H2, one_warp_H3
    
    ### for testing
    #return H1_motion, H2_motion, mesh_motion,warp2_depth, mesh, warp2_H1, warp2_H2, warp2_H3, one_warp_H1, one_warp_H2, one_warp_H3
    return Model(inputs=[train_inputs_aug, train_inputs, train_depth], outputs=[H1_motion, H2_motion, mesh_motion,warp2_depth, mesh, warp2_H1, warp2_H2, warp2_H3, one_warp_H1, one_warp_H2, one_warp_H3])

def _conv_block(x,n, num_out_layers, kernel_sizes, strides):
    x = Input(shape=( None, None, n))
    #conv1 = slim.conv2d(inputs=x, num_outputs=num_out_layers[0], kernel_size=kernel_sizes[0], activation_fn=tf.nn.relu, scope='conv1')
    conv1 = keras.layers.Conv2D(filters=num_out_layers[0], kernel_size=kernel_sizes[0], activation='relu',padding="same", name='conv1')(x)  

    #conv2 = slim.conv2d(inputs=conv1, num_outputs=num_out_layers[1], kernel_size=kernel_sizes[1], activation_fn=tf.nn.relu, scope='conv2')
    conv2 = keras.layers.Conv2D(filters=num_out_layers[1], kernel_size=kernel_sizes[1], activation='relu',padding="same", name='conv2')(conv1) 
    return Model(inputs=x, outputs=conv2)


def feature_extractor(image_tf):  ## [None, 512, 512, 3]
    image_tf = Input(shape=( 512, 512, 3))  ## or (None,None, 3)
    feature = []
    #with tf.compat.v1.variable_scope('conv_block1'):
    #conv1 = _conv_block(image_tf, ([64, 64]), (3, 3), (1, 1))      ## [None, 512, 512, 64]
    #maxpool1 = slim.max_pool2d(conv1, 2, stride=2, padding = 'SAME')
    cnn_block = _conv_block(image_tf,3, ([64, 64]), (3, 3), (1, 1))      ## 3 to 64 [None, 512, 512, 64]
    conv1 = cnn_block(image_tf)      ## 3 to 64[None, 512, 512, 64]
    maxpool1 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(conv1)##[None, 256, 256, 64]
    
    
      
    #with tf.compat.v1.variable_scope('conv_block2'):
    #conv2 = _conv_block(maxpool1, ([64, 64]), (3, 3), (1, 1))      #[None, 256, 256, 64]
    #maxpool2 = slim.max_pool2d(conv2, 2, stride=2, padding = 'SAME') ## [None, 128, 128, 64]
    cnn_block = _conv_block(maxpool1,64, ([64, 64]), (3, 3), (1, 1))     # 64 to 64 [None, 256, 256, 64]
    conv2 = cnn_block(maxpool1)      #64 to 64 [None, 256, 256, 64]
    maxpool2 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(conv2)  ## [None, 128, 128, 64]
      
    
    #with tf.compat.v1.variable_scope('conv_block3'):
    #conv3 = _conv_block(maxpool2, ([128, 128]), (3, 3), (1, 1))    #[None, 128, 128, 128]
    #maxpool3 = slim.max_pool2d(conv3, 2, stride=2, padding = 'SAME')## [None, 64, 64, 128]
    cnn_block = _conv_block(maxpool2,64, ([128, 128]), (3, 3), (1, 1))     #64 to 128 [None, 128, 128, 128]
    conv3 = cnn_block(maxpool2)      # 64 to 128 [None, 128, 128, 128]
    maxpool3 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(conv3)  ## [None, 64, 64, 128]
       
 
    #with tf.compat.v1.variable_scope('conv_block4'):
    #conv4 = _conv_block(maxpool3, ([128, 128]), (3, 3), (1, 1))    ## [None, 64, 64, 128]
    cnn_block = _conv_block(maxpool3,128, ([128, 128]), (3, 3), (1, 1))     #128 to 128 [None, 64, 64, 128]
    conv4 = cnn_block(maxpool3)      #128 to 128 [None, 64, 64, 128]    
    
    #conv1_r64 = tf.image.resize_images(conv1, [64, 64], method=0)
    #conv1_r64 = tf.image.resize(conv1, [64, 64], method=tf.image.ResizeMethod.BILINEAR)## [None, 64, 64, 64]
    #conv2_r64 = tf.image.resize(conv2, [64, 64], method=tf.image.ResizeMethod.BILINEAR)## [None, 64, 64, 64]
    #conv3_r64 = tf.image.resize(conv3, [64, 64], method=tf.image.ResizeMethod.BILINEAR)## [None, 64, 64, 128]
    
    conv1_r64 = tf.keras.layers.Resizing(64,64,interpolation='bilinear')(conv1) ## [None, 64, 64, 64]
    conv2_r64 = tf.keras.layers.Resizing(64,64,interpolation='bilinear')(conv2) ## [None, 64, 64, 64]
    conv3_r64 = tf.keras.layers.Resizing(64,64,interpolation='bilinear')(conv3) ## [None, 64, 64, 128]]
    print('conv1,2,3_r64.shape ',conv1_r64.shape,conv2_r64.shape,conv3_r64.shape,conv4.shape) ##(None, 64, 64, 64) (None, 64, 64, 64) (None, 64, 64, 128) (None, None, None, 128)
    
    #feature.append(tf.concat([conv1, conv2, conv3, conv4], 3))  ### for test  and debug
    #feature.append(tf.concat([conv4, conv1_r64, conv2_r64, conv3_r64], 3))##[None, 64, 64, 384]=[[None, 64, 64, 128][None, 64, 64, 64][None, 64, 64, 64][None, 64, 64, 128]]
    conc0=keras.layers.concatenate([conv4, conv1_r64, conv2_r64, conv3_r64])
    print('conc0',conc0.shape)  ## conc0 (None, 64, 64, 384)
    feature.append(conc0) ##[None, 64, 64, 384]=[[None, 64, 64, 128][None, 64, 64, 64][None, 64, 64, 64][None, 64, 64, 128]]
    maxpool4 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(conv4)  ## [None, 32, 32, 128]
    
    #with tf.compat.v1.variable_scope('conv_block5'):
    #conv5 = _conv_block(maxpool4, ([256, 256]), (3, 3), (1, 1))    #32
    cnn_block = _conv_block(maxpool4,128, ([256, 256]), (3, 3), (1, 1))     #128 to 256 [None, 32, 32, 256]
    conv5 = cnn_block(maxpool4)      #128 to 256 [None, 32, 32, 256]
    
    #conv1_r32 = tf.image.resize(conv1, [32, 32], method=tf.image.ResizeMethod.BILINEAR)
    #conv2_r32 = tf.image.resize(conv2, [32, 32], method=tf.image.ResizeMethod.BILINEAR)
    #conv3_r32 = tf.image.resize(conv3, [32, 32], method=tf.image.ResizeMethod.BILINEAR)
    #conv4_r32 = tf.image.resize(conv4, [32, 32], method=tf.image.ResizeMethod.BILINEAR)
    
    conv1_r32 = tf.keras.layers.Resizing(32, 32,interpolation='bilinear')(conv1) ##[None, 512, 512, 64] 
    conv2_r32 = tf.keras.layers.Resizing(32, 32,interpolation='bilinear')(conv2) ##[None, 256, 256, 64]
    conv3_r32 = tf.keras.layers.Resizing(32, 32,interpolation='bilinear')(conv3) ##[None, 128, 128, 128]
    conv4_r32 = tf.keras.layers.Resizing(32, 32,interpolation='bilinear')(conv4) ##[None, 64, 64, 128]
    print('conv1,2,3,4_r32.shape ',conv1_r32.shape,conv2_r32.shape,conv3_r32.shape,conv4_r32,conv5.shape) ##
    
    #feature.append(tf.concat([conv5, conv1_r32, conv2_r32, conv3_r32, conv4_r32], 3))
    conc1=keras.layers.concatenate([conv5, conv1_r32, conv2_r32, conv3_r32, conv4_r32])
    print('conc1',conc1.shape)  ## conc1 
    feature.append(conc1) ##
    
    maxpool5 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(conv5)  ## [None, 16, 16, 256]
    
    #with tf.compat.v1.variable_scope('conv_block6'):                                      
    #conv6 = _conv_block(maxpool5, ([256, 256]), (3, 3), (1, 1))    #16
    cnn_block = _conv_block(maxpool5,256, ([256, 256]), (3, 3), (1, 1))     #256 to 256[None, 16, 16, 256]
    conv6 = cnn_block(maxpool5)      #256 to 256[None, 16, 16, 256]  
    
    #conv1_r16 = tf.image.resize(conv1, [16, 16], method=tf.image.ResizeMethod.BILINEAR)
    #conv2_r16 = tf.image.resize(conv2, [16, 16], method=tf.image.ResizeMethod.BILINEAR)
    #conv3_r16 = tf.image.resize(conv3, [16, 16], method=tf.image.ResizeMethod.BILINEAR)
    #conv4_r16 = tf.image.resize(conv4, [16, 16], method=tf.image.ResizeMethod.BILINEAR)
    #conv5_r16 = tf.image.resize(conv5, [16, 16], method=tf.image.ResizeMethod.BILINEAR)
    
    conv1_r16 = tf.keras.layers.Resizing(16, 16,interpolation='bilinear')(conv1) ##
    conv2_r16 = tf.keras.layers.Resizing(16, 16,interpolation='bilinear')(conv2) ##
    conv3_r16 = tf.keras.layers.Resizing(16, 16,interpolation='bilinear')(conv3) ##
    conv4_r16 = tf.keras.layers.Resizing(16, 16,interpolation='bilinear')(conv4) ##
    conv5_r16 = tf.keras.layers.Resizing(16, 16,interpolation='bilinear')(conv5) ##
    print('conv1,2,3,4,5_r16.shape ',conv1_r16.shape,conv2_r16.shape,conv3_r16.shape,conv4_r16,conv5_r16,conv6.shape) ##
    
    #feature.append(tf.concat([conv6, conv1_r16, conv2_r16, conv3_r16, conv4_r16, conv5_r16], 3))
    conc2=keras.layers.concatenate([conv6, conv1_r16, conv2_r16, conv3_r16, conv4_r16, conv5_r16])
    print('conc1',conc2.shape)  ## conc2 
    feature.append(conc2) ##
    
    import keras.backend as K
    #autoencoder.add(Lambda(lambda x: K.l2_normalize(x,axis=1)))
    ff=keras.layers.Lambda(lambda x: K.l2_normalize(x,axis=-1))
    f_1norm = keras.layers.LayerNormalization(axis=[-1])(feature[-1])
    f_2norm = keras.layers.LayerNormalization(axis=[-1])(feature[-2])
    f_3norm = keras.layers.LayerNormalization(axis=[-1])(feature[-3])
    
    #print('feature.shape-1 ===', tf.nn.l2_normalize(feature[-1]).shape)### (None, 16, 16, 896)feature is list[0--2] 
    print('feature.shape-1 ===',f_1norm.shape,ff)### (None, 16, 16, 896)  ##feature is list[0--2] 
    print('feature.shape-2 ===', f_2norm.shape)### (None, 32, 32, 640)
    print('feature.shape-3 ===', f_3norm.shape)### (None, 64, 64, 384)
    return feature,Model(inputs=image_tf, outputs=feature)


# contextual correlation layer
def CCL(c1, warp,s1,s2,s3):
    c1 = Input(shape=( s1, s2, s3),batch_size=3)  ## or (16, 16, 896)
    warp = Input(shape=( s1, s2, s3),batch_size=3)  ## or (16, 16, 896)
    
    print('c1.shape ===', c1.shape)  ## (None, 16, 16, 896) or (None, None, None, None)
    print('warp.shape ===', warp.shape)  ## (None, 16, 16, 896)
    
    #shape = warp.get_shape().as_list()
    shape = warp.shape
    print('shape  shape[0] shape[1] ===',shape,shape[0],shape[1])  ### [64, 16, 16, 896] 16
    kernel = 3
    stride = 1
    rate = 1
    if shape[1] == 16:
      rate = 1
      stride = 1
    elif shape[1] == 32:
      rate = 2
      stride = 1
    else:
      rate = 3
      stride = 1
    
    # extract patches as convolutional filters
    ##patches = tf.extract_image_patches(warp, [1,kernel,kernel,1], [1,stride,stride,1], [1,rate,rate,1], padding='SAME')
    ## Extract patches from images and put them in the "depth" output dimension
    # Ideally, given an input image tensor of size 1x225x225x3 (where 1 is the batch size),
    # I want to be able to get Kx32x32x3 as output,
    # where K is the total number of patches 
    # and 32x32x3 is the dimension of each patch.
    # Is there something in tensorflow that already achieves this?
    
    #patches = tf.compat.v1.extract_image_patches(warp, [1,kernel,kernel,1], [1,stride,stride,1], [1,rate,rate,1], padding='SAME')
    #patches = keras.ops.image.extract_patches(warp,(kernel,kernel), (stride,stride),(rate,rate),padding="same")
    patches = ExtractPatchesLayer(kernel_size=kernel, strides=stride, rates=rate, padding="same")(warp)
    print('patches===',patches.shape)  ###  patches=== ((64, 16, 16, 8064)  3*3*896=8064  
    
    #patches = tf.reshape(patches, [shape[0], -1, kernel, kernel, shape[3]])
    ## [None, -1, 3, 3, 896]
    patches = keras.layers.Reshape((-1, kernel, kernel, shape[3]))(patches)
    print('patches===',patches.shape)  ## ((64, 256, 3, 3, 896)
    
    #matching_filters = tf.transpose(patches, [0, 2, 3, 4, 1])
    matching_filters = TransposeLayer(perm=[0, 2, 3, 4, 1])(patches)  
    print('matching_filters.shape===',matching_filters.shape)  ##(None, 3, 3, 896, 256)
    
    # using convolution to match
    match_vol = []
      ## shape[0]== None
    for i in range(shape[0]):
      #expc1i0 = tf.expand_dims(c1[i], [0])
      expc1i0 = ExpandDimsLayer(axis=0)(c1[i])
      matchfili = matching_filters[i]
      #print('expc1i0.shape  ,matchfili.shape ===',expc1i0.shape,matchfili.shape)
      ## (1, 16, 16, 896) (3, 3, 896, 256)
      ## (1, 32, 32, 640) (3, 3, 640, 1024)
      ## (1, 64, 64, 384) (3, 3, 384, 4096)
      
      #single_match = tf.nn.atrous_conv2d(expc1i0, matchfili, rate=rate, padding='SAME')
      single_match = tf.keras.layers.Conv2D(
        filters=matchfili.shape[-1],  # Number of output channels (derived from matchfili)
        kernel_size=(matchfili.shape[0], matchfili.shape[1]), # Filter dimensions
        dilation_rate=rate,
        padding='same',  # Keras uses lowercase 'same'
        use_bias=False,  # Typically no bias when directly mimicking a convolution operation
        #weights=[matchfili] # If you want to initialize with the 'matchfili' weights
      )(expc1i0)
      
      #print('single_match.shape==',single_match.shape) ##single_match.shape== (1, 16, 16, 256)
      match_vol.append(single_match)
    
    print(len(match_vol))  ## 64
    #match_vol = tf.concat(match_vol, axis=0)
    match_vol = layers.Concatenate(axis=0)(match_vol)
    print('match_vol.shape==',match_vol.shape) ##(64, 16, 16, 256)
    channels = match_vol.shape[3]
    print("channels=====",match_vol.shape[3])  ##256
   
    
    # scale softmax
    softmax_scale = 10
    #match_vol = tf.nn.softmax(match_vol*softmax_scale,3)
    match_vol = layers.Softmax(axis=3)(match_vol*softmax_scale)
    print('match_vol.shape==',match_vol.shape) ##(64, 16, 16, 256)
    
    # convert the correlation volume to feature flow
    h_one = tf.linspace(0., shape[1]-1., int(match_vol.shape[1]))
    w_one = tf.linspace(0., shape[2]-1., int(match_vol.shape[2]))
    h_one = tf.matmul(tf.expand_dims(h_one, 1), tf.ones(shape=tf.stack([1, shape[2]])))
    w_one = tf.matmul(tf.ones(shape=tf.stack([shape[1], 1])), tf.transpose(tf.expand_dims(w_one, 1), [1, 0]))
    h_one = tf.tile(tf.expand_dims(tf.expand_dims(h_one, 0),3), [shape[0],1,1,channels])
    w_one = tf.tile(tf.expand_dims(tf.expand_dims(w_one, 0),3), [shape[0],1,1,channels])
    
    i_one = tf.expand_dims(tf.linspace(0., channels-1., channels),0)
    i_one = tf.expand_dims(i_one,0)
    i_one = tf.expand_dims(i_one,0)
    i_one = tf.tile(i_one, [shape[0], shape[1], shape[2], 1])
 
    flow_w = match_vol*(i_one%shape[2] - w_one)
    flow_h = match_vol*(i_one//shape[2] - h_one)
    
    #flow_w = tf.expand_dims(tf.reduce_sum(flow_w,3),3)
    # Reduce sum along axis 3 using a Lambda layer
    reduced_flow_w = layers.Lambda(lambda x: tf.reduce_sum(x, axis=3))(flow_w)
    #flow_w = tf.expand_dims(reduced_flow_w,3)
    flow_w = ExpandDimsLayer(axis=3)(reduced_flow_w)
    
    #flow_h = tf.expand_dims(tf.reduce_sum(flow_h,3),3)
    # Reduce sum along axis 3 using a Lambda layer
    reduced_flow_h = layers.Lambda(lambda x: tf.reduce_sum(x, axis=3))(flow_h)
    #flow_h = tf.expand_dims(reduced_flow_h,3)
    flow_h = ExpandDimsLayer(axis=3)(reduced_flow_h)
    
    
    #flow = tf.concat([flow_w, flow_h], 3)
    flow = layers.Concatenate(axis=3)([flow_w, flow_h])
    print("flow.shape",flow.shape)  ### (64, 16, 16, 2)  ,  (64, 32, 32, 2)  ,(64, 64, 64, 2)

    return Model(inputs=[c1,warp], outputs=flow)
    #return flow
    
def regression_H4pt_Net1(correlation):
    correlation = Input(shape=(16, 16, 2))  ## 64,16, 16, 2
    #conv1 = slim.conv2d(inputs=correlation, num_outputs=64, kernel_size=3, activation_fn=tf.nn.relu)
    #conv1 = slim.conv2d(inputs=conv1, num_outputs=64, kernel_size=3, activation_fn=tf.nn.relu)
    conv1 = keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu',padding="same")(correlation)  
    conv1 = keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu',padding="same", name='conv1')(conv1)  

    #maxpool1 = slim.max_pool2d(conv1, 2, stride=2, padding = 'SAME')
    #conv2 = slim.conv2d(inputs=maxpool1, num_outputs=64, kernel_size=3, activation_fn=tf.nn.relu)
    #conv2 = slim.conv2d(inputs=conv2, num_outputs=64, kernel_size=3, activation_fn=tf.nn.relu)
    maxpool1 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(conv1)
    conv2 = keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu',padding="same")(maxpool1)  
    conv2 = keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu',padding="same", name='conv2')(conv2)  

    
    
    #maxpool2 = slim.max_pool2d(conv2, 2, stride=2, padding = 'SAME')
    #conv3 = slim.conv2d(inputs=maxpool2, num_outputs=128, kernel_size=3, activation_fn=tf.nn.relu)
    #conv3 = slim.conv2d(inputs=conv3, num_outputs=128, kernel_size=3, activation_fn=tf.nn.relu)
    maxpool2 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(conv2)
    conv3 = keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu',padding="same")(maxpool2)  
    conv3 = keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu',padding="same", name='conv3')(conv3)  
   
    #fc1 = slim.conv2d(inputs=conv3, num_outputs=128, kernel_size=4, activation_fn=tf.nn.relu, padding="VALID")
    #fc2 = slim.conv2d(inputs=fc1, num_outputs=128, kernel_size=1, activation_fn=tf.nn.relu)
    #fc3 = slim.conv2d(inputs=fc2, num_outputs=8, kernel_size=1, activation_fn=None)
    
    fc1 = keras.layers.Conv2D(filters=128, kernel_size=4, activation='relu',padding="valid", name='fc1')(conv3)  
    fc2 = keras.layers.Conv2D(filters=128, kernel_size=1, activation='relu',padding="same", name='fc2')(fc1)
    fc3 = keras.layers.Conv2D(filters=8, kernel_size=1, activation=None,padding="same", name='fc3')(fc2)

    #H1_motion = tf.expand_dims(tf.squeeze(tf.squeeze(fc3,1),1), [2])
    tf_expand_squeeze_layer = tf_expand_squeeze()
    H1_motion = tf_expand_squeeze_layer(fc3)
    
    
    return Model(inputs=correlation, outputs=H1_motion)
    
def regression_H4pt_Net2(correlation):
    correlation = Input(shape=(32, 32, 2))  ## 64,32, 32, 2
    #conv1 = slim.conv2d(inputs=correlation, num_outputs=64, kernel_size=3, activation_fn=tf.nn.relu)
    #conv1 = slim.conv2d(inputs=conv1, num_outputs=64, kernel_size=3, activation_fn=tf.nn.relu)
    conv1 = keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu',padding="same")(correlation)  
    conv1 = keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu',padding="same", name='conv1')(conv1)  
    
    #maxpool1 = slim.max_pool2d(conv1, 2, stride=2, padding = 'SAME')
    #conv2 = slim.conv2d(inputs=maxpool1, num_outputs=64, kernel_size=3, activation_fn=tf.nn.relu)
    #conv2 = slim.conv2d(inputs=conv2, num_outputs=64, kernel_size=3, activation_fn=tf.nn.relu)
    maxpool1 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(conv1)
    conv2 = keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu',padding="same")(maxpool1)  
    conv2 = keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu',padding="same", name='conv2')(conv2)  
    
    #maxpool2 = slim.max_pool2d(conv2, 2, stride=2, padding = 'SAME')
    #conv3 = slim.conv2d(inputs=maxpool2, num_outputs=128, kernel_size=3, activation_fn=tf.nn.relu)
    #conv3 = slim.conv2d(inputs=conv3, num_outputs=128, kernel_size=3, activation_fn=tf.nn.relu)
    maxpool2 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(conv2)
    conv3 = keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu',padding="same")(maxpool2)  
    conv3 = keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu',padding="same", name='conv3')(conv3)  
    
    #maxpool3 = slim.max_pool2d(conv3, 2, stride=2, padding = 'SAME')
    #conv4 = slim.conv2d(inputs=maxpool3, num_outputs=128, kernel_size=3, activation_fn=tf.nn.relu)
    #conv4 = slim.conv2d(inputs=conv4, num_outputs=128, kernel_size=3, activation_fn=tf.nn.relu)
    maxpool3 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(conv3)
    conv4 = keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu',padding="same")(maxpool3)  
    conv4 = keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu',padding="same", name='conv4')(conv4)  
   
    #fc1 = slim.conv2d(inputs=conv4, num_outputs=128, kernel_size=4, activation_fn=tf.nn.relu, padding="VALID")
    #fc2 = slim.conv2d(inputs=fc1, num_outputs=128, kernel_size=1, activation_fn=tf.nn.relu)
    #fc3 = slim.conv2d(inputs=fc2, num_outputs=8, kernel_size=1, activation_fn=None)
    fc1 = keras.layers.Conv2D(filters=128, kernel_size=4, activation='relu',padding="valid", name='fc1')(conv4)  
    fc2 = keras.layers.Conv2D(filters=128, kernel_size=1, activation='relu',padding="same", name='fc2')(fc1)
    fc3 = keras.layers.Conv2D(filters=8, kernel_size=1, activation=None,padding="same", name='fc3')(fc2)

    #H2_motion = tf.expand_dims(tf.squeeze(tf.squeeze(fc3,1),1), [2])
    tf_expand_squeeze_layer = tf_expand_squeeze()
    H2_motion = tf_expand_squeeze_layer(fc3)
    
    return Model(inputs=correlation, outputs=H2_motion)
    
def regression_H4pt_Net3(correlation):
    correlation = Input(shape=(64, 64, 2))  ## 64, 64, 64, 2
    #conv1 = slim.conv2d(inputs=correlation, num_outputs=64, kernel_size=3, activation_fn=tf.nn.relu)
    #conv1 = slim.conv2d(inputs=conv1, num_outputs=64, kernel_size=3, activation_fn=tf.nn.relu)
    conv1 = keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu',padding="same")(correlation)  
    conv1 = keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu',padding="same", name='conv1')(conv1)  
    
    #maxpool1 = slim.max_pool2d(conv1, 2, stride=2, padding = 'SAME')
    #conv2 = slim.conv2d(inputs=maxpool1, num_outputs=64, kernel_size=3, activation_fn=tf.nn.relu)
    #conv2 = slim.conv2d(inputs=conv2, num_outputs=64, kernel_size=3, activation_fn=tf.nn.relu)
    maxpool1 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(conv1)
    conv2 = keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu',padding="same")(maxpool1)  
    conv2 = keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu',padding="same", name='conv2')(conv2)  
    
    #maxpool2 = slim.max_pool2d(conv2, 2, stride=2, padding = 'SAME')
    #conv3 = slim.conv2d(inputs=maxpool2, num_outputs=128, kernel_size=3, activation_fn=tf.nn.relu)
    #conv3 = slim.conv2d(inputs=conv3, num_outputs=128, kernel_size=3, activation_fn=tf.nn.relu)
    maxpool2 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(conv2)
    conv3 = keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu',padding="same")(maxpool2)  
    conv3 = keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu',padding="same", name='conv3')(conv3)  
    
    #maxpool3 = slim.max_pool2d(conv3, 2, stride=2, padding = 'SAME')
    #conv4 = slim.conv2d(inputs=maxpool3, num_outputs=128, kernel_size=3, activation_fn=tf.nn.relu)
    #conv4 = slim.conv2d(inputs=conv4, num_outputs=128, kernel_size=3, activation_fn=tf.nn.relu)
    maxpool3 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(conv3)
    conv4 = keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu',padding="same")(maxpool3)  
    conv4 = keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu',padding="same", name='conv4')(conv4)  
    
    #maxpool4 = slim.max_pool2d(conv4, 2, stride=2, padding = 'SAME')
    #conv5 = slim.conv2d(inputs=maxpool4, num_outputs=256, kernel_size=3, activation_fn=tf.nn.relu)
    #conv5 = slim.conv2d(inputs=conv5, num_outputs=256, kernel_size=3, activation_fn=tf.nn.relu)
    maxpool4 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(conv4)
    conv5 = keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu',padding="same")(maxpool4)  
    conv5 = keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu',padding="same", name='conv5')(conv5)  
  
    #fc1 = slim.conv2d(inputs=conv5, num_outputs=2048, kernel_size=4, activation_fn=tf.nn.relu, padding="VALID")
    #fc2 = slim.conv2d(inputs=fc1, num_outputs=1024, kernel_size=1, activation_fn=tf.nn.relu)
    #fc3 = slim.conv2d(inputs=fc2, num_outputs=(grid_w+1)*(grid_h+1)*2, kernel_size=1, activation_fn=None)

    fc1 = keras.layers.Conv2D(filters=2048, kernel_size=4, activation='relu',padding="valid", name='fc1')(conv5)  
    fc2 = keras.layers.Conv2D(filters=1024, kernel_size=1, activation='relu',padding="same", name='fc2')(fc1)
    fc3 = keras.layers.Conv2D(filters=(grid_w+1)*(grid_h+1)*2, kernel_size=1, activation=None,padding="same", name='fc3')(fc2)


    print('fc3 shape ===',fc3.shape)
    #net3_f = tf.expand_dims(tf.squeeze(tf.squeeze(fc3,1),1), [2])
    #mesh_motion = tf.reshape(fc3, (-1, grid_h+1, grid_w+1, 2))
    mesh_motion = keras.layers.Reshape((grid_h+1, grid_w+1, 2))(fc3)
    print('mesh_motion shape ===',mesh_motion.shape)
    
    return Model(inputs=correlation, outputs=mesh_motion)



def _vgg(input1, input2):
    input1 = Input(shape=( 512, 512, 3),batch_size=3)  ## or (None,None, 3)
    input2 = Input(shape=( 512, 512, 3),batch_size=3)  ## or (None,None, 3)
    batch_size = input1.shape[0]
    print('batch_size ===',batch_size)
    
    # ## feature extractors with shared weights
    # with tf.compat.v1.variable_scope('feature_extract', reuse = None): 
      # feature1 = feature_extractor(input1)
    # with tf.compat.v1.variable_scope('feature_extract', reuse = True): 
      # feature2 = feature_extractor(input2)
     
    f,feature_extract_ = feature_extractor(input1)    
    feature1 = feature_extract_(input1)
    f,feature_extract_ = feature_extractor(input2)    
    feature2 = feature_extract_(input2)
     
    print('feature1[-1] feature2[-1] ==' ,feature1[-1].shape, feature2[-1].shape) 
    ##(64, 16, 16, 896) (64, 16, 16, 896)
     
    ################################################# the 1st layer of the pyramid
    #featureflow_1 = CCL(tf.nn.l2_normalize(feature1[-1],axis=3), tf.nn.l2_normalize(feature2[-1],axis=3))
    a = keras.layers.LayerNormalization(axis=[-1])(feature1[-1]) ##(64, 16, 16, 896)
    b = keras.layers.LayerNormalization(axis=[-1])(feature2[-1]) ##(64, 16, 16, 896)
    CCL_1 = CCL(a,b,a.shape[1],a.shape[2],a.shape[3])  
    featureflow_1 = CCL_1([a,b]) ## (64, 16, 16, 2)
    #CCL_Layar_1 = CCL_Layar()
    #featureflow_1 = CCL_Layar_1(a,b)
 
    regression_H4pt_Net1_1 = regression_H4pt_Net1(featureflow_1)
    H1_motion = regression_H4pt_Net1_1(featureflow_1)
    print('H1_motion shape ===',H1_motion.shape)
    ##(None, 8,1)
    
    # warp the feature map
    patch_size = 32.
    #H1 = solve_DLT(H1_motion/16., patch_size)
    solve_DLT_Layer_1 = solve_DLT_Layer_32()
    H1 = solve_DLT_Layer_1(H1_motion/16.)  
    ## H1.shape ==  (64, 3, 3)
    
    M = np.array([[patch_size / 2.0, 0., patch_size / 2.0],
                  [0., patch_size / 2.0, patch_size / 2.0],
                  [0., 0., 1.]]).astype(np.float32)
    M_tensor = tf.constant(M, tf.float32)
    M_tile = tf.tile(tf.expand_dims(M_tensor, [0]), [batch_size, 1, 1])
    M_inv = np.linalg.inv(M)
    M_tensor_inv = tf.constant(M_inv, tf.float32)
    M_tile_inv = tf.tile(tf.expand_dims(M_tensor_inv, [0]), [batch_size, 1, 1])
    
    print('M_tile_inv , H1  M_tile shape ===',M_tile_inv.shape,H1.shape,M_tile.shape )##(64, 3, 3) (64, 3, 3)(64, 3, 3)
    #H1 = tf.matmul(tf.matmul(M_tile_inv, H1), M_tile)
    MatMulLayer_1 = MatMulLayer()
    H1 = MatMulLayer_1(M_tile_inv,H1,M_tile)
    print(' H1 shape after matmul ===',H1.shape ) ##(64, 3, 3)
    
    #feature2_warp = transform(tf.nn.l2_normalize(feature2[-2],axis=3), H1)
    a = keras.layers.LayerNormalization(axis=[-1])(feature2[-2])
    transformLayer_1 = transformLayer()
    feature2_warp = transformLayer_1(a,H1)
    print(' feature2_warp shape  ===',feature2_warp.shape,a.shape,H1.shape ) ##(None, 32, 32, 640) (None, 32, 32, 640) (64, 3, 3)
    
    ############################################################### the 2nd layer of the pyramid
    ##(64, 32, 32, 640) (64, 32, 32, 640)
    #featureflow_2 = CCL(tf.nn.l2_normalize(feature1[-2],axis=3), feature2_warp) ## (64, 32, 32, 2)
    a = keras.layers.LayerNormalization(axis=[-1])(feature1[-2])
    b = feature2_warp  ##(64, 32, 32, 640)
    CCL_2 = CCL(a,b,a.shape[1],a.shape[2],a.shape[3])  
    featureflow_2 = CCL_2([a,b])
    print("featureflow_2.shape",featureflow_2.shape)  ##(64, 32, 32, 2)
    
    
    #H2_motion = regression_H4pt_Net2(featureflow_2)
    regression_H4pt_Net2_1 = regression_H4pt_Net2(featureflow_2)
    H2_motion = regression_H4pt_Net2_1(featureflow_2)
    print('H2_motion shape ===',H2_motion.shape) ##(64, 8, 1)
   
    
    # warp the feature map
    patch_size = 64.
    #H2 = solve_DLT((H1_motion+H2_motion)/8., patch_size)
    solve_DLT_Layer_2 = solve_DLT_Layer_64()
    H2 = solve_DLT_Layer_2((H1_motion+H2_motion)/8.)  
    print('H2 shape ===',H2.shape)  ## H2.shape ==  (64, 3, 3)
    
    M = np.array([[patch_size / 2.0, 0., patch_size / 2.0],
                  [0., patch_size / 2.0, patch_size / 2.0],
                  [0., 0., 1.]]).astype(np.float32)
    M_tensor = tf.constant(M, tf.float32)
    M_tile = tf.tile(tf.expand_dims(M_tensor, [0]), [batch_size, 1, 1])
    M_inv = np.linalg.inv(M)
    M_tensor_inv = tf.constant(M_inv, tf.float32)
    M_tile_inv = tf.tile(tf.expand_dims(M_tensor_inv, [0]), [batch_size, 1, 1])
    
    #H2 = tf.matmul(tf.matmul(M_tile_inv, H2), M_tile)
    MatMulLayer_2 = MatMulLayer()
    H2 = MatMulLayer_2(M_tile_inv,H2,M_tile)
    print(' H2 shape after matmul ===',H2.shape ) ##(64, 3, 3)
    
    #feature3_warp = transform(tf.nn.l2_normalize(feature2[-3],axis=3), H2)
    a = keras.layers.LayerNormalization(axis=[-1])(feature2[-3])
    transformLayer_2 = transformLayer()
    feature3_warp = transformLayer_2(a,H2)
    print(' feature3_warp shape  ===',feature3_warp.shape,a.shape,H2.shape ) ##(None, 64, 64, 384) (None, 64, 64, 384) (64, 3, 3)
    
    
    ################################################################# the 3rd layer of the pyramid
    ## (64, 64, 64, 384) (64, 64, 64, 384)
    #featureflow_3 = CCL(tf.nn.l2_normalize(feature1[-3],axis=3), feature3_warp) ##(64, 64, 64, 2)
    a = keras.layers.LayerNormalization(axis=[-1])(feature1[-3])
    b = feature3_warp  ##(64, 32, 32, 640)
    CCL_3 = CCL(a,b,a.shape[1],a.shape[2],a.shape[3])  
    featureflow_3 = CCL_3([a,b])
    print("featureflow_3.shape",featureflow_3.shape)  ##(64, 64, 64, 2)
    
    #mesh_motion = regression_H4pt_Net3(featureflow_3)
    regression_H4pt_Net3_1 = regression_H4pt_Net3(featureflow_3)
    mesh_motion = regression_H4pt_Net3_1(featureflow_3)
    print('mesh_motion shape ===',mesh_motion.shape)  ##(None, 9, 9, 2)
    
    
    return Model(inputs=[input1, input2], outputs=[H1_motion, H2_motion, mesh_motion])
    #return H1_motion, H2_motion, mesh_motion
#######################################################################################
class AugmentLayer(Layer):
    def call(self, inputs):
        return disjoint_augment_image_pair(inputs)  # This line should be indented
    def compute_output_shape(self, input_shape):
        # Assuming disjoint_augment_image_pair preserves the shape
        return input_shape

class H2_Mesh_layer_512(tf.keras.layers.Layer):
    def call(self, H2):
        return H2Mesh(H2, 512.)
        
class tf_local_transformer(tf.keras.layers.Layer):
    def call(self, image2_tensor, one, depth, mesh):
        return tf_spatial_transform_local.transformer(image2_tensor, one, depth, mesh)
        

class AbsoluteDifferenceLayer(layers.Layer):
    def call(self, inputs):
        x, y = inputs
        return K.abs(x - y)
        
class tf_ones_like(tf.keras.layers.Layer):
    def call(self, image2_tensor):
        return tf.ones_like(image2_tensor, dtype=tf.float32)
        
class Hmodel_512(tf.keras.layers.Layer):
    def call(self, train_inputs_aug, train_inputs, train_depth):
                ##Ensure all inputs are tensors
        train_inputs_aug = tf.convert_to_tensor(train_inputs_aug)
        train_inputs = tf.convert_to_tensor(train_inputs)
        train_depth = tf.convert_to_tensor(train_depth)
        return H_model(train_inputs_aug, train_inputs, train_depth,512)

        
class solve_DLT_Layer_32(tf.keras.layers.Layer):
    def call(self, x):
        #y = np.float32(y)
        #y = tf.cast(y, tf.float32)
        #y = tf.float32(y)
        return solve_DLT(x,32.)
        
class solve_DLT_Layer_64(tf.keras.layers.Layer):
    def call(self, x):
        #y = np.float32(y)
        #y = tf.cast(y, tf.float32)
        #y = tf.float32(y)
        return solve_DLT(x,64.)
        
class solve_DLT_Layer_512(tf.keras.layers.Layer):
    def call(self, x):
        #y = np.float32(y)
        #y = tf.cast(y, tf.float32)
        #y = tf.float32(y)
        return solve_DLT(x,512.)
        
class MatMulLayer(tf.keras.layers.Layer):
    def call(self, A, B, C):
        return tf.matmul(tf.matmul(A, B), C)
        
class transformLayer(tf.keras.layers.Layer):
    def call(self, A, B):
        return transform(A, B)
   
 
        
class tf_expand_squeeze(tf.keras.layers.Layer):
    def call(self, x):
        return tf.expand_dims(tf.squeeze(tf.squeeze(x,1),1), [2])

class tf_expand_reduce_mean(tf.keras.layers.Layer):
    def call(self, warp2_depth):
        return tf.expand_dims(tf.reduce_mean(warp2_depth, 3),3)

class tf_clip_by_value(tf.keras.layers.Layer):
    def call(self, warp2_depth):
        return tf.clip_by_value(warp2_depth,  0, 1) 
     
class CCL_Layar(tf.keras.layers.Layer):
    def call(self,c1, warp):
        return CCL(c1, warp)
       
class ExtractPatchesLayer(keras.layers.Layer):
    def __init__(self, kernel_size, strides, rates, padding="same", **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = self._validate_tuple(kernel_size, "kernel_size")
        self.strides = self._validate_tuple(strides, "strides")
        self.rates = self._validate_tuple(rates, "rates")
        self.padding = padding.upper()  # Keras ops expects uppercase padding

    def _validate_tuple(self, value, name):
        if isinstance(value, int):
            return (value, value)
        elif isinstance(value, tuple) and len(value) == 2:
            return value
        else:
            raise ValueError(
                f"'{name}' must be an integer or a tuple of two integers. "
                f"Received: {value}"
            )

    def call(self, inputs):
        return keras.ops.image.extract_patches(
            inputs,
            size =self.kernel_size,
            strides=self.strides,
            dilation_rate=self.rates,
            padding=self.padding
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "rates": self.rates,
            "padding": self.padding,
        })
        return config
class ExpandDimsLayer(keras.layers.Layer):
    def __init__(self, axis, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({'axis': self.axis})
        return config

class TransposeLayer(keras.layers.Layer):
    def __init__(self, perm, **kwargs):
        super().__init__(**kwargs)
        self.perm = perm

    def call(self, inputs):
        return tf.transpose(inputs, perm=self.perm)

    def get_config(self):
        config = super().get_config()
        config.update({'perm': self.perm})
        return config


############# Define a function to calculate the loss
def compute_loss(train_inputs_aug, train_inputs, train_depth):
    # content loss
    #with tf.compat.v1.variable_scope('generator', reuse=tf.compat.v1.AUTO_REUSE):  # Added reuse=tf.compat.v1.AUTO_REUSE
    #H1_motion, H2_motion, mesh_motion,train_warp2_depth, train_mesh, train_warp2_H1, train_warp2_H2, train_warp2_H3, train_one_warp_H1, train_one_warp_H2, train_one_warp_H3 = H_estimator(train_inputs_aug, train_inputs, train_depth)
    
    # ... In compute_loss function ...
    h_estimator_layer = HEstimatorLayer()  # Create an instance
    H1_motion, H2_motion, mesh_motion,train_warp2_depth, train_mesh, train_warp2_H1, train_warp2_H2, train_warp2_H3, train_one_warp_H1, train_one_warp_H2, train_one_warp_H3 = h_estimator_layer(train_inputs_aug, train_inputs, train_depth) 


    lam_lp = 1
    loss1 = intensity_loss(gen_frames=train_warp2_H1, gt_frames=train_inputs[...,0:3]*train_one_warp_H1, l_num=1)
    loss2 = intensity_loss(gen_frames=train_warp2_H2, gt_frames=train_inputs[...,0:3]*train_one_warp_H2, l_num=1)
    loss3 = intensity_loss(gen_frames=train_warp2_H3, gt_frames=train_inputs[...,0:3]*train_one_warp_H3, l_num=1)
    lp_loss = 16. * loss1 + 4. * loss2 + 1. * loss3
    # mesh loss
    lam_mesh = 0   ##   deph assistance??   without lam_mesh = 0      within 10
    mesh_loss = depth_consistency_loss3(train_warp2_depth, train_mesh)
    # total loss
    g_loss = tf.add_n([lp_loss * lam_lp, mesh_loss * lam_mesh], name='g_loss')
    
    print('mesh_loss==',tf.shape(mesh_loss),format(mesh_loss),mesh_loss)
    print('lp_loss==',tf.shape(lp_loss),format(lp_loss),lp_loss)
    print('g_loss==',tf.shape(g_loss),format(g_loss),g_loss)
    print('lam_mesh==',tf.shape(lam_mesh),format(lam_mesh),lam_mesh)
     
    return g_loss, lp_loss, mesh_loss # Return all losses
##################################################################
###################################################################
# homograph_pic = np.load('/content/gdrive/My Drive/homograph_pic_np.npy')
# #### homograph_pic.shape (8, 3, 512, 512, 6) <class 'tuple'> 5
# ## normalize_image(frame2 / 127.5) - 1.0

# batch_size = 3   ###constant.TEST_BATCH_SIZE
# height, width = 512, 512
# ##test_inputs_clips_tensor =tf.ones(shape=[batch_size, height, width, 3 * 2], dtype=tf.float32)
# ##test_inputs_clips_tensor =tf.random.normal(shape=[batch_size, height, width, 3 * 2], dtype=tf.float32)
# ##test_inputs_clips_np=np.random.rand(batch_size, height, width, 3 * 2).astype(np.float32)
# ##test_inputs = test_inputs_clips_np
# test_inputs = homograph_pic[129,:,:,:,:]
# print('test inputs = {}'.format(test_inputs))
# ##########  depth is not needed in the inference process, 
# ##we assign "test_depth" arbitrary values such as an all-one map
# ##test_depth = tf.ones_like(test_inputs[...,0:1])
# ##test_depth = test_inputs[...,0:1]
# ##test_depth =  tf.random.normal(shape=tf.shape(test_inputs[...,0:1]), dtype=test_inputs.dtype)
# test_depth=np.random.rand(batch_size, height, width, 1).astype(np.float32)
# print("test_depth.shape")
# train_inputs_aug = disjoint_augment_image_pair(test_inputs)
# g_loss, lp_loss, mesh_loss = compute_loss(train_inputs_aug, test_inputs, test_depth)

# cv2_imshow(cv2.resize((test_inputs[0][:,:,3:6]+1)*127.5,(64,64),interpolation = cv2.INTER_CUBIC))
# ### Convert train_inputs_aug to a NumPy array before using cv2_imshow
# train_inputs_aug_np = train_inputs_aug.numpy()  # Convert to NumPy array
# cv2_imshow(cv2.resize((train_inputs_aug_np[0][:,:,3:6]+1)*127.5,(64,64),interpolation = cv2.INTER_CUBIC))

#######################################################
##########################################

input_1_0 =  keras.layers.Input(shape=(15, 64, 64, 3))  ## input should be Alone  and uint8

def normalize2(x):
    return (tf.cast(x, tf.float32) / 255.0) - 0.0
input_1 = keras.layers.Lambda(normalize2)(input_1_0)

#output_1 = keras.layers.BatchNormalization()(input_1) ## ----
# 1111111 output_1.shape  (None, 15, 64, 64, 64)
# 1111111 output_1.shape  (None, 7, 32, 32, 64)
# 222222 output_1.shape  (None, 7, 32, 32, 128)
# 222222 output_1.shape  (None, 3, 16, 16, 128)
# 333333 output_1.shape  (None, 3, 16, 16, 256)
# 3333333 output_1.shape  (None, 3, 8, 8, 256)
# 4444444 output_1.shape  (None, 3, 8, 8, 512)
# 4444444 output_1.shape  (None, 3, 4, 4, 512)
# 555555 output_1.shape  (None, 107)
# 666666 output_2.shape  (None, 105)
# 777777 output_3.shape  (None, 48)
# 888888 output_4.shape  (None, 972)
# ok ok input_5.shape  (None, 512, 512, 6)
# 99999 outcombine.shape  (None, 256)
# 10-10-10-10 outcombine.shape  (None, 1024)
# 11-11-11-11 Decept_Detect.shape  (None, 2)


output_1 = input_1

#output_1 = keras.layers.Conv3D(64, (5, 3, 3), activation='relu', padding='same')(output_1)
output_1 = keras.layers.Conv3D(64, (5, 3, 3), padding='same')(output_1)
output_1 = keras.layers.BatchNormalization()(output_1)  ## ++++
output_1 = keras.activations.relu(output_1)  ## ++++
print('1111111 output_1.shape ',output_1.shape) ## (None, 15, 64, 64, 64)
output_1 = keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(output_1)
print('1111111 output_1.shape ',output_1.shape) ##(None, 7, 32, 32, 64)
output_1 = keras.layers.Dropout(0.3)(output_1)

#output_1 = keras.layers.Conv3D(128, (5, 3, 3), activation='relu', padding='same')(output_1)
output_1 = keras.layers.Conv3D(128, (5, 3, 3), padding='same')(output_1)
output_1 = keras.layers.BatchNormalization()(output_1)  ## ++++
output_1 = keras.activations.relu(output_1)  ## ++++
print('222222 output_1.shape ',output_1.shape) ## (None, 7, 32, 32, 128)
output_1 = keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(output_1)
print('222222 output_1.shape ',output_1.shape) ## (None, 3, 16, 16, 128)
output_1 = keras.layers.Dropout(0.3)(output_1)

#output_1 = keras.layers.Conv3D(256, (5, 3, 3), activation='relu', padding='same')(output_1)
output_1 = keras.layers.Conv3D(256, (5, 3, 3), padding='same')(output_1)
output_1 = keras.layers.BatchNormalization()(output_1)  ## ++++
output_1 = keras.activations.relu(output_1)  ## ++++
print('333333 output_1.shape ',output_1.shape) ## (None, 3, 16, 16, 256)
output_1 = keras.layers.MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(output_1)
print('3333333 output_1.shape ',output_1.shape) ## (None, 3, 8, 8, 256)
output_1 = keras.layers.Dropout(0.3)(output_1)

#output_1 = keras.layers.Conv3D(512, (5, 3, 3), activation='relu', padding='same')(output_1)
output_1 = keras.layers.Conv3D(512, (5, 3, 3), padding='same')(output_1)
output_1 = keras.layers.BatchNormalization()(output_1)  ## ++++
output_1 = keras.activations.relu(output_1)  ## ++++
print('4444444 output_1.shape ',output_1.shape) ## (None, 3, 8, 8, 512)
output_1 = keras.layers.MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(output_1)
print('4444444 output_1.shape ',output_1.shape) ## (None, 3, 4, 4, 512
output_1 = keras.layers.Dropout(0.3)(output_1)


output_1 = keras.layers.Flatten()(output_1)

#output_1 = keras.layers.Dense(1024,activation='relu',name='M_EXP')(output_1) ## fc1
output_1 = keras.layers.Dense(1024)(output_1) ## fc1
output_1 = keras.layers.BatchNormalization()(output_1)  ## ++++
output_1 = layers.Activation('relu')(output_1)
output_11 = output_1

#output_1 = keras.layers.Dense(107,activation='softmax',name='M_EXP')(output_1) ## fc2
output_1 = keras.layers.Dense(107)(output_1) ## fc2
output_1 = keras.layers.BatchNormalization()(output_1)  ## ++++
#output_1 = keras.activations.softmax(name='M_EXP')(output_1)  ## ++++
output_1 = layers.Activation('softmax',name='M_EXP')(output_1)
M_EXP = output_1  #### branching from output_1
print('555555 output_1.shape ',output_1.shape) ##  (None, 107)
# output_1 = keras.layers.Flatten()(output_1)
#output_1 = keras.layers.BatchNormalization()(output_1)   ### ?????? Can delete


input_2 =  keras.layers.Input(shape=(15,1,7))
#output_2 = keras.layers.Conv2D(32,(5,5), activation='relu')(input_2)
#output_2 = keras.layers.MaxPooling2D(pool_size=(2,2))(output_2)
output_2 = keras.layers.Flatten()(input_2)
print('666666 output_2.shape ',output_2.shape) ##
##output_2 = keras.layers.BatchNormalization()(output_2)    ###### ?????? Can delete -----


input_3 =  keras.layers.Input(shape=(6,8,1))
#output_3 = keras.layers.Conv2D(32,(5,5), activation='relu')(input_3)
#output_3 = keras.layers.MaxPooling2D(pool_size=(2,2))(output_3)
#output_3 = keras.activations.sigmoid(output_3)   ## OUTPUT RANGE === 0 to 1
#output_3 = keras.layers.BatchNormalization()(output_3)     ###(output_3)
output_3 = keras.layers.Flatten()(input_3)
print('777777 output_3.shape ',output_3.shape) ##
#output_3 = keras.layers.BatchNormalization()(output_3)     ###-----

input_4 =  keras.layers.Input(shape=(6,9,9,2))
#output_4 = keras.layers.Conv3D(64, (5, 3, 3), activation='relu', padding='same')(input_4)
#output_4 = keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(output_4)
#output_4 = keras.activations.sigmoid(output_4)   ## OUTPUT RANGE === 0 to 1
#output_4 = keras.layers.BatchNormalization()(output_4)              ###(output_4)
output_4 = keras.layers.Flatten()(input_4)
print('888888 output_4.shape ',output_4.shape) ##
#output_4 = keras.layers.BatchNormalization()(output_4)##-------

##//////////////////##????????????????????????????###////////////////////////    
################
# input_5_0 = input_1_0  ## b,15,64, 64, 3 //input should be Alone  and uint8

# # frame1_tensor = input_5_0[:][2+j*4][:,:,:] ## select frame 2,6,10,14 from 20=pack  (b ,64,64,3)
# # frame1_tensor = input_5_0[:][2+4+j*4][:,:,:] ## select frame 6,10,14,18 from 20=pack (b ,64,64,3)

# frame1_tensor = input_5_0[:,2,:,:,:] ## select frame 2, from 20=pack  (b ,64,64,3)
# frame2_tensor = input_5_0[:,2+4,:,:,:] ## select frame 6, from 20=pack (b ,64,64,3)

# # Normalization (can be done with a Lambda layer for clarity)
# def normalize1(x):
    # return (tf.cast(x, tf.float32) / 127.5) - 1.0
# frame1_normalized = keras.layers.Lambda(normalize1)(frame1_tensor)
# frame2_normalized = keras.layers.Lambda(normalize1)(frame2_tensor)
# # Resizing 
# frame1_resized = tf.keras.layers.Resizing(512,512,interpolation='bicubic')(frame1_normalized)
# frame2_resized = tf.keras.layers.Resizing(512,512,interpolation='bicubic')(frame2_normalized)
# # Concatenate the processed frames
# input_5 = keras.layers.Concatenate(axis=3)([frame1_resized, frame2_resized]) ## (b=3,512,512,6)
# #########################

# print('ok ok input_5.shape ',input_5.shape)  ## b=3,512, 512, 6
# batch_size = input_5.shape[0]
# print('batch_size ========',batch_size)

# #test_depth=np.random.rand( 512, 512, 1).astype(np.float32)   ### 3,512, 512, 6
# #train_inputs_aug = disjoint_augment_image_pair(input_5)

# augment_layer = AugmentLayer()
# train_inputs_aug = augment_layer(input_5)

# H_model_1 = H_model(train_inputs_aug, input_5, input_5[...,0:1],patch_size=512.)  
# [H1_motion, H2_motion, mesh_motion,train_warp2_depth, train_mesh, train_warp2_H1, train_warp2_H2,train_warp2_H3, train_one_warp_H1, train_one_warp_H2,train_one_warp_H3] = H_model_1([train_inputs_aug, input_5, input_5[...,0:1]])
# #H1_motion, H2_motion, mesh_motion,train_warp2_depth, train_mesh, train_warp2_H1, train_warp2_H2,train_warp2_H3, train_one_warp_H1, train_one_warp_H2,train_one_warp_H3 = H_model(train_inputs_aug, input_5, input_5[...,0:1], patch_size=512.)

# l_num = 1.
# lam_lp = 1.

# abs_diff_layer = AbsoluteDifferenceLayer()

# # [gen_frames=train_warp2_H1, gt_frames=input_5[...,0:3]*train_one_warp_H1])
# # [gen_frames=train_warp2_H2, gt_frames=input_5[...,0:3]*train_one_warp_H2])
# # [gen_frames=train_warp2_H3, gt_frames=input_5[...,0:3]*train_one_warp_H3])

# # loss1 = abs_diff_layer[(train_warp2_H1, input_5[...,0:3]*train_one_warp_H1)]
# # loss2 = abs_diff_layer[(train_warp2_H2, input_5[...,0:3]*train_one_warp_H2)]
# # loss3 = abs_diff_layer[(train_warp2_H3, input_5[...,0:3]*train_one_warp_H3]
# # lp_loss = 16. * loss1 + 4. * loss2 + 1. * loss3
# # g_loss = lp_loss * lam_lp 

# def loss_operation(train_warp2_H1, input_5, train_one_warp_H1,
                     # train_warp2_H2, train_one_warp_H2,
                     # train_warp2_H3, train_one_warp_H3):

    # loss1 = train_warp2_H1 - input_5[..., 0:3] * train_one_warp_H1
    # loss2 = train_warp2_H2 - input_5[..., 0:3] * train_one_warp_H2
    # loss3 = train_warp2_H3 - input_5[..., 0:3] * train_one_warp_H3

    # lp_loss = 16.0 * (loss1*loss1) + 4.0 * (loss2*loss2) + 1.0 * (loss3*loss3)
    # lam_lp = 1.
    # g_loss000 = lp_loss * lam_lp #(3=b, 512, 512, 3)
    # # Sum along axes 1, 2, and 3 using tf.reduce_sum()
    # g_loss00 = tf.reduce_sum(g_loss000, axis=(1, 2, 3)) ##Shape'g_loss00' (after summing axes 1, 2, 3): (b=3,)

    # return g_loss00

# # Create a Lambda layer to perform the custom operation
# tt_layer = layers.Lambda(lambda tensors: loss_operation(tensors[0], tensors[1], tensors[2],
                                                           # tensors[3], tensors[4], tensors[5],
                                                           # tensors[6]),
                                                 # )(
    # [train_warp2_H1, input_5, train_one_warp_H1,
     # train_warp2_H2, train_one_warp_H2,
     # train_warp2_H3, train_one_warp_H3]
# )
# g_loss = tt_layer

# # loss1 = train_warp2_H1 - input_5[...,0:3]*train_one_warp_H1
# # loss2 = train_warp2_H2 - input_5[...,0:3]*train_one_warp_H2
# # loss3 = train_warp2_H3 - input_5[...,0:3]*train_one_warp_H3
# # lp_loss = 16. * loss1* loss1 + 4. * loss2* loss2 + 1. * loss3* loss3
# # g_loss = lp_loss * lam_lp 

# print('g_loss.shape======',g_loss.shape,g_loss)  ##(3=b,) , dtype=float32, sparse=True, name=keras_tensor_604>
# #g_loss = keras.layers.Flatten()(g_loss)  ## xx=None*512*512*3 --- (3=b, 786432)
# ####### ????? g_loss = keras.activations.sigmoid(g_loss)   ## OUTPUT RANGE === 0 to 1 !!!!!!!!! use tanh 
# g_loss = keras.layers.Flatten(name='g_loss')(g_loss)  ##  only fo naming name='g_loss' for plot acc , loss xx=None*512*512*3 --- (3, 786432)
# print('g_loss.shape======',g_loss.shape,g_loss)   ### (3=b,)
# #g_loss = layers.Embedding( sparse=True)(g_loss)
# out01 = g_loss
        
# output_5 = keras.layers.Flatten()(mesh_motion)
# mesh = keras.layers.Dense(1,activation='sigmoid',name='mesh')(output_5)
# print('mesh.shape ========',mesh) #<KerasTensor shape=(3, 1), dtype=float32, sparse=False, name=keras_tensor_606>



#out00 = np.random.rand(1).astype(np.float32)
#output_5 = keras.layers.Flatten()(input_5)
#out00 = keras.layers.Dense(1,activation='softmax',name='out00')(output_5)
##//////////////////##????????????????????????????###////////////////////////

#out12345 = [output_1,output_2,output_3,output_4,output_5]
out12345 = [output_11,M_EXP,output_2,output_3,output_4]
outcombine = keras.layers.concatenate(out12345)

#outcombine = keras.layers.Dense(256,activation='relu')(outcombine)  ## 815   ---> 256 fc1
outcombine = keras.layers.Dropout(0.2)(outcombine)
outcombine = keras.layers.Dense(256)(outcombine)  ## 815   ---> 256 fc1
outcombine = keras.layers.BatchNormalization()(outcombine)  ## ++++
outcombine = keras.activations.relu(outcombine)  ## ++++
print('99999 outcombine.shape ',outcombine.shape) ##



#outcombine = keras.layers.Dense(1024,activation='relu')(outcombine)  ## 256   --->  1024 Hidden_Layer
outcombine = keras.layers.Dropout(0.2)(outcombine)
outcombine = keras.layers.Dense(1024)(outcombine)  ## 815   ---> 256 fc1
outcombine = keras.layers.BatchNormalization()(outcombine)  ## ++++
outcombine = keras.activations.relu(outcombine)  ## ++++
print('10-10-10-10 outcombine.shape ',outcombine.shape) ##


#Decept_Detect = keras.layers.Dense(2,activation='softmax',name='Decept_Detect')(outcombine)## fc2
outcombine = keras.layers.Dropout(0.2)(outcombine)
outcombine = keras.layers.Dense(2)(outcombine)  ## 815   ---> 256 fc1
outcombine = keras.layers.BatchNormalization()(outcombine)  ## ++++
#Decept_Detect = keras.activations.softmax(name='Decept_Detect')(outcombine)  ## ++++
Decept_Detect = layers.Activation('softmax',name='Decept_Detect')(outcombine)
print('11-11-11-11 Decept_Detect.shape ',Decept_Detect.shape) ##

  
#MHEmodel = keras.models.Model([input_1_0,input_2,input_3,input_4],[M_EXP,Decept_Detect,mesh,g_loss])
MHEmodel = keras.models.Model([input_1_0,input_2,input_3,input_4],[M_EXP,Decept_Detect])


keras.utils.plot_model(MHEmodel,
    to_file="model1.jpg", ##to_file="model.png"
    show_shapes=True,
    show_dtype=True,
    show_layer_names=True,
    expand_nested=True,
    show_layer_activations=True,
    show_trainable=True)
MHEmodel.summary()
print('len(MHEmodel.layers)===',len(MHEmodel.layers))    

#############################################################################
#############################################################################
###############################  part 5 ######################################
#############################################################################
#################################### load inputs && outputs  ###############
#all_video_np=np.random.randn(124,20, 112, 112, 3).astype(np.float32)
#### Train Data
all_video_np=np.load('/content/gdrive/My Drive/train_30clip_video_np.npy')
all_video_label_01_np=np.load('/content/gdrive/My Drive/train_30clip_label_01_np.npy')
micro_exp_labels=np.load('/content/gdrive/My Drive/train_30clip_micro_exp_labels.npy')
Emotion_label_np=np.load('/content/gdrive/My Drive/train_6464_30clip_Emotion_label_np.npy')
homograph_label_H1_np=np.load('/content/gdrive/My Drive/train_30clip_homograph_label_H1_np.npy')
homograph_label_MESH_np=np.load('/content/gdrive/My Drive/train_30clip_homograph_label_MESH_np.npy')

#homograph_pic = np.load('/content/gdrive/My Drive/homograph_pic_np.npy')
#### homograph_pic.shape (None, 3, 512, 512, 6) <class 'tuple'> 5
## normalize_image(frame2 / 127.5) - 1.0

#### Val  Data  ---- or  test  Data
val_vid_np =np.load('/content/gdrive/My Drive/val_30clip_video_np.npy')
val_vid_label_01_np=np.load('/content/gdrive/My Drive/val_30clip_label_01_np.npy')
val_mic_exp_labels=np.load('/content/gdrive/My Drive/val_30clip_micro_exp_labels.npy')
val_Emo_label_np=np.load('/content/gdrive/My Drive/val_6464_30clip_Emotion_label_np.npy')
val_homo_label_H1_np=np.load('/content/gdrive/My Drive/val_30clip_homograph_label_H1_np.npy')
val_homo_label_MESH_np=np.load('/content/gdrive/My Drive/val_30clip_homograph_label_MESH_np.npy')

#######################################################################################################################
#######################################################################################################################
#################################### 22222 shuffling splited videos train  ###################################################

zipped = zip(all_video_np,all_video_label_01_np, micro_exp_labels,Emotion_label_np,homograph_label_H1_np,homograph_label_MESH_np)

zip_splited_videos = tuple(zipped)
l = list(zip_splited_videos)
random.shuffle(l) 
zip_splited_videos=tuple(l)
(all_video_np_sh,all_video_label_01_np_sh, micro_exp_labels_sh,Emotion_label_np_sh,homograph_label_H1_np_sh,homograph_label_MESH_np_sh)=zip(*zip_splited_videos)

all_video_np_sh = np.asarray(all_video_np_sh).astype(np.uint8) ##(None, 20, 64, 64, 3) less RAM
all_video_label_01_np_sh = np.asarray(all_video_label_01_np_sh).astype(np.float32) ##(None, 2) 
micro_exp_labels_sh = np.asarray(micro_exp_labels_sh).astype(np.float32) ##(None, 107) LIST TO ARRAY FLOAT32
Emotion_label_np_sh =  np.asarray(Emotion_label_np_sh).astype(np.float32) ##(None, 3, 1, 7) 
homograph_label_H1_np_sh =  np.asarray(homograph_label_H1_np_sh).astype(np.float32)##(None, 3, 8, 1) 
homograph_label_MESH_np_sh =  np.asarray(homograph_label_MESH_np_sh).astype(np.float32) ##(None, 3, 9, 9, 2) 
#homograph_pic_sh =  np.asarray(homograph_pic_sh).astype(np.uint8) ##(None, 3, 512, 512, 6) less RAM
#print('all_video_label_01_np_sh.shape',all_video_label_01_np_sh.shape,all_video_label_01_np_sh[0:300])

# #################################### 22222 shuffling splited videos val  ###################################################

zipped = zip(val_vid_np,val_vid_label_01_np, val_mic_exp_labels,val_Emo_label_np,val_homo_label_H1_np,val_homo_label_MESH_np)

zip_splited_videos = tuple(zipped)
l = list(zip_splited_videos)
random.shuffle(l) 
zip_splited_videos=tuple(l)
(val_vid_np_sh,val_vid_label_01_np_sh, val_mic_exp_labels_sh,val_Emo_label_np_sh,val_homo_label_H1_np_sh,val_homo_label_MESH_np_sh)=zip(*zip_splited_videos)

val_vid_np_sh = np.asarray(val_vid_np_sh).astype(np.uint8) ##(None, 20, 64, 64, 3) less RAM
val_vid_label_01_np_sh = np.asarray(val_vid_label_01_np_sh).astype(np.float32) ##(None, 2) 
val_mic_exp_labels_sh = np.asarray(val_mic_exp_labels_sh).astype(np.float32) ##(None, 107) LIST TO ARRAY FLOAT32
val_Emo_label_np_sh =  np.asarray(val_Emo_label_np_sh).astype(np.float32) ##(None, 3, 1, 7) 
val_homo_label_H1_np_sh =  np.asarray(val_homo_label_H1_np_sh).astype(np.float32)##(None, 3, 8, 1) 
val_homo_label_MESH_np_sh =  np.asarray(val_homo_label_MESH_np_sh).astype(np.float32) ##(None, 3, 9, 9, 2) 
#homograph_pic_sh =  np.asarray(homograph_pic_sh).astype(np.uint8) ##(None, 3, 512, 512, 6) less RAM
#print('val_vid_label_01_np_sh.shape',val_vid_label_01_np_sh.shape,val_vid_label_01_np_sh[0:300])


############################  transfer outputs in range [0 to 1] linear  ###################
#### Train Data
max_val_H1 = np.max(homograph_label_H1_np_sh)
min_val_H1 = np.min(homograph_label_H1_np_sh)
homograph_label_H1_np_sh = (homograph_label_H1_np_sh - min_val_H1) / (max_val_H1 - min_val_H1 + keras.backend.epsilon())
print('max_val_H1,min_val_H1====',max_val_H1 ,min_val_H1) 

max_val_MESH = np.max(homograph_label_MESH_np_sh)
min_val_MESH = np.min(homograph_label_MESH_np_sh)
homograph_label_MESH_np_sh = (homograph_label_MESH_np_sh - min_val_MESH) / (max_val_MESH - min_val_MESH + keras.backend.epsilon())
print('max_val_MESH,min_val_MESH====',max_val_MESH,min_val_MESH)
##max_val_H1,min_val_H1==== 414.92575 -356.62973
##max_val_MESH,min_val_MESH==== 64.38657 -65.6642

# #max_val_H1,min_val_H1==== 429.20575 -398.6831
# #max_val_MESH,min_val_MESH==== 62.75726 -61.36485

# #max_val_H1,min_val_H1==== 338.85938 -401.8903
# #max_val_MESH,min_val_MESH==== 68.732445 -45.588158

# print('homograph_label_H1_np[0] ,homograph_label_MESH_np[0]' ,
    # homograph_label_H1_np[0] ,homograph_label_MESH_np[0])
    
# ####################################### Val Data
# max_val_H1 = np.max(val_homo_label_H1_np_sh)
# min_val_H1 = np.min(val_homo_label_H1_np_sh)
val_homo_label_H1_np_sh = (val_homo_label_H1_np_sh - min_val_H1) / (max_val_H1 - min_val_H1 + keras.backend.epsilon())
# print('2 max_val_H1,2 min_val_H1====',max_val_H1 ,min_val_H1) 

# max_val_MESH = np.max(val_homo_label_MESH_np_sh)
# min_val_MESH = np.min(val_homo_label_MESH_np_sh)
val_homo_label_MESH_np_sh = (val_homo_label_MESH_np_sh - min_val_MESH) / (max_val_MESH - min_val_MESH + keras.backend.epsilon())
# print('2max_val_MESH,2min_val_MESH====',max_val_MESH,min_val_MESH)

# # 2 max_val_H1,2 min_val_H1==== 272.69824 -201.65582
# # 2max_val_MESH,2min_val_MESH==== 49.775505 -50.316322

# print('val_homo_label_H1_np_sh[0] ,val_homo_label_MESH_np_sh[0]' ,
    # val_homo_label_H1_np_sh[0] ,val_homo_label_MESH_np_sh[0])
    
print('Train max_H1,min_H1====',np.max(homograph_label_H1_np_sh) ,np.min(homograph_label_H1_np_sh)) 
print('Train max_MESH,min_MESH====',np.max(homograph_label_MESH_np_sh) ,np.min(homograph_label_MESH_np_sh))  

print('validation max_H1,min_H1====',np.max(val_homo_label_H1_np_sh) ,np.min(val_homo_label_H1_np_sh)) 
print('validation max_MESH,min_MESH====',np.max(val_homo_label_MESH_np_sh) ,np.min(val_homo_label_MESH_np_sh)) 

    
#########################################################################################################
#############################  make label for homo_out ,homo_out_1   ****
######################################################################################### 

#### Train Data
bb = all_video_np_sh.shape[0]  ### batc size ==== b
homo_out = np.zeros(bb,).astype(np.float32)##(None,)(130,)
print('homo_out.shape',homo_out.shape)  ## (b=130,) <class 'numpy.ndarray'>

homo_out_1 = homo_out   ###  (b=130,) <class 'numpy.ndarray'>
# xx=512*512*3
# homo_out_1 = np.zeros([bb,xx]).astype(np.float32)##homo_out_1.shape (130, 786432)
# print('homo_out_1.shape',homo_out_1.shape)  ## (30, 786432)


print('all_video_label_01_np.shape',all_video_label_01_np.shape
        ,'micro_exp_labels.shape',micro_exp_labels.shape
        ,'homo_out.shape',homo_out.shape
        ,'homo_out_1.shape',homo_out_1.shape
        ,'all_video_np.shape',all_video_np.shape)
# homo_out.shape (3002,)
# all_video_label_01_np_sh.shape (3002, 2)
# micro_exp_labels_sh.shape (3002, 107) 
# homo_out.shape (3002,) homo_out_1.shape (3002,)
# all_video_np_sh.shape (3002, 20, 64, 64, 3)

######################### VAL Data
bb2 = val_vid_np_sh.shape[0]  ### batc size ==== b
val_homo_out = np.zeros(bb2,).astype(np.float32)##(None,)(130,)
val_homo_out_1 = val_homo_out   ###  (b=353,) shape (353,)

print('val_homo_out.shape',val_homo_out.shape
        ,'val_homo_out_1.shape',val_homo_out_1.shape
        ,'val_mic_exp_labels_sh.shape',val_mic_exp_labels_sh.shape
        ,'val_vid_np_sh.shape',val_vid_np_sh.shape
        ,'val_vid_label_01_np_sh.shape',val_vid_label_01_np_sh.shape)  ## (b=353,) val_homo_out.shape (353,)


# all_video_label_01_np.shape (5300, 2) micro_exp_labels.shape (5300, 107) homo_out.shape (5300,) homo_out_1.shape (5300,) all_video_np.shape (5300, 15, 64, 64, 3)
# val_homo_out.shape (196,) val_homo_out_1.shape (196,) val_mic_exp_labels_sh.shape (196, 107) val_vid_np_sh.shape (196, 15, 64, 64, 3) val_vid_label_01_np_sh.shape (196, 2)

##############################################################
#############################  Releasing RAM 1 '''''   
###############################################################
# zip_splited_videos = 0
# zipped = 0
# l = 0

# #homograph_pic = 0
# #homograph_pic_sh =  0
# all_video_np= []
# Emotion_label_np= []
# homograph_label_H1_np= []
# homograph_label_MESH_np=[]
# all_video_label_01_np= []
# micro_exp_labels= []

# val_vid_np = []
# val_vid_label_01_np = []
# val_mic_exp_labels = []
# val_Emo_label_np = []
# val_homo_label_H1_np = []
# val_homo_label_MESH_np = []

###########################################################################
###########################################################################
###############################  part 6 ###################################
###########################################################################
######################################### testing model ##################

##############    this block Can do not execute for redusing RAM
# a0=np.expand_dims( all_video_np_sh[0], axis=0)
# a1=np.expand_dims( Emotion_label_np_sh[0], axis=0)
# a2=np.expand_dims( homograph_label_H1_np_sh[0], axis=0)
# a3=np.expand_dims( homograph_label_MESH_np_sh[0], axis=0)
#a4=np.expand_dims( homograph_pic_sh[0][0], axis=0)  ## note [0][0]

a0= all_video_np[0:3] 
a1= Emotion_label_np[0:3] 
a2= homograph_label_H1_np[0:3] 
a3= homograph_label_MESH_np[0:3] 
#a4 = homograph_pic_sh_1[0:3] 

#pred1,pred2,pred3,loss10 = MHEmodel.predict([a0,a1,a2,a3])  ## 4 == NUM of input MHEmodel
pred1,pred2 = MHEmodel.predict([a0,a1,a2,a3])  ## 4 == NUM of input MHEmodel

print ('testing model shape M_EXP,Decept_Detect,mesh ====',pred1.shape,pred2.shape)  ##== (3, 107) (3, 2) (3, 1)
#print ('testing model g_loss , homo _mesh ====',loss10.shape,pred3.shape)  ## (3, 786432)
## (3, 107) (3, 2) ()  ##  shape=(3, 1), dtype=float32)
#print('testing model mesh,g_loss ====',pred3.shape,loss10.shape,pred3,loss10)  ## tf.Tensor(1.2502934, shape=(), dtype=float32) only one num.
print('testing model Decept_Detect ====',pred2)
print('testing model Micro_Exp ====',pred1)
# tf.Tensor(
# [[0.50505805]
 # [0.507365  ]
 # [0.5066267 ]], shape=(3, 1), dtype=float32) tf.Tensor(1.76392, shape=(), dtype=float32)
## (3, 786432)
#############################  Releasing RAM 2'''''
a0=[]
a1=[]
a2=[]
a3=[]
#############################################################################
#############################################################################
###############################  part 7 ######################################
###############################################################################
############## define epochs ,batch_size ,train_in_data train_out_data val_in_data val_out_data
#######################
epochs =20# 7 # 10 #30 #20 #10 #2  ##50
batch_size = 3  ##4  ##2
pack=15  ##20  ## 144
l1=len(micro_exp_labels)   ### len all dete None== 124
print('number of all batchs====== {} , '.format(l1))
print('number of frames in one  batch ====== {} , '.format(pack))

###################### train data  5 input &  4 Y_true out
#keras.layers.Concatenate(axis=0)([homograph_label_H1_np0,homograph_label_H1_np2])
#np.concatenate([xx0, yy0], axis=2)
# train_in_data =[np.concatenate([all_video_np[0:(1482-161)],all_video_np[1482:(3002-154)]])
  # ,np.concatenate([Emotion_label_np[0:(1482-161)],Emotion_label_np[1482:(3002-154)]])
  # ,np.concatenate([homograph_label_H1_np[0:(1482-161)],homograph_label_H1_np[1482:(3002-154)]])
  # ,np.concatenate([homograph_label_MESH_np[0:(1482-161)],homograph_label_MESH_np[1482:(3002-154)]])
  # ]

# train_out_data = [np.concatenate([micro_exp_labels[0:(1482-161)],micro_exp_labels[1482:(3002-154)]])
  # ,np.concatenate([all_video_label_01_np[0:(1482-161)],all_video_label_01_np[1482:(3002-154)]])
  # ,np.concatenate([homo_out[0:(1482-161)],homo_out[1482:(3002-154)]])
  # ,np.concatenate([homo_out_1[0:(1482-161)],homo_out_1[1482:(3002-154)]])
  # ] 
train_in_data =[all_video_np_sh,Emotion_label_np_sh
                ,homograph_label_H1_np_sh,homograph_label_MESH_np_sh
                ]

train_out_data = [micro_exp_labels_sh,all_video_label_01_np_sh
                ]   # ,homo_out,homo_out_1]               

print ('NUM of train_in_data====',len(train_in_data),len(train_in_data[0]),train_in_data[0].shape,train_in_data[1].shape
            ,train_in_data[2].shape,train_in_data[3].shape)

print ('NUM of train_out_data====',len(train_out_data),train_out_data[0].shape,train_out_data[1].shape
            )   ## ,train_out_data[2].shape)

### testig train_in_data ,, train_out_data
        
# data0 = 1482-161-1
# fac_pic = cv2.resize(train_in_data[0][data0][0][:,:,:],(128,128),interpolation = cv2.INTER_CUBIC)##(0,3002, 20, 64, 64, 3)
# cv2_imshow(fac_pic)
# print ('Micri Exp  , decept det labels====',train_out_data[0][data0],train_out_data[1][data0],
       # len(train_out_data[0][data0]),
       # len(train_out_data[1][data0]),
       # train_out_data[0][data0].shape,
       # train_out_data[1][data0].shape

        # )##(0,3002, 107) (1,3002, 2)
                    
### testig train_in_data ,, train_out_data
# data0 =1673 #2686 #1482-161
# fac_pic = cv2.resize(train_in_data[0][data0][0][:,:,:],(128,128),interpolation = cv2.INTER_CUBIC)##(0,3002, 20, 64, 64, 3)
# cv2_imshow(fac_pic)
# print ('Micri Exp  , decept det labels====',train_out_data[0][data0],train_out_data[1][data0],
       # len(train_out_data[0][data0]),
       # len(train_out_data[1][data0]),
       # train_out_data[0][data0].shape,
       # train_out_data[1][data0].shape

        # )##(0,3002, 107) (1,3002, 2)
        
###################### validation  data 5 in &  4 out 
# val_in_data =[np.concatenate([all_video_np[(1482-161):1482],all_video_np[(3002-154):3002]])
   # ,np.concatenate([Emotion_label_np[(1482-161):1482],Emotion_label_np[(3002-154):3002]])
   # ,np.concatenate([homograph_label_H1_np[(1482-161):1482],homograph_label_H1_np[(3002-154):3002]])
   # ,np.concatenate([homograph_label_MESH_np[(1482-161):1482],homograph_label_MESH_np[(3002-154):3002]])
   # ]
# val_out_data = [np.concatenate([micro_exp_labels[(1482-161):1482],micro_exp_labels[(3002-154):3002]])
   # ,np.concatenate([all_video_label_01_np[(1482-161):1482],all_video_label_01_np[(3002-154):3002]])
   # ,np.concatenate([homo_out[(1482-161):1482],homo_out[(3002-154):3002]])
   # ,np.concatenate([homo_out_1[(1482-161):1482],homo_out_1[(3002-154):3002]])
   # ]


val_in_data =[val_vid_np_sh,val_Emo_label_np_sh
                ,val_homo_label_H1_np_sh,val_homo_label_MESH_np_sh
                ]
val_out_data = [val_mic_exp_labels_sh,val_vid_label_01_np_sh
                ]  ### ,val_homo_out,val_homo_out_1]

print ('val_in_data====',len(val_in_data),len(val_in_data[0]))
print ('val_out_data====',len(val_out_data),len(val_out_data[0]))

# data0 =304 #2686 #1482-161
# fac_pic = cv2.resize(val_in_data[0][data0][0][:,:,:],(128,128),interpolation = cv2.INTER_CUBIC)##(0,3002, 20, 64, 64, 3)
# cv2_imshow(fac_pic)
# print ('Micri Exp  , decept det labels====',val_out_data[0][data0],val_out_data[1][data0],
       # len(val_out_data[0][data0]),
       # len(val_out_data[1][data0]),
       # val_out_data[0][data0].shape,
       # val_out_data[1][data0].shape

        # )##(0,3002, 107) (1,3002, 2)
        
# number of all batchs====== 3002 , 
# number of frames in one  batch ====== 20 , 
# NUM of train_in_data==== 4 3002 (3002, 20, 64, 64, 3) (3002, 4, 1, 7) (3002, 4, 8, 1) (3002, 4, 9, 9, 2)
# NUM of train_out_data==== 4 (3002, 107) (3002, 2) (3002,)
# val_in_data==== 4 353
# val_out_data==== 4 353
###########################################
#############################  Releasing RAM 3'''''   

zip_splited_videos = 0
zipped = 0
l = 0

#homograph_pic = 0
#homograph_pic_sh =  0
all_video_np= []
Emotion_label_np= []
homograph_label_H1_np= []
homograph_label_MESH_np=[]
all_video_label_01_np= []
micro_exp_labels= []

val_vid_np = []
val_vid_label_01_np = []
val_mic_exp_labels = []
val_Emo_label_np = []
val_homo_label_H1_np = []
val_homo_label_MESH_np = []

#homograph_pic = 0
#homograph_pic_sh =  0
all_video_np_sh = []
Emotion_label_np_sh = []
homograph_label_H1_np_sh = []
homograph_label_MESH_np_sh =[] 

micro_exp_labels_sh = []
all_video_label_01_np_sh = []
homo_out = []
homo_out_1 = []

val_vid_np_sh = []
val_Emo_label_np_sh = []
val_homo_label_H1_np_sh = []
val_homo_label_MESH_np_sh = []

val_mic_exp_labels_sh = []
val_vid_label_01_np_sh = []
val_homo_out = []
val_homo_out_1 = []

##############################################################################
#############################################################################
#############################################################################
###############################  part 8 ######################################
#############################################################################
###############################  Training ######################################

### ACC ###
checkpoint_filepath = pathgd+'best_model/HME23_EMO15_46_E_{epoch:02d}_val_loss_{val_loss:.4f}_val_ACC_{val_Decept_Detect_categorical_accuracy:.4f}.weights.h5' 

####  ROC_AUC ####
##checkpoint_filepath = pathgd+'best_model/HME23_EMO15_E_{epoch:02d}_val_loss_{val_loss:.4f}_val_AUC_ROC_{val_Decept_Detect_auc_1:.4f}.weights.h5' 


# The model (that are considered the best) can be loaded as -
#MHEmodel = keras.models.load_model(checkpoint_filepath)

#Loading the best model from previous Training
#MHEmodel.load_weights(pathgd+'best_model/Epoach_07_val_loss_0.1330.MHEmodel_22.weights.h5')
#MHEmodel.load_weights(pathgd+'best_model/Epoach_01_val_loss_0.0610.MHEmodel_22.weights.h5')                     

MHEmodel.load_weights(pathgd+'best_model/HME23_EMO15_46_E_10_val_loss_1.1626_val_ACC_0.8387.weights.h5')

############################################### keras.callbacks  defining
# Model is saved at the end of every epoch, if it's the best seen so far.
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss', mode='min',
    #monitor='val_ACC', mode='max',
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
    )

model_early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    #monitor='val_ACC',
    patience= 20, #8, #10,# Number of epochs with no improvement after which training will be stopped.
    restore_best_weights=True,# Restores model weights from the epoch with the best value of the monitored quantity.monitor='loss',
    verbose = 1,
    )

# reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
                          # patience=5, min_lr=0.00002)
# csv_logger = CSVLogger('history.log')

# . Define the learning rate scheduling function for exponential decay
import math
def exponential_decay_schedule(epoch, lr):
    #initial_lr = 0.0001   #0.0001 #  
    k = 0.1 # Decay factor. Higher k means faster decay.
    # Note: epoch is 0-indexed, so for epoch 0, decay factor is e^0 = 1, LR stays initial_lr
    #new_lr = initial_lr * math.exp(-k * (epoch)) ##  - decreasing lr 1,0.9,0.8,... + increasing 1.1,1.2,1.3
    new_lr = lr * math.exp(-k * epoch)  ##  - decreasing lr 1,0.9,0.8,... + increasing 1.1,1.2,1.3
    print(f"Last MHEmodel save_weights &&& Epoch {epoch+1}: Current LR = {lr:.8f}, New LR = {new_lr:.8f}")
    MHEmodel.save_weights((pathgd+'best_model/HME23_EMO15_46_END_Epoch_{epoch:02d}.weights.h5'))
    return new_lr

# . Create the LearningRateScheduler callback
lr_scheduler_callback_exp = tf.keras.callbacks.LearningRateScheduler(
    exponential_decay_schedule,
    verbose=1 # This will print the messages from our function
)
################################################################################
MHEmodel.compile(
    #optimizer=keras.optimizers.Adam(3e-4),
    optimizer = keras.optimizers.Adam(learning_rate=0.00000498), ##0.0001 0.00002466  0.0001/2#0.00001506 ## base rate=0.0001  0.00005 0.00001  ##0.00002744
    #loss=keras.losses.categorical_crossentropy,## binary_crossentropy  None  ##[M_EXP,Decept_Detect,mash,g_loss]
    #loss = [None,None,None ,'categorical_crossentropy'], ## 'mean_squared_error'
    #loss = ['categorical_crossentropy','categorical_crossentropy', 'binary_crossentropy','categorical_crossentropy'],
    #loss = ['categorical_crossentropy','categorical_crossentropy', 'binary_crossentropy','mean_squared_error'],
    #loss = ['categorical_crossentropy','categorical_crossentropy', None,None],
    loss = ['categorical_crossentropy','categorical_crossentropy'],
    #loss_weights=[.3, .7, .0, .0],  ##[.2, .6, .1, .1]
    loss_weights=[.4, .6],  ##
    
    ### ACC ###
    #metrics=[None,None,None,'accuracy']
    #metrics=['accuracy','accuracy','accuracy','accuracy']
    #metrics=['categorical_accuracy','categorical_accuracy','binary_accuracy','binary_accuracy'],
    #metrics=[None,'categorical_accuracy',None,None],
    metrics=['categorical_accuracy','categorical_accuracy'],
    
    ####  ROC_AUC ####
    #metrics=[keras.metrics.AUC(),'categorical_accuracy',None,None]
    #metrics=[keras.metrics.AUC(multi_label=True, num_labels=107),keras.metrics.AUC(multi_label=True, num_labels=2),None,None]
    ##metrics=[keras.metrics.AUC(multi_label=True, num_labels=107),keras.metrics.AUC(multi_label=True, num_labels=2)]
    
    #weighted_metrics= [['accuracy'], ['mse']]) ,
    #weighted_metrics= [[['categorical_accuracy'], ['categorical_accuracy'] , ['binary_accuracy'] , ['binary_accuracy']])] ,
    )
    
model_info=MHEmodel.fit(
    train_in_data,
    train_out_data,
    epochs=epochs,
    batch_size=batch_size,
    verbose=1,shuffle=True,
    #callbacks=[callbacks],
    callbacks=[model_checkpoint_callback,model_early_stopping,lr_scheduler_callback_exp],
    validation_split=0.2,  ## splite train data for val data if we want !!!!
    #validation_data=(val_in_data,val_out_data)
    )

#MHEmodel.save_weights((pathgd+'MHEmodel021.weights.h5'))  ### note emotion model.h5 new model exist
plot_model_history(model_info)

##############################################################################
#############################################################################
#############################################################################
###############################  part 9 ######################################
#############################################################################
###############################  LOGGING ######################################


dense_4 (Dense)     │ (None, 2)         │      2,050 │ dropout_6[0][0]   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (None, 2)         │          8 │ dense_4[0][0]     │
│ (BatchNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ Decept_Detect       │ (None, 2)         │          0 │ batch_normalizat… │
│ (Activation)        │                   │            │                   │
└─────────────────────┴───────────────────┴────────────┴───────────────────┘
 Total params: 33,884,065 (129.26 MB)
 Trainable params: 33,877,319 (129.23 MB)
 Non-trainable params: 6,746 (26.35 KB)
len(MHEmodel.layers)=== 42

number of all batchs====== 5300 , 
number of frames in one  batch ====== 15 , 
NUM of train_in_data==== 4 5300 (5300, 15, 64, 64, 3) (5300, 15, 1, 7) (5300, 6, 8, 1) (5300, 6, 9, 9, 2)
NUM of train_out_data==== 2 (5300, 107) (5300, 2)
val_in_data==== 4 196
val_out_data==== 2 196

  
1414/1414 ━━━━━━━━━━━━━━━━━━━━ 7577s 5s/step - Decept_Detect_categorical_accuracy: 0.5173 - Decept_Detect_loss: 0.8446 - M_EXP_categorical_accuracy: 0.0251 - M_EXP_loss: 4.8452 - loss: 2.4449 - val_Decept_Detect_categorical_accuracy: 0.6425 - val_Decept_Detect_loss: 0.6561 - val_M_EXP_categorical_accuracy: 0.0340 - val_M_EXP_loss: 5.1587 - val_loss: 2.4582 - learning_rate: 2.4660e-05
Epoch 2: Current LR = 0.00002466, New LR = 0.00002231
1414/1414 ━━━━━━━━━━━━━━━━━━━━ 7552s 5s/step - Decept_Detect_categorical_accuracy: 0.5575 - Decept_Detect_loss: 0.6967 - M_EXP_categorical_accuracy: 0.0576 - M_EXP_loss: 4.2764 - loss: 2.1286 - val_Decept_Detect_categorical_accuracy: 0.4877 - val_Decept_Detect_loss: 0.8523 - val_M_EXP_categorical_accuracy: 0.0094 - val_M_EXP_loss: 4.8868 - val_loss: 2.4674 - learning_rate: 2.2313e-05
Epoch 3: Current LR = 0.00002231, New LR = 0.00001827
1414/1414 ━━━━━━━━━━━━━━━━━━━━ 7454s 5s/step - Decept_Detect_categorical_accuracy: 0.5912 - Decept_Detect_loss: 0.6627 - M_EXP_categorical_accuracy: 0.0639 - M_EXP_loss: 4.1367 - loss: 2.0523 - val_Decept_Detect_categorical_accuracy: 0.4821 - val_Decept_Detect_loss: 0.8573 - val_M_EXP_categorical_accuracy: 0.0236 - val_M_EXP_loss: 4.7279 - val_loss: 2.4075 - learning_rate: 1.8269e-05
Epoch 4: Current LR = 0.00001827, New LR = 0.00001353


 ################################################################
Epoch 1: Current LR = 0.00010000, New LR = 0.00010000
1414/1414 ━━━━━━━━━━━━━━━━━━━━ 7310s 5s/step - Decept_Detect_categorical_accuracy: 0.5804 - Decept_Detect_loss: 0.6780 - M_EXP_categorical_accuracy: 0.0710 - M_EXP_loss: 4.1584 - loss: 2.0702 - val_Decept_Detect_categorical_accuracy: 0.6566 - val_Decept_Detect_loss: 0.5846 - val_M_EXP_categorical_accuracy: 0.2358 - val_M_EXP_loss: 3.1598 - val_loss: 1.6133 - learning_rate: 1.0000e-04
Epoch 2: Current LR = 0.00010000, New LR = 0.00009048
1414/1414 ━━━━━━━━━━━━━━━━━━━━ 7325s 5s/step - Decept_Detect_categorical_accuracy: 0.6057 - Decept_Detect_loss: 0.6449 - M_EXP_categorical_accuracy: 0.0889 - M_EXP_loss: 3.9075 - loss: 1.9500 - val_Decept_Detect_categorical_accuracy: 0.7057 - val_Decept_Detect_loss: 0.5530 - val_M_EXP_categorical_accuracy: 0.2434 - val_M_EXP_loss: 3.1713 - val_loss: 1.5999 - learning_rate: 9.0484e-05
Epoch 3: Current LR = 0.00009048, New LR = 0.00007408
1414/1414 ━━━━━━━━━━━━━━━━━━━━ 7421s 5s/step - Decept_Detect_categorical_accuracy: 0.6329 - Decept_Detect_loss: 0.6269 - M_EXP_categorical_accuracy: 0.1058 - M_EXP_loss: 3.8373 - loss: 1.9111 - val_Decept_Detect_categorical_accuracy: 0.7179 - val_Decept_Detect_loss: 0.5117 - val_M_EXP_categorical_accuracy: 0.3726 - val_M_EXP_loss: 2.7577 - val_loss: 1.4093 - learning_rate: 7.4082e-05
Epoch 4: Current LR = 0.00007408, New LR = 0.00005488
1414/1414 ━━━━━━━━━━━━━━━━━━━━ 7405s 5s/step - Decept_Detect_categorical_accuracy: 0.6586 - Decept_Detect_loss: 0.6106 - M_EXP_categorical_accuracy: 0.1191 - M_EXP_loss: 3.7873 - loss: 1.8813 - val_Decept_Detect_categorical_accuracy: 0.7349 - val_Decept_Detect_loss: 0.5504 - val_M_EXP_categorical_accuracy: 0.4179 - val_M_EXP_loss: 2.7471 - val_loss: 1.4283 - learning_rate: 5.4881e-05
Epoch 5: Current LR = 0.00005488, New LR = 0.00003679
1414/1414 ━━━━━━━━━━━━━━━━━━━━ 7393s 5s/step - Decept_Detect_categorical_accuracy: 0.6806 - Decept_Detect_loss: 0.5911 - M_EXP_categorical_accuracy: 0.1256 - M_EXP_loss: 3.7233 - loss: 1.8440 - val_Decept_Detect_categorical_accuracy: 0.7755 - val_Decept_Detect_loss: 0.4551 - val_M_EXP_categorical_accuracy: 0.4642 - val_M_EXP_loss: 2.5821 - val_loss: 1.3056 - learning_rate: 3.6788e-05
Epoch 6: Current LR = 0.00003679, New LR = 0.00002231
##################################################################
Last MHEmodel save_weights &&& Epoch 1: Current LR = 0.00002231, New LR = 0.00002231
1414/1414 ━━━━━━━━━━━━━━━━━━━━ 7458s 5s/step - Decept_Detect_categorical_accuracy: 0.6717 - Decept_Detect_loss: 0.5927 - M_EXP_categorical_accuracy: 0.1479 - M_EXP_loss: 3.6861 - loss: 1.8300 - val_Decept_Detect_categorical_accuracy: 0.7877 - val_Decept_Detect_loss: 0.5207 - val_M_EXP_categorical_accuracy: 0.4538 - val_M_EXP_loss: 2.6511 - val_loss: 1.3731 - learning_rate: 2.2310e-05
Last MHEmodel save_weights &&& Epoch 2: Current LR = 0.00002231, New LR = 0.00002019
1414/1414 ━━━━━━━━━━━━━━━━━━━━ 7448s 5s/step - Decept_Detect_categorical_accuracy: 0.6925 - Decept_Detect_loss: 0.5770 - M_EXP_categorical_accuracy: 0.1661 - M_EXP_loss: 3.6509 - loss: 1.8066 - val_Decept_Detect_categorical_accuracy: 0.8179 - val_Decept_Detect_loss: 0.4436 - val_M_EXP_categorical_accuracy: 0.5226 - val_M_EXP_loss: 2.3940 - val_loss: 1.2238 - learning_rate: 2.0187e-05
Last MHEmodel save_weights &&& Epoch 3: Current LR = 0.00002019, New LR = 0.00001653
Epoch 3: LearningRateScheduler setting learning rate to 1.6527654247525e-05.
1414/1414 ━━━━━━━━━━━━━━━━━━━━ 7470s 5s/step - Decept_Detect_categorical_accuracy: 0.6980 - Decept_Detect_loss: 0.5640 - M_EXP_categorical_accuracy: 0.1791 - M_EXP_loss: 3.6456 - loss: 1.7966 - val_Decept_Detect_categorical_accuracy: 0.8170 - val_Decept_Detect_loss: 0.4411 - val_M_EXP_categorical_accuracy: 0.5255 - val_M_EXP_loss: 2.3107 - val_loss: 1.1891 - learning_rate: 1.6528e-05
Last MHEmodel save_weights &&& Epoch 4: Current LR = 0.00001653, New LR = 0.00001224
1414/1414 ━━━━━━━━━━━━━━━━━━━━ 7552s 5s/step - Decept_Detect_categorical_accuracy: 0.7174 - Decept_Detect_loss: 0.5418 - M_EXP_categorical_accuracy: 0.2028 - M_EXP_loss: 3.5990 - loss: 1.7647 - val_Decept_Detect_categorical_accuracy: 0.8340 - val_Decept_Detect_loss: 0.3984 - val_M_EXP_categorical_accuracy: 0.5283 - val_M_EXP_loss: 2.3099 - val_loss: 1.1634 - learning_rate: 1.2244e-05
Last MHEmodel save_weights &&& Epoch 5: Current LR = 0.00001224, New LR = 0.00000821
1414/1414 ━━━━━━━━━━━━━━━━━━━━ 7467s 5s/step - Decept_Detect_categorical_accuracy: 0.7170 - Decept_Detect_loss: 0.5410 - M_EXP_categorical_accuracy: 0.2094 - M_EXP_loss: 3.5962 - loss: 1.7631 - val_Decept_Detect_categorical_accuracy: 0.8377 - val_Decept_Detect_loss: 0.3951 - val_M_EXP_categorical_accuracy: 0.5292 - val_M_EXP_loss: 2.3178 - val_loss: 1.1641 - learning_rate: 8.2074e-06

 ###################################################################

Epoch 1/20
1414/1414 ━━━━━━━━━━━━━━━━━━━━ 4086s 3s/step - Decept_Detect_categorical_accuracy: 0.7150 - Decept_Detect_loss: 0.5321 - M_EXP_categorical_accuracy: 0.1885 - M_EXP_loss: 3.6060 - loss: 1.7617 - val_Decept_Detect_categorical_accuracy: 0.8302 - val_Decept_Detect_loss: 0.4745 - val_M_EXP_categorical_accuracy: 0.5160 - val_M_EXP_loss: 2.3456 - val_loss: 1.2239 - learning_rate: 4.9800e-06
Last MHEmodel save_weights &&& Epoch 2: Current LR = 0.00000498, New LR = 0.00000451

Epoch 2/20
1414/1414 ━━━━━━━━━━━━━━━━━━━━ 3995s 3s/step - Decept_Detect_categorical_accuracy: 0.7176 - Decept_Detect_loss: 0.5350 - M_EXP_categorical_accuracy: 0.2039 - M_EXP_loss: 3.5996 - loss: 1.7608 - val_Decept_Detect_categorical_accuracy: 0.8377 - val_Decept_Detect_loss: 0.4263 - val_M_EXP_categorical_accuracy: 0.5311 - val_M_EXP_loss: 2.2744 - val_loss: 1.1660 - learning_rate: 4.5061e-06
Last MHEmodel save_weights &&& Epoch 3: Current LR = 0.00000451, New LR = 0.00000369

Epoch 3/20
1414/1414 ━━━━━━━━━━━━━━━━━━━━ 3977s 3s/step - Decept_Detect_categorical_accuracy: 0.7238 - Decept_Detect_loss: 0.5246 - M_EXP_categorical_accuracy: 0.2168 - M_EXP_loss: 3.5940 - loss: 1.7523 - val_Decept_Detect_categorical_accuracy: 0.8368 - val_Decept_Detect_loss: 0.4291 - val_M_EXP_categorical_accuracy: 0.5283 - val_M_EXP_loss: 2.3736 - val_loss: 1.2076 - learning_rate: 3.6893e-06
Last MHEmodel save_weights &&& Epoch 4: Current LR = 0.00000369, New LR = 0.00000273

Epoch 4/20
1414/1414 ━━━━━━━━━━━━━━━━━━━━ 3910s 3s/step - Decept_Detect_categorical_accuracy: 0.7275 - Decept_Detect_loss: 0.5273 - M_EXP_categorical_accuracy: 0.2102 - M_EXP_loss: 3.5764 - loss: 1.7470 - val_Decept_Detect_categorical_accuracy: 0.8396 - val_Decept_Detect_loss: 0.4234 - val_M_EXP_categorical_accuracy: 0.5396 - val_M_EXP_loss: 2.2837 - val_loss: 1.1683 - learning_rate: 2.7331e-06
Last MHEmodel save_weights &&& Epoch 5: Current LR = 0.00000273, New LR = 0.00000183

Epoch 5/20
1414/1414 ━━━━━━━━━━━━━━━━━━━━ 4084s 3s/step - Decept_Detect_categorical_accuracy: 0.7249 - Decept_Detect_loss: 0.5379 - M_EXP_categorical_accuracy: 0.2234 - M_EXP_loss: 3.5756 - loss: 1.7530 - val_Decept_Detect_categorical_accuracy: 0.8358 - val_Decept_Detect_loss: 0.4523 - val_M_EXP_categorical_accuracy: 0.5387 - val_M_EXP_loss: 2.2676 - val_loss: 1.1792 - learning_rate: 1.8320e-06
Last MHEmodel save_weights &&& Epoch 6: Current LR = 0.00000183, New LR = 0.00000111

Epoch 6/20
1414/1414 ━━━━━━━━━━━━━━━━━━━━ 4082s 3s/step - Decept_Detect_categorical_accuracy: 0.7165 - Decept_Detect_loss: 0.5397 - M_EXP_categorical_accuracy: 0.2123 - M_EXP_loss: 3.5789 - loss: 1.7554 - val_Decept_Detect_categorical_accuracy: 0.8377 - val_Decept_Detect_loss: 0.4493 - val_M_EXP_categorical_accuracy: 0.5443 - val_M_EXP_loss: 2.2850 - val_loss: 1.1846 - learning_rate: 1.1112e-06
Last MHEmodel save_weights &&& Epoch 7: Current LR = 0.00000111, New LR = 0.00000061

Epoch 7/20
1414/1414 ━━━━━━━━━━━━━━━━━━━━ 3945s 3s/step - Decept_Detect_categorical_accuracy: 0.7366 - Decept_Detect_loss: 0.5134 - M_EXP_categorical_accuracy: 0.2231 - M_EXP_loss: 3.5740 - loss: 1.7377 - val_Decept_Detect_categorical_accuracy: 0.8330 - val_Decept_Detect_loss: 0.4508 - val_M_EXP_categorical_accuracy: 0.5311 - val_M_EXP_loss: 2.2782 - val_loss: 1.1826 - learning_rate: 6.0983e-07
Last MHEmodel save_weights &&& Epoch 8: Current LR = 0.00000061, New LR = 0.00000030

Epoch 8/20
1414/1414 ━━━━━━━━━━━━━━━━━━━━ 3987s 3s/step - Decept_Detect_categorical_accuracy: 0.7376 - Decept_Detect_loss: 0.5123 - M_EXP_categorical_accuracy: 0.2223 - M_EXP_loss: 3.5771 - loss: 1.7382 - val_Decept_Detect_categorical_accuracy: 0.8349 - val_Decept_Detect_loss: 0.4546 - val_M_EXP_categorical_accuracy: 0.5462 - val_M_EXP_loss: 2.2888 - val_loss: 1.1891 - learning_rate: 3.0283e-07
Last MHEmodel save_weights &&& Epoch 9: Current LR = 0.00000030, New LR = 0.00000014

Epoch 9/20
1414/1414 ━━━━━━━━━━━━━━━━━━━━ 4068s 3s/step - Decept_Detect_categorical_accuracy: 0.7167 - Decept_Detect_loss: 0.5383 - M_EXP_categorical_accuracy: 0.2174 - M_EXP_loss: 3.5847 - loss: 1.7569 - val_Decept_Detect_categorical_accuracy: 0.8387 - val_Decept_Detect_loss: 0.4355 - val_M_EXP_categorical_accuracy: 0.5292 - val_M_EXP_loss: 2.2678 - val_loss: 1.1691 - learning_rate: 1.3607e-07
Last MHEmodel save_weights &&& Epoch 10: Current LR = 0.00000014, New LR = 0.00000006

Epoch 10/20
1414/1414 ━━━━━━━━━━━━━━━━━━━━ 3997s 3s/step - Decept_Detect_categorical_accuracy: 0.7234 - Decept_Detect_loss: 0.5313 - M_EXP_categorical_accuracy: 0.2288 - M_EXP_loss: 3.5681 - loss: 1.7460 - val_Decept_Detect_categorical_accuracy: 0.8387 - val_Decept_Detect_loss: 0.4213 - val_M_EXP_categorical_accuracy: 0.5434 - val_M_EXP_loss: 2.2724 - val_loss: 1.1626 - learning_rate: 5.5323e-08
Last MHEmodel save_weights &&& Epoch 11: Current LR = 0.00000006, New LR = 0.00000002


 ###################################################################




import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense
from tensorflow.keras.models import Model, Sequential

# Define a reusable CNN block (sub-model)
def create_cnn_block():
    inputs = Input(shape=(28, 28, 1))
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    return Model(inputs=inputs, outputs=x)

# Main model using Functional API
cnn_block = create_cnn_block()  # Sub-model

inputs = Input(shape=(28, 28, 1))
x = cnn_block(inputs)  # Call the sub-model
x = Flatten()(x)
outputs = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)
model.summary()
########################################################
############################################################
from tensorflow.keras.layers import Lambda

cnn_block = create_cnn_block()  # Sub-model from above

model = Sequential([
    Lambda(lambda x: cnn_block(x), input_shape=(28, 28, 1)),  # Call sub-model
    Flatten(),
    Dense(10, activation='softmax')
])
model.summary()
#########################################################################
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense
from tensorflow.keras.models import Model, Sequential

# Define a reusable CNN block (sub-model)
def create_cnn_block(y,n, num_out_layers, kernel_sizes, strides):
    y = Input(shape=(None, None,n))
    x = Conv2D(num_out_layers, kernel_sizes,strides, activation='relu')(y)
    x = Conv2D(num_out_layers, kernel_sizes,strides, activation='relu')(x)
    #x = Conv2D(64, (3, 3), activation='relu')(x)
    return Model(inputs=y, outputs=x)

# Main model using Functional API


inputs = Input(shape=(28, 28, 1))
cnn_block = create_cnn_block(inputs,1,32,(3,3),(1,1))  # Sub-model
x = cnn_block(inputs)  # Call the sub-model
x = Flatten()(x)
outputs = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)
model.summary()
######################################################################
ww = np.zeros(1,).astype(np.float32)
print('ww.shape',ww.shape)

tt = tf.zeros([1], tf.float32)
print('tt.shape',tt.shape,tt)
rr=5.
rr= tf.constant(rr)
print('rr.shape',rr.shape,rr)
print(tt-rr)

rr2=[rr,rr,rr]
#rr2=  np.asarray(rr2).astype(np.float32
rr2 = tf.convert_to_tensor(rr2)                             
print('rr2.shape',rr2.shape,rr2)
##############################################################
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model

X1 = np.random.uniform(0,1, (100,5))
X2 = np.random.uniform(0,1, (100,5))

y1 = np.random.uniform(0,1, 100)
y2 = np.random.uniform(0,1, 100)


def MyLoss(true1, true2, out1, out2, out3):

    loss1 = tf.keras.losses.mse(out1, true1)
    loss2 = tf.keras.losses.mse(out2, true2)
    loss3 = tf.keras.losses.mse(out2, out3)

    loss = loss1 + loss2 + loss3
    return loss


input1 = Input(shape=(5,))
input2 = Input(shape=(5,))

output1 = Dense(1)(Concatenate()([input1,input2]))
output2 = Dense(1)(output1)
output3 = Dense(1)(output1)

true1 = Input(shape=(1,))
true2 = Input(shape=(1,))

model = Model([input1,input2,true1,true2], [output1,output2,output3])
model.summary()
model.add_loss(MyLoss(true1, true2, output1, output2, output3))
model.compile(optimizer='adam', loss=None)

model.fit(x=[X1,X2,y1,y2], y=None, epochs=3)
#######################################################################
input1 = Input(shape=input1_shape)
input2 = Input(shape=input2_shape)
output1 = submodel1()([input1,input2]) #do not pay attention to the code notation, as it is a code to explain the problem.
output2 = submodel2()(output1)
output3 =  submodel3()(output1)
@tf.function
def MyLoss(y_true, y_pred):
    out1, out2, out3 = y_pred
    inp1, inp2 = y_true
            
    loss1 = tf.keras.losses.some_loss1(out1,inp1)
    loss2 = tf.keras.losses.some_loss2(out2, inp2)
    loss3 = tf.keras.losses.some_loss3(out2,out3)

    loss = loss1 + loss2 + loss3
    return loss

model = Model([input1,input2],[output1,output2,output3])
model.compile(optimizer='adam',loss = MyLoss)
###################################################################
def My_Loss(input_5,train_warp2_depth, train_mesh, train_warp2_H1, train_warp2_H2,train_warp2_H3, train_one_warp_H1, train_one_warp_H2,train_one_warp_H3):
    def loss_fun(y_true, y_pred):
        intensity_loss_Layer=intensity__loss()
        lam_lp = 1
        l_num = 1
        loss1 = intensity_loss_Layer(gen_frames=train_warp2_H1, gt_frames=input_5[...,0:3]*train_one_warp_H1, l_num=1)
        loss2 = intensity_loss_Layer(gen_frames=train_warp2_H2, gt_frames=input_5[...,0:3]*train_one_warp_H2, l_num=1)
        loss3 = intensity_loss_Layer(gen_frames=train_warp2_H3, gt_frames=input_5[...,0:3]*train_one_warp_H3, l_num=1)
        lp_loss = 16. * loss1 + 4. * loss2 + 1. * loss3
        # mesh loss
        depth_consistency_loss3_Layer = depth_consistency__loss3()
        lam_mesh = 0   ##   deph assistance??   without lam_mesh = 0      within 10
        mesh_loss = depth_consistency_loss3_Layer(train_warp2_depth, train_mesh)
        #g_loss = tf.add_n([lp_loss * lam_lp, mesh_loss * lam_mesh], name='g_loss')
        g_loss = lp_loss * lam_lp + mesh_loss * lam_mesh

        # g_loss2=[g_loss,g_loss,g_loss]
        # g_loss2 = tf.convert_to_tensor(g_loss2)                             
        # print('g_loss2.shape',g_loss2.shape,g_loss2)
        #out00 = g_loss
        return g_loss
    return loss_fun
    ###################################################################################
###############################################################################
    
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model

X1 = np.random.uniform(0,1, (100,5))
X2 = np.random.uniform(0,1, (100,5))

X10 = np.random.uniform(0,1, 100)
X20 = np.random.uniform(0,1, 100)


def MyLoss(y_true, y_pred):  # Changed to take y_true, y_pred
    true1 = y_true[0]  # Access X10 from y_true list
    true2 = y_true[1]  # Access X20 from y_true list
    out1 = y_pred[0]  # Access output1 from y_pred list
    out2 = y_pred[1]  # Access output2 from y_pred list
    out3 = y_pred[2]  # Access output3 from y_pred list
    loss1 = tf.keras.losses.mse(out1, true1)
    loss2 = tf.keras.losses.mse(out2, true2)
    loss3 = tf.keras.losses.mse(out2, out3)
    loss = loss1 + loss2 + loss3
    return loss

# Define the inputs
input1 = Input(shape=(5,))
input2 = Input(shape=(5,))
true1 = Input(shape=(1,)) # Input for y1
true2 = Input(shape=(1,)) # Input for y2

# Define the model architecture
output1 = Dense(1)(Concatenate()([input1,input2]))
output2 = Dense(1)(output1)
output3 = Dense(1)(output1)

# Create the model
model = Model([input1,input2,true1,true2], [output1,output2,output3]) # specify outputs here
model.summary()

# Add the custom loss function
# We pass the MyLoss function itself, not its calculated value

# Compile the model with no explicit loss since it's already added
model.compile(optimizer='adam', loss=MyLoss) # Pass the function, not its result

# Fit the model
model.fit(x=[X1, X2, X10, X20], y=[X10, X20, X20], epochs=3) # y should match outputs
#####################################################################################

def My_Loss(y_true, y_pred):  # Changed to take y_true, y_pred
  
    input_55 = y_true[3][...,0:3]
    
    out0,out1,out00 = y_pred[0:3]
    train_warp2_depth, train_mesh, train_warp2_H1, train_warp2_H2,train_warp2_H3 = y_pred[3:8]
    train_one_warp_H1, train_one_warp_H2, train_one_warp_H3 = y_pred[8:11]
    
    intensity_loss_Layer=intensity__loss()
    lam_lp = 1
    l_num = 1
    loss1 = intensity_loss_Layer(gen_frames=train_warp2_H1, gt_frames=input_55*train_one_warp_H1, l_num=1)
    loss2 = intensity_loss_Layer(gen_frames=train_warp2_H2, gt_frames=input_55*train_one_warp_H2, l_num=1)
    loss3 = intensity_loss_Layer(gen_frames=train_warp2_H3, gt_frames=input_55*train_one_warp_H3, l_num=1)
    lp_loss = 16. * loss1 + 4. * loss2 + 1. * loss3
    # mesh loss
    depth_consistency_loss3_Layer = depth_consistency__loss3()
    lam_mesh = 0   ##   deph assistance??   without lam_mesh = 0      within 10
    mesh_loss = depth_consistency_loss3_Layer(train_warp2_depth, train_mesh)
    #g_loss = tf.add_n([lp_loss * lam_lp, mesh_loss * lam_mesh], name='g_loss')
    g_loss = lp_loss * lam_lp + mesh_loss * lam_mesh

    # g_loss2=[g_loss,g_loss,g_loss]
    # g_loss2 = tf.convert_to_tensor(g_loss2)                             
    print('g_loss.shape',g_loss.shape,g_loss)
    #out00 = g_loss
    return g_loss ##g_loss  
##############################################################################
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model

X1 = np.random.uniform(0,1, (100,5))
X2 = np.random.uniform(0,1, (100,5))

X10 = np.random.uniform(0,1, 100)
X20 = np.random.uniform(0,1, 100)


def MyLoss(y_true, y_pred):  # Changed to take y_true, y_pred
    true1 = y_true[0]  # Access X10 from y_true list
    true2 = y_true[1]  # Access X20 from y_true list
    out1 = y_pred[0]  # Access output1 from y_pred list
    out2 = y_pred[1]  # Access output2 from y_pred list
    out3 = y_pred[2]  # Access output3 from y_pred list
    loss1 = tf.keras.losses.mse(out1, true1)
    loss2 = tf.keras.losses.mse(out2, true2)
    loss3 = tf.keras.losses.mse(out2, out3)
    loss = loss1 + loss2 + loss3
    return loss

# Define the inputs
input1 = Input(shape=(5,))
input2 = Input(shape=(5,))
true1 = Input(shape=(1,)) # Input for y1
true2 = Input(shape=(1,)) # Input for y2

# Define the model architecture
output1 = Dense(1,name='out1')(Concatenate()([input1,input2]))
output2 = Dense(1,name='out2')(output1)
output3 = Dense(1,name='out3')(output1)
print(output3.shape,output2.shape,output1.shape)
l_num = 1
#loss =  tf.reduce_mean(tf.abs((output3 - output2) ** l_num))

# Create the model
model = Model([input1,input2,true1,true2], [output1,output2,output3]) # specify outputs here
model.summary()

# Add the custom loss function
# We pass the MyLoss function itself, not its calculated value

# Compile the model with no explicit loss since it's already added   None
model.compile(optimizer='adam', loss=[MyLoss, None, 'binary_crossentropy'],
                 loss_weights=[.2,  None, .3],
                 metrics=['accuracy', None,'accuracy']
    ) # Pass the function, not its result

# Fit the model
model.fit(x=[X1, X2, X10, X20], y=[X10, X20, X20],
          epochs=3,
          batch_size=10,
          verbose=1,shuffle=1,) # y should match outputs
#####################################################################################

#########################################################
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model
x =tf.ones(shape=[3, 512, 512, 3 ], dtype=tf.float32)
y =tf.random.normal(shape=[3, 512, 512, 3 ], dtype=tf.float32)
ll1 = 16. * x + 4. * y 
l_num = 1.
ww1 = tf.abs(x - y)
ww= tf.reduce_mean(tf.abs((x - y) ** l_num))
print(ww,ww1,ll1)
#########################################################
#####################################################################################
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K

class AbsoluteDifferenceLayer(layers.Layer):
    def call(self, inputs):
        x, y = inputs
        return K.abs(x - y)

# Example of using the custom layer
input_x = layers.Input(shape=(512, 512, 3))
input_y = layers.Input(shape=(512, 512, 3))

abs_diff_layer = AbsoluteDifferenceLayer()
output_layer = abs_diff_layer([input_x, input_y])

from tensorflow.keras.models import Model

# Create a dummy model to see the output
model = Model(inputs=[input_x, input_y], outputs=output_layer)

# Example input data (similar to your original tensors)
x_example = tf.ones(shape=[3, 512, 512, 3], dtype=tf.float32)
y_example = tf.random.normal(shape=[3, 512, 512, 3], dtype=tf.float32)

output = model.predict([x_example.numpy(), y_example.numpy()])

print("Output from Keras Layer:", output.shape)
print("Example output (first element of the first batch, first pixel, all channels):\n", output[0, 0, 0, :])

# Verify against the original TensorFlow operation
tf_output = tf.abs(x_example - y_example)
print("\nOutput from TensorFlow operation (first element):\n", tf_output.numpy()[0, 0, 0, :])
########################################################
#########################################################
#####################################################################################

m = keras.metrics.CategoricalAccuracy()                     ### only on hot y_true , y_pred
m.update_state([[1. , 0.], [0. , 1.], [0. , 1.]],           ### first argmax then cheking acc softmax
 [[.93 , .55], [0.49 ,.98], [0.1 , 1.]])                    ### for on hot multi class
print(m.result())

m = keras.metrics.BinaryAccuracy()
m.update_state([[1. , 0.], [0. , 1.], [0. , 1.]],           ### first 1,0 with thereshold .5  then cheking acc
 [[.93 , .55], [0.49 ,.98], [0.1 , 1.]])                    ### for multilable  sigmod
print(m.result())

m = keras.metrics.Accuracy()
m.update_state([[1. , 0.], [0. , 1.], [0. , 1.]],           ### exact the same and equal , evry digit
 [[.93 , .55], [0.49 ,.98], [0.1 , 1.]])
print(m.result())


tf.Tensor(1.0, shape=(), dtype=float32)
tf.Tensor(0.8333333, shape=(), dtype=float32)
tf.Tensor(0.16666667, shape=(), dtype=float32)
#########################################################

#####################################################
import numpy as np 
test_depth=np.random.rand(3, 512, 512, 3).astype(np.float32)
s1=sum(test_depth)
sum_axis_0 = np.sum(test_depth, axis=(1,2,3))
print("test_depth.shape",test_depth.shape,s1.shape,sum_axis_0)
#####################################################
import tensorflow as tf

test = tf.random.normal(shape=[3, 512, 512, 3], dtype=tf.float32)
print("Shape of 'test' tensor:", test.shape)

# Sum along axes 1, 2, and 3 using tf.reduce_sum()
t1 = tf.reduce_sum(test, axis=(1, 2, 3))
print("Shape of 't1' (after summing axes 1, 2, 3):", t1.shape)
print("Value of 't1':", t1)

# Shape of 'test' tensor: (3, 512, 512, 3)
# Shape of 't1' (after summing axes 1, 2, 3): (3,)
# Value of 't1': tf.Tensor([ -178.29108   687.97363 -1557.8868 ], shape=(3,), dtype=float32)
#####################################################
keras.callbacks.ModelCheckpoint(
    filepath,
    monitor="val_loss",
    verbose=0,
    save_best_only=False,
    save_weights_only=False,
    mode="auto",
    save_freq="epoch",
    initial_value_threshold=None,
)



model.compile(loss=..., optimizer=...,
              metrics=['accuracy'])

EPOCHS = 10
checkpoint_filepath = '/tmp/ckpt/checkpoint.model.keras'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

# Model is saved at the end of every epoch, if it's the best seen so far.
model.fit(epochs=EPOCHS, callbacks=[model_checkpoint_callback])

# The model (that are considered the best) can be loaded as -
keras.models.load_model(checkpoint_filepath)

######################################################
##########################################################
################################################################

# Alternatively, one could checkpoint just the model weights as -
checkpoint_filepath = '/tmp/ckpt/checkpoint.weights.h5'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

# Model weights are saved at the end of every epoch, if it's the best seen
# so far.
model.fit(epochs=EPOCHS, callbacks=[model_checkpoint_callback])

# The model weights (that are considered the best) can be loaded as -
model.load_weights(checkpoint_filepath)
###################################################################################################
import keras
import numpy as np
#from sklearn.metrics import roc_curve, auc
#from keras.models import Sequential
#from keras.layers import Dense

#metrics=[keras.metrics.AUC()])
#metric = keras.metrics.F1Score(threshold=2)
metric = keras.metrics.AUC()
#metric = keras.metrics.CategoricalAccuracy()
y_true = np.array([[0, 1],[0, 1],[0, 1],[1, 0],[1, 0],[1, 0]], np.int32)
y_pred = np.array([[0.3, 0.7],[0.2, 0.8],[0.6, 0.4],[0.3, 0.7],[0.2, 0.8],[0.6, 0.4]], np.float32)

metric.update_state(y_true, y_pred)
result = metric.result()
print('auc',result)


m = keras.metrics.CategoricalAccuracy()                     
m.update_state(y_true, y_pred)
print('cat_acc',m.result())

##tauc tf.Tensor(0.5, shape=(), dtype=float32)
##cat_acc tf.Tensor(0.5, shape=(), dtype=float32) 
###########################################################    
y_true = np.array([[0, 1],[0, 1],[0, 1],[1, 0],[1, 0],[1, 0]], np.int32)
y_pred = np.array([[0.7, 0.3],[0.8, 0.2],[0.6, 0.4],[0.3, 0.7],[0.2, 0.8],[0.6, 0.4]], np.float32)
auc tf.Tensor(0.05555556, shape=(), dtype=float32)
cat_acc tf.Tensor(0.16666667, shape=(), dtype=float32) 
###############################################################################
y_true = np.array([[0, 1],[0, 1],[0, 1],[1, 0],[1, 0],[1, 0]], np.int32)
y_pred = np.array([[0.3, 0.7],[0.2, 0.8],[0.4, 0.6],[0.3, 0.7],[0.2, 0.8],[0.6, 0.4]], np.float32)
auc tf.Tensor(0.5555556, shape=(), dtype=float32)
cat_acc tf.Tensor(0.6666667, shape=(), dtype=float32)
####################################  AUC_ROC ####################################
metric = keras.metrics.AUC(multi_label=True, num_labels=2)
y_true = np.array([[0, 1],[0, 1],[0, 1],[1, 0],[1, 0],[1, 0]], np.int32)
y_pred = np.array([[0.7, 0.3],[0.8, 0.2],[0.6, 0.4],[0.3, 0.7],[0.2, 0.8],[0.6, 0.4]], np.float32) 
auc tf.Tensor(0.055555552, shape=(), dtype=float32)
cat_acc tf.Tensor(0.16666667, shape=(), dtype=float32)
###################################### face detection app ###################
Higher minNeighbors (e.g., 5-8 or more):
Higher quality detections: It reduces false positives (detecting something as a face when it's not)
because it requires multiple "votes" from the classifier for a region to be considered a face.
Fewer detections: You might miss some true faces that have fewer overlapping detections.

Smaller scaleFactor (e.g., 1.01-1.1):
More thorough: It means you're taking smaller steps in resizing the image, 
creating more scales in the image pyramid.
This increases the chance of finding a face at the "perfect" size for the trained cascade classifier.
Higher accuracy (potentially): You're less likely to miss faces because you're checking more scales.
Slower: More scales mean more processing, leading to slower detection.
#################################################################
total_videos_batch: ['/content/Deceptive/trial_lie_001.mp4', '/content/Deceptive/trial_lie_002.mp4', '/content/Deceptive/trial_lie_003.mp4', '/content/Deceptive/trial_lie_004.mp4', '/content/Deceptive/trial_lie_005.mp4', '/content/Deceptive/trial_lie_006.mp4', '/content/Deceptive/trial_lie_007.mp4', '/content/Deceptive/trial_lie_008.mp4', '/content/Deceptive/trial_lie_009.mp4', '/content/Deceptive/trial_lie_010.mp4', '/content/Deceptive/trial_lie_011.mp4', '/content/Deceptive/trial_lie_012.mp4', '/content/Deceptive/trial_lie_013.mp4', '/content/Deceptive/trial_lie_014.mp4', '/content/Deceptive/trial_lie_015.mp4', '/content/Deceptive/trial_lie_016.mp4', '/content/Deceptive/trial_lie_017.mp4', '/content/Deceptive/trial_lie_018.mp4', '/content/Deceptive/trial_lie_019.mp4', '/content/Deceptive/trial_lie_020.mp4', '/content/Deceptive/trial_lie_021.mp4', '/content/Deceptive/trial_lie_022.mp4', '/content/Deceptive/trial_lie_023.mp4', '/content/Deceptive/trial_lie_024.mp4', '/content/Deceptive/trial_lie_025.mp4', '/content/Deceptive/trial_lie_026.mp4', '/content/Deceptive/trial_lie_027.mp4', '/content/Deceptive/trial_lie_028.mp4', '/content/Deceptive/trial_lie_029.mp4', '/content/Deceptive/trial_lie_030.mp4', '/content/Deceptive/trial_lie_031.mp4', '/content/Deceptive/trial_lie_032.mp4', '/content/Deceptive/trial_lie_033.mp4', '/content/Deceptive/trial_lie_034.mp4', '/content/Deceptive/trial_lie_035.mp4', '/content/Deceptive/trial_lie_036.mp4', '/content/Deceptive/trial_lie_037.mp4', '/content/Deceptive/trial_lie_038.mp4', '/content/Deceptive/trial_lie_039.mp4', '/content/Deceptive/trial_lie_040.mp4', '/content/Deceptive/trial_lie_041.mp4', '/content/Deceptive/trial_lie_042.mp4', '/content/Deceptive/trial_lie_043.mp4', '/content/Deceptive/trial_lie_044.mp4', '/content/Deceptive/trial_lie_045.mp4', '/content/Deceptive/trial_lie_046.mp4', '/content/Deceptive/trial_lie_047.mp4', '/content/Deceptive/trial_lie_048.mp4', '/content/Deceptive/trial_lie_049.mp4', '/content/Deceptive/trial_lie_050.mp4', '/content/Deceptive/trial_lie_051.mp4', '/content/Deceptive/trial_lie_052.mp4', '/content/Deceptive/trial_lie_053.mp4', '/content/Deceptive/trial_lie_054.mp4', '/content/Deceptive/trial_lie_055.mp4', '/content/Deceptive/trial_lie_056.mp4', '/content/Deceptive/trial_lie_057.mp4', '/content/Deceptive/trial_lie_058.mp4', '/content/Deceptive/trial_lie_059.mp4', '/content/Deceptive/trial_lie_060.mp4', '/content/Deceptive/trial_lie_061.mp4', '/content/Truthful/trial_truth_001.mp4', '/content/Truthful/trial_truth_002.mp4', '/content/Truthful/trial_truth_003.mp4', '/content/Truthful/trial_truth_004.mp4', '/content/Truthful/trial_truth_005.mp4', '/content/Truthful/trial_truth_006.mp4', '/content/Truthful/trial_truth_007.mp4', '/content/Truthful/trial_truth_008.mp4', '/content/Truthful/trial_truth_009.mp4', '/content/Truthful/trial_truth_010.mp4', '/content/Truthful/trial_truth_011.mp4', '/content/Truthful/trial_truth_012.mp4', '/content/Truthful/trial_truth_013.mp4', '/content/Truthful/trial_truth_014.mp4', '/content/Truthful/trial_truth_015.mp4', '/content/Truthful/trial_truth_016.mp4', '/content/Truthful/trial_truth_017.mp4', '/content/Truthful/trial_truth_018.mp4', '/content/Truthful/trial_truth_019.mp4', '/content/Truthful/trial_truth_020.mp4', '/content/Truthful/trial_truth_021.mp4', '/content/Truthful/trial_truth_022.mp4', '/content/Truthful/trial_truth_023.mp4', '/content/Truthful/trial_truth_024.mp4', '/content/Truthful/trial_truth_025.mp4', '/content/Truthful/trial_truth_026.mp4', '/content/Truthful/trial_truth_027.mp4', '/content/Truthful/trial_truth_028.mp4', '/content/Truthful/trial_truth_029.mp4', '/content/Truthful/trial_truth_030.mp4', '/content/Truthful/trial_truth_031.mp4', '/content/Truthful/trial_truth_032.mp4', '/content/Truthful/trial_truth_033.mp4', '/content/Truthful/trial_truth_034.mp4', '/content/Truthful/trial_truth_035.mp4', '/content/Truthful/trial_truth_036.mp4', '/content/Truthful/trial_truth_037.mp4', '/content/Truthful/trial_truth_038.mp4', '/content/Truthful/trial_truth_039.mp4', '/content/Truthful/trial_truth_040.mp4', '/content/Truthful/trial_truth_041.mp4', '/content/Truthful/trial_truth_042.mp4', '/content/Truthful/trial_truth_043.mp4', '/content/Truthful/trial_truth_044.mp4', '/content/Truthful/trial_truth_045.mp4', '/content/Truthful/trial_truth_046.mp4', '/content/Truthful/trial_truth_047.mp4', '/content/Truthful/trial_truth_048.mp4', '/content/Truthful/trial_truth_049.mp4', '/content/Truthful/trial_truth_050.mp4', '/content/Truthful/trial_truth_051.mp4', '/content/Truthful/trial_truth_052.mp4', '/content/Truthful/trial_truth_053.mp4', '/content/Truthful/trial_truth_054.mp4', '/content/Truthful/trial_truth_055.mp4', '/content/Truthful/trial_truth_056.mp4', '/content/Truthful/trial_truth_057.mp4', '/content/Truthful/trial_truth_058.mp4', '/content/Truthful/trial_truth_059.mp4', '/content/Truthful/trial_truth_060.mp4']
kk==len(total_videos_batch)== 121
video_label: [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0]]
inner_names ==  ['trial_lie_001.mp4', 'trial_lie_002.mp4', 'trial_lie_003.mp4', 'trial_lie_004.mp4', 'trial_lie_005.mp4', 'trial_lie_006.mp4', 'trial_lie_007.mp4', 'trial_lie_008.mp4', 'trial_lie_009.mp4', 'trial_lie_010.mp4', 'trial_lie_011.mp4', 'trial_lie_012.mp4', 'trial_lie_013.mp4', 'trial_lie_014.mp4', 'trial_lie_015.mp4', 'trial_lie_016.mp4', 'trial_lie_017.mp4', 'trial_lie_018.mp4', 'trial_lie_019.mp4', 'trial_lie_020.mp4', 'trial_lie_021.mp4', 'trial_lie_022.mp4', 'trial_lie_023.mp4', 'trial_lie_024.mp4', 'trial_lie_025.mp4', 'trial_lie_026.mp4', 'trial_lie_027.mp4', 'trial_lie_028.mp4', 'trial_lie_029.mp4', 'trial_lie_030.mp4', 'trial_lie_031.mp4', 'trial_lie_032.mp4', 'trial_lie_033.mp4', 'trial_lie_034.mp4', 'trial_lie_035.mp4', 'trial_lie_036.mp4', 'trial_lie_037.mp4', 'trial_lie_038.mp4', 'trial_lie_039.mp4', 'trial_lie_040.mp4', 'trial_lie_041.mp4', 'trial_lie_042.mp4', 'trial_lie_043.mp4', 'trial_lie_044.mp4', 'trial_lie_045.mp4', 'trial_lie_046.mp4', 'trial_lie_047.mp4', 'trial_lie_048.mp4', 'trial_lie_049.mp4', 'trial_lie_050.mp4', 'trial_lie_051.mp4', 'trial_lie_052.mp4', 'trial_lie_053.mp4', 'trial_lie_054.mp4', 'trial_lie_055.mp4', 'trial_lie_056.mp4', 'trial_lie_057.mp4', 'trial_lie_058.mp4', 'trial_lie_059.mp4', 'trial_lie_060.mp4', 'trial_lie_061.mp4', 'trial_truth_001.mp4', 'trial_truth_002.mp4', 'trial_truth_003.mp4', 'trial_truth_004.mp4', 'trial_truth_005.mp4', 'trial_truth_006.mp4', 'trial_truth_007.mp4', 'trial_truth_008.mp4', 'trial_truth_009.mp4', 'trial_truth_010.mp4', 'trial_truth_011.mp4', 'trial_truth_012.mp4', 'trial_truth_013.mp4', 'trial_truth_014.mp4', 'trial_truth_015.mp4', 'trial_truth_016.mp4', 'trial_truth_017.mp4', 'trial_truth_018.mp4', 'trial_truth_019.mp4', 'trial_truth_020.mp4', 'trial_truth_021.mp4', 'trial_truth_022.mp4', 'trial_truth_023.mp4', 'trial_truth_024.mp4', 'trial_truth_025.mp4', 'trial_truth_026.mp4', 'trial_truth_027.mp4', 'trial_truth_028.mp4', 'trial_truth_029.mp4', 'trial_truth_030.mp4', 'trial_truth_031.mp4', 'trial_truth_032.mp4', 'trial_truth_033.mp4', 'trial_truth_034.mp4', 'trial_truth_035.mp4', 'trial_truth_036.mp4', 'trial_truth_037.mp4', 'trial_truth_038.mp4', 'trial_truth_039.mp4', 'trial_truth_040.mp4', 'trial_truth_041.mp4', 'trial_truth_042.mp4', 'trial_truth_043.mp4', 'trial_truth_044.mp4', 'trial_truth_045.mp4', 'trial_truth_046.mp4', 'trial_truth_047.mp4', 'trial_truth_048.mp4', 'trial_truth_049.mp4', 'trial_truth_050.mp4', 'trial_truth_051.mp4', 'trial_truth_052.mp4', 'trial_truth_053.mp4', 'trial_truth_054.mp4', 'trial_truth_055.mp4', 'trial_truth_056.mp4', 'trial_truth_057.mp4', 'trial_truth_058.mp4', 'trial_truth_059.mp4', 'trial_truth_060.mp4']

num_face_videos = [433, 1856, 181, 333, 1212, 444, 1378, 218, 618, 875, 1009, 236, 561, 272, 339, 708, 1118, 504, 913, 193, 262, 821, 792, 390, 381, 426, 694, 597, 583, 785, 627, 698, 419, 693, 347, 12, 681, 588, 707, 647, 9, 637, 300, 130, 231, 988, 479, 1410, 687, 32, 51, 1182, 8, 465, 959, 633, 497, 367, 1152, 31, 166, 89, 462, 382, 2442, 1036, 917, 2137, 1193, 648, 2118, 1224, 844, 833, 379, 1066, 126, 56, 173, 324, 168, 147, 832, 0, 481, 139, 553, 415, 264, 288, 1367, 411, 924, 424, 774, 749, 924, 502, 466, 778, 550, 119, 223, 213, 188, 2, 344, 253, 204, 1, 4, 868, 10, 253, 719, 869, 1018, 1119, 626, 806, 483]
####################  train , val dataset  #####################################
This separation Dataset is a cornerstone of robust machine learning model development.
Hyperparameters are set before training 
(e.g., learning rate, number of hidden layers, number of neurons, regularization strength).

Considerations for Your Specific Setup:
Adam Optimizer and Learning Rate (0.0001): Adam is generally robust and less prone to exploding gradients. A learning rate of 0.0001 is quite small, which is good for preventing rapid, unstable updates but might mean training takes longer to converge. It doesn't directly prevent overfitting, but a very high learning rate can exacerbate it by causing the model to jump into sharp minima.
Large Model Size (104M params, 36 layers): With such a large model, data augmentation, dropout, batch normalization, and early stopping are absolutely critical. Without them, it's highly probable your model will overfit, especially if your dataset isn't colossal.
By systematically applying a combination of these techniques, monitoring your validation metrics closely, and iteratively tuning hyperparameters, you can significantly mitigate overfitting and build a model that generalizes well to unseen data. Start by implementing early stopping, data augmentation, and dropout/L2 regularization, and then fine-tune.
Overfitting is a common problem in machine learning where a model learns the training data too well, including its noise and specific quirks, leading to poor performance on new, unseen data. Given your model's characteristics (104 million parameters, 36 layers), it's a large model, which makes it more prone to overfitting. The Adam optimizer with a learning_rate=0.0001 is a good starting point, but it doesn't inherently prevent overfitting; rather, it's about how effectively the model navigates the loss landscape.

Here are the key techniques to combat overfitting during model training, especially relevant for large models like yours:

1. Data-Related Techniques
More Data: The most effective way to combat overfitting is to simply provide more diverse training data. A larger and more representative dataset helps the model learn generalizable patterns rather than memorizing specific examples. If acquiring more real data is difficult, consider:
Data Augmentation: This involves creating new training examples by applying various transformations to your existing data. For images, this could include rotations, flips, shifts, zooms, brightness changes, color jitter, etc. For text, it could be synonym replacement, back-translation, or random insertion/deletion. This artificially expands your dataset and exposes the model to more variations.
Data Cleaning and Preprocessing: Ensure your data is clean and free from errors, outliers, or irrelevant features. Noisy data can mislead the model into learning spurious correlations.
2. Model-Related Techniques (Complexity Control)
Regularization: This is a broad category of techniques that add a penalty to the loss function based on the complexity of the model's weights. This discourages the model from assigning very large weights to individual features, thus making the model "smoother" and less prone to memorizing.
L1 Regularization (Lasso): Adds the absolute value of the weights to the loss function. It encourages sparsity, meaning it can drive some weights exactly to zero, effectively performing feature selection.
Loss=OriginalLoss+λ i=1∑n∣w i ∣
L2 Regularization (Ridge / Weight Decay): Adds the squared value of the weights to the loss function. It encourages smaller weights, pushing them towards zero but rarely exactly to zero. This is a very common and effective technique for neural networks.
Loss=OriginalLoss+λ 
i=1∑n w i2
 
(Note: Adam's built-in weight decay might be slightly different from traditional L2 regularization; sometimes AdamW is preferred for a more decoupled weight decay.)
Adding Regularizers in Keras/TensorFlow: You can typically add L1/L2 regularizers to Dense or Conv2D layers using kernel_regularizer and bias_regularizer arguments.
Dropout: This is a powerful regularization technique specifically for neural networks. During training, a certain percentage of neurons (and their connections) are randomly "dropped out" (i.e., temporarily ignored or set to zero) at each training step. This prevents neurons from co-adapting too much and forces the network to learn more robust features.
Placement: Dropout layers are typically placed after activation functions in hidden layers.
Rate: A common dropout rate is between 0.2 and 0.5.
Reduce Model Complexity: With 104 million parameters and 36 layers, your model is quite large. If your dataset is not equally massive and diverse, this complexity could be a primary reason for overfitting.
Fewer Layers: Consider reducing the number of hidden layers.
Fewer Neurons per Layer: Decrease the number of units (neurons) in each layer.
Simpler Architecture: For convolutional networks, fewer filters or smaller kernel sizes might be appropriate.
Feature Selection/Engineering: If applicable, selecting only the most relevant features or creating better, more informative features can simplify the learning task for the model.
3. Training Process-Related Techniques
Early Stopping: This is a simple yet very effective technique. You monitor the model's performance (e.g., validation loss or accuracy) on a separate validation set during training. When the validation performance stops improving for a certain number of epochs (the "patience"), you stop the training. This prevents the model from continuing to optimize on the training data beyond the point where it generalizes well.
Implementation: Keras callbacks.EarlyStopping is perfect for this.
Cross-Validation: While not directly during training, cross-validation is a crucial technique for robust model evaluation and hyperparameter tuning, which indirectly helps prevent overfitting. Instead of a single train/validation split, you divide the data into K "folds." The model is trained K times, each time using K-1 folds for training and one different fold for validation. This gives a more reliable estimate of the model's performance and helps in selecting the best hyperparameters.
Batch Normalization: This technique normalizes the inputs of each layer within a batch. It helps stabilize and accelerate the training process and can also act as a form of regularization, reducing the need for strong dropout or L2 regularization.
4. Optimizer and Hyperparameter Tuning
Learning Rate Scheduling: While Adam adaptively adjusts learning rates, a fixed learning_rate=0.0001 might still be too high or too low at different stages of training.
Decay: Gradually reducing the learning rate over time (e.g., step decay, exponential decay, cosine annealing) can help the model converge more precisely in the later stages of training without overshooting the minima.
Learning Rate Finder (Advanced): Tools exist to automatically find a good learning rate range.
Adam Variants: Consider using AdamW (Adam with decoupled weight decay), which often performs better than standard Adam when L2 regularization is applied, as it separates the weight decay from the adaptive learning rates.
Example Considerations for Your Model:
Given:

Total params: 104,192,010 (397.46 MB)
len(MHEmodel.layers) == 36
This is a very large model. Here's a tailored approach:

Start with the basics:
Early Stopping: Implement tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True). Adjust patience based on your training dynamics.
Validation Set: Ensure you have a truly separate validation set (e.g., 10-20% of your data).
Regularization (Crucial for Large Models):
Dropout: Add tf.keras.layers.Dropout layers, especially after convolutional layers and before dense layers. Start with rates like 0.2-0.5. Experiment with different rates for different layers.
L2 Regularization: Add kernel_regularizer=tf.keras.regularizers.l2(0.001) to Dense and Conv2D layers. Start with a small λ (e.g., 0.001 or 0.0001) and tune it.
Batch Normalization: Add tf.keras.layers.BatchNormalization() layers, typically after convolutional layers and before activations, or before fully connected layers. This often helps a lot.
Data Augmentation: If you're working with image data, definitely use image augmentation. For text or other data types, explore relevant augmentation techniques.
Learning Rate Schedule: Even with Adam, a learning rate scheduler can be beneficial. tf.keras.optimizers.schedules.CosineDecay or ExponentialDecay are good options. You could wrap your Adam optimizer with this.
Monitor Closely: Plot training loss vs. validation loss, and training accuracy vs. validation accuracy. If the training loss keeps decreasing while validation loss increases, you are overfitting.
By systematically applying these techniques, you can significantly reduce overfitting and improve your model's ability to generalize to new data. You'll likely need to experiment with different combinations and hyperparameter values to find the optimal setup for your specific problem and dataset.
#############################################
A list of objects to apply during training. Callbacks allow you to perform actions at various stages of the training process (e.g., at the end of an epoch, after a batch). Common callbacks include:

ModelCheckpoint: Saves the model's weights during training.

EarlyStopping: Stops training early if validation performance stops improving.

TensorBoard: Logs training metrics for visualization in TensorBoard.

LearningRateScheduler: Adjusts the learning rate during training.

##################################################################################
####### #####################takes a list of videos and returns a list of gray-scale video
def rgb2list2face(rgb_videos: list, resize_shape: tuple=(64, 64)) -> list:  #tuple=(112, 112) tuple=(256, 256))
    face_orgin__videos = []
    num_face_videos = []
    facecasc = cv2.CascadeClassifier(pathgd+'haarcascade_frontalface_default.xml') ## Extracting FACE
    
    ##for index,u in enumerate(num_face_videos) :
    ##for video_path in tqdm(video_paths):
    for index,video in enumerate(rgb_videos):
 ##############################################################
 ######################   3d vgg  ###############################
 # Define a function to build the 3D VGG model
def build_vgg19_3d(input_shape, num_classes):
    # Input shape is (Depth, Height, Width, Channels) for 'channels_last'
    # Example for a video clip: (16, 112, 112, 3) 
    
    model = Sequential([
        Input(shape=input_shape)
    ])

    # --- Block 1: Two Conv3D layers with 64 filters ---
    model.add(Conv3D(64, (3, 3, 3), activation='relu', padding='same', name='block1_conv1'))
    model.add(Conv3D(64, (3, 3, 3), activation='relu', padding='same', name='block1_conv2'))
    model.add(MaxPool3D((2, 2, 2), strides=(2, 2, 2), name='block1_pool'))

    # --- Block 2: Two Conv3D layers with 128 filters ---
    model.add(Conv3D(128, (3, 3, 3), activation='relu', padding='same', name='block2_conv1'))
    model.add(Conv3D(128, (3, 3, 3), activation='relu', padding='same', name='block2_conv2'))
    model.add(MaxPool3D((2, 2, 2), strides=(2, 2, 2), name='block2_pool'))
    
    # --- Block 3: Four Conv3D layers with 256 filters (VGG19 style) ---
    model.add(Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='block3_conv1'))
    model.add(Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='block3_conv2'))
    model.add(Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='block3_conv3'))
    model.add(Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='block3_conv4'))
    model.add(MaxPool3D((2, 2, 2), strides=(2, 2, 2), name='block3_pool'))

    # --- Block 4: Four Conv3D layers with 512 filters ---
    model.add(Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='block4_conv1'))
    model.add(Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='block4_conv2'))
    model.add(Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='block4_conv3'))
    model.add(Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='block4_conv4'))
    model.add(MaxPool3D((2, 2, 2), strides=(2, 2, 2), name='block4_pool'))
    
    # --- Block 5: Four Conv3D layers with 512 filters ---
    model.add(Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='block5_conv1'))
    model.add(Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='block5_conv2'))
    model.add(Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='block5_conv3'))
    model.add(Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='block5_conv4'))
    model.add(MaxPool3D((2, 2, 2), strides=(2, 2, 2), name='block5_pool')) # Output size depends on initial input

    # --- Classification Head ---
    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, activation='relu', name='fc1'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', name='fc2'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax', name='predictions'))

    return model

# Example Usage:
# Assuming an input video of 16 frames, 112x112 pixels, 3 color channels
INPUT_DIMS = (16, 112, 112, 3) 
NUM_CLASSES = 10 

vgg19_3d_model = build_vgg19_3d(INPUT_DIMS, NUM_CLASSES)
vgg19_3d_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
vgg19_3d_model.summary()