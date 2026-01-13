import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import joblib

#____________________________________________________________________________________________
#step 1: path setup
#____________________________________________________________________________________________
BASE_DIR=os.path.dirname(os.path.dirname(__file__))
DATASET_DIR=os.path.join(BASE_DIR,"dataset")
#also joining the path of two subfolder inside the "dataset"
OPEN_DIR=os.path.join(DATASET_DIR,"open_eyes")# images for class "open"
CLOSED_DIR=os.path.join(DATASET_DIR,"closed_eyes")#images for class "closed"

#model folder:project_root/models (to save preprocessed data/ trained model)
MODELS_DIR=os.path.join(BASE_DIR,"models")
os.makedirs(MODELS_DIR,exist_ok=True) # create folder if not exist

#we will resize all images to 24x24 
IMG_SIZE=24 #final image size: 24x24
#___________________________________________________________________________________________
# step 2: Function to load image from the folder 
#____________________________________________________________________________________________

def load_images_from_folder(folder,label):
    """
     this function will Reads all images from the given folder, preprocess them and returns
     -data:list of flattened image arrays
     -labels:list of labels (same length as data)
     label=1 for open eyes, 0 for closed eyes(as per your choice)
    """
    data=[]
    labels=[]
    #Loop through all files in the folder and perform some operations
    for filename in os.listdir(folder):
        img_path=os.path.join(folder,filename)
        #read image in grayscale
        img=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        #if if image is not loaded correctly (e.g broen file) skip it
        if img is None:
            continue
        #Resize image to fixed size: 24x24
        img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        #Normalize the pixel value from[0,255] to [0,1]
        img=img/255.0
        #Flatten 2D image(24x24) into 1D VECTOR (576)
        #Many ML models in sklearn expect 1D feature vector
        data.append(img.flatten())
        #Append the given level for this image
        labels.append(label)
    return data,labels

#______________________________________________________________________________________________
#step 3 : Load open and closed eyes images
#______________________________________________________________________________________________

#For open eyes, we will use label=1, label=0 for closed eyes
open_data,open_labels=load_images_from_folder(OPEN_DIR,1)
closed_data,closed_labels=load_images_from_folder(CLOSED_DIR,0)
#COMBINING BOTH DATA INTO SINGLE ARRAY
X=np.array(open_data+closed_data) #Feature
y=np.array(open_labels+closed_labels) #labels

print("Total samples:",X.shape[0])# this will represent the number of images
print("Feature size:",X.shape[1])# feature per images

#_____________________________________________________________________________________
#step 4:splitting the dataset
# ____________________________________________________________________________________

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42,stratify=y #20%test and 80%train data, Stratify ensures ratio of closed and open eyes remain same across train/test
) 

#__________________________________________________________________________________________________________________________________________________________________
#step 5. Save preprocessed data(optionalbut useful)
# we will save numpy array to disk , so next time we dont need to redo preprocessing
#we can directly reload these files when training a model
#___________________________________________________________________________________________________________________________________________________________________
np.save(os.path.join(MODELS_DIR,"X_train.npy"),X_train)
np.save(os.path.join(MODELS_DIR,"X_test.npy"),X_test)
np.save(os.path.join(MODELS_DIR,"y_train.npy"),y_train)
np.save(os.path.join(MODELS_DIR,"y_test.npy"),y_test)

print("Preprocessing done and data saved.")
