import os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import joblib
#_____________________________________________________________________________________________________
#step 1:PATH SETUP
#______________________________________________________________________________________________________
BASE_DIR=os.path.dirname(os.path.dirname(__file__)) # BASE_DIR=project root(one level above the folder where the script is)
MODELS_DIR=os.path.join(BASE_DIR,"models") #models(where the previously saved X_train,X_test,etc exist)

#________________________________________________________________________________________________________________________________________
#step 2:LOAD PREPROCESSED DATA
#These .npy files were created during the preprocessing
#________________________________________________________________________________________________________________________________________
X_train=np.load(os.path.join(MODELS_DIR,"X_train.npy"))
X_test=np.load(os.path.join(MODELS_DIR,"X_test.npy"))
y_train=np.load(os.path.join(MODELS_DIR,"y_train.npy"))
y_test=np.load(os.path.join(MODELS_DIR,"y_test.npy"))

print("Training samples:",X_train.shape[0])#how many images are in training dataset
print("Test samples:",X_test.shape[0]) #how many images in test set

#_________________________________________________________________________________________________________________________________________
# step 3:Define and train the svm model
#_________________________________________________________________________________________________________________________________________
model=SVC(kernel='linear',C=1,probability=True)
"""
kernel='linear'-> uses a linear decision boundary(a hyperplane)
C=1 -> regularization strength(control margin vs misclassificaion error)
probability=True -> enables model.predict_proba() later (for probablities)
"""
#train the model using the training data
model.fit(X_train,y_train)
#___________________________________________________________________________________
#step 4: Predict on test data
#____________________________________________________________________________________________
y_pred=model.predict(X_test)

#____________________________________________________________________________________________
#step 5 Evaluate the model
#____________________________________________________________________________________________
#1) Accuracy:overall percentage of correct prediction
print("Accuracy:",accuracy_score(y_test,y_pred))

#2) Confusion matrix
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))

#3) Classification Repot:
#shows precision,,recall,f1_score for each class (0 and 1)

print("Classification Report:\n",classification_report(y_test,y_pred))

#__________________________________________________________
# step 6: Save the train model to the disk
#_________________________________________________________________
model_path=os.path.join(MODELS_DIR,"eye_state_svm.pkl")

# save the trained svm model using joblib
joblib.dump(model,model_path)
print(f"Model saved at:{model_path}")
