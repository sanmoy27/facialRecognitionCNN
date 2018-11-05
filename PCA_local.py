from __future__ import print_function
from time import time
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA
from sklearn import decomposition
from sklearn.svm import SVC
from PIL import Image
import os, numpy
from sklearn import metrics, svm, neighbors
from sklearn.metrics import classification_report
import pandas as pd

os.chdir("C:\\F\\NMIMS\\DataScience\\Sem-2\\Python\\opencv")

path="training-data\\s2"
X=[]
h=150
w=125
target_names=dict()
target_names={'amber':0, 'amy':1, 'andrew':2, 'andy':3, 'erin':4, 'gabe':5, 'hill':6, 'jack':7, 'zach':8}

#### Image Conversion
#count=0
#for filePath in sorted(os.listdir(path)):
#    count+=1
#    imagePath = os.path.join(path, filePath)
#    img=Image.open(imagePath)
#    img=img.resize((150,125), Image.ANTIALIAS)
#    newimg = img.convert(mode='P', colors=8)
#    newimg.save('test_'+str(count)+'.png')

#######Converting the images into matrix
for filePath in sorted(os.listdir(path)):
    imagePath = os.path.join(path, filePath)
    img=Image.open(imagePath)
    featurevector=numpy.array(img).flatten()
    X.append(featurevector)
        

X=numpy.asarray(X)
print("Image Matrix of all images:\n")
print(X)
#######Compute Labels
Y = pd.read_csv('training-data\\class.csv')
Y=Y['Class_label'].replace({'amber':0, 'amy':1, 'andrew':2, 'andy':3, 'erin':4, 'gabe':5, 'hill':6, 'jack':7, 'zach':8})
Y=Y.values
#Y=numpy.asarray(Y).reshape(27,1)

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)


###############Compute PCA and Eigen faces################
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit on training set only.
scaler.fit(X_train)
# Apply transform to both the training set and the test set.
train_img = scaler.transform(X_train)
test_img = scaler.transform(X_test)
pca = PCA(0.95).fit(train_img)



print("\nExplained Variance\n")
print(pca.explained_variance_)
   
print("\nPCA Components\n")
print(pca.components_)

eigenfaces = pca.components_.reshape((pca.n_components_, h, w))

print("\nTransformed Train Matrix\n")
X_train_pca = pca.transform(train_img)
print(X_train_pca)
print("\nTransformed Test Matrix\n")
X_test_pca = pca.transform(test_img)
print(X_test_pca)


############################# SVM ############################
knn = neighbors.KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='auto')
knn.fit(X_train_pca, y_train)


y_trainPredKNN = knn.predict(X_train_pca)  
acc_knnTrain = metrics.accuracy_score(y_train, y_trainPredKNN)
print("Train Set accuracy KNN: {0}".format(acc_knnTrain))

y_testPredKNN = knn.predict(X_test_pca)  
acc_knnTest = metrics.accuracy_score(y_test, y_testPredKNN)
print("Test Set accuracy for KNN: {0}".format(acc_knnTest))
cm_knn = metrics.confusion_matrix(y_test, y_testPredKNN)
print("===========Report for Test Set KNN=============")
print(cm_knn)
print(classification_report(y_test, y_testPredKNN))



############################# SVM ############################

svm_classifier=svm.SVC(kernel='linear')              
svm_classifier.fit(X_train_pca, y_train)

preds_trainSVM = svm_classifier.predict(X_train_pca)		
acc_trainSVM = metrics.accuracy_score(y_train, preds_trainSVM)		
print("Train Set accuracy for SVM: {} for {}".format(acc_trainSVM, 'linear'))

preds_testSVM = svm_classifier.predict(X_test_pca)
acc_testSVM = metrics.accuracy_score(y_test, preds_testSVM)
print("Test Set accuracy in SVM: {} for {}".format(acc_testSVM, 'linear'))

print("=========Report for Test Set SVM in {}".format('linear'))
cm_svm = metrics.confusion_matrix(y_test, preds_testSVM)
print(cm_svm)
print(classification_report(y_test, preds_testSVM))


def plot_gallery(images, titles, h, w, n_row=4, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(len(images)):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# plot the result of the prediction on a portion of the test set

def title(y_pred, y_test, target_names, i):
    idx_true=y_test[i]
    idx_pred=y_pred[i]
    pred_name = list(target_names.keys())[list(target_names.values()).index(idx_pred)]
    true_name = list(target_names.keys())[list(target_names.values()).index(idx_true)]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = [title(preds_testSVM, y_test, target_names, i)
                     for i in range(preds_testSVM.shape[0])]

plot_gallery(X_test, prediction_titles, h, w)



eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)
plt.show()
