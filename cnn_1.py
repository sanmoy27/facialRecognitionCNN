import os
os.getcwd()
os.chdir("C:/F/NMIMS/DataScience/Sem-3/AI/project/Convolutional_Neural_Networks/Convolutional_Neural_Networks")

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import SGD, adam
from keras.constraints import maxnorm
from keras import regularizers
from sklearn.metrics import classification_report
from sklearn import metrics
import numpy as np
# Initialising the CNN
classifier = Sequential()
#classifier.add(Conv2D(32, 3, 3, input_shape=(64, 64, 3), border_mode='same', activation='relu', W_constraint=maxnorm(3)))
#classifier.add(Dropout(0.2))
#classifier.add(Conv2D(32, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
#classifier.add(MaxPooling2D(pool_size=(2, 2)))
#classifier.add(Flatten())
#classifier.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
#classifier.add(Dropout(0.5))
#classifier.add(Dense(6, activation='softmax'))
## Compile model
#epochs = 10  # >>> should be 25+
#lrate = 0.01
#decay = lrate/epochs
#sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
#classifier.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#print(classifier.summary())

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), padding="same", input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(BatchNormalization())

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(64, (3, 3), padding="same", activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.1))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# Adding a third convolutional layer
classifier.add(Conv2D(128, (3, 3), padding="same", activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.2))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# Step 3 - Flattening
classifier.add(Flatten())



# Step 4 - Full connection
classifier.add(Dense(units = 50, activation = 'relu', kernel_regularizer=regularizers.l2(0.0001)))
classifier.add(Dense(units = 9, activation = 'softmax'))

# Compiling the CNN
optimizer = adam(lr=0.001)
classifier.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('data/training_set/data',
                                                 target_size = (64, 64),
                                                 batch_size = 16,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('data/test_set/data',
                                            target_size = (64, 64),
                                            batch_size = 8,
                                            class_mode = 'categorical')

classifier.fit_generator(training_set,
                         steps_per_epoch = 162,
                         epochs = 15,
                         validation_data = test_set,
                         validation_steps = 59)



filenames=test_set.filenames
nb_samples=len(filenames)
Y_pred = classifier.predict_generator(test_set, np.ceil(nb_samples/8))
y_pred = np.argmax(Y_pred, axis=1)
#print(Y_pred)
print('Confusion Matrix')
print(metrics.confusion_matrix(test_set.classes, y_pred))
print(metrics.accuracy_score(test_set.classes, y_pred))
print('Classification Report')
target_names = ['amber', 'amy', 'andrew', 'andy', 'erin', 'gabe', 'hill', 'jack', 'zach']
print(classification_report(test_set.classes, y_pred, target_names=target_names))
#from skimage.io import imread
#from skimage.transform import resize
#import numpy as np
##
### Class labels
#class_labels = {v: k for k, v in training_set.class_indices.items()}
#class_labels
##
## reading the input image
#img = imread('data\\single_prediction\\1.jpg') 
#img = resize(img,(64,64)) 
#img = np.expand_dims(img,axis=0) 
#prediction = classifier.predict_classes(img)
#prediction


import numpy as np
from keras.preprocessing import image
test_image = image.load_img('data/test_set/data/gabe/gabe3.png', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
#result_1 = classifier.predict_classes(test_image)
#result_1

training_set.class_indices
if result[0][0] == 0:
    prediction='amber'
elif result[0][0] == 1:
    prediction='amy'
elif result[0][0] == 2:
    prediction='andrew'
elif result[0][0] == 3:
    prediction='andy'
elif result[0][0] == 4:
    prediction='erin'
elif result[0][0] == 5:
    prediction='gabe'
elif result[0][0] == 6:
    prediction='hill'
elif result[0][0] == 7:
    prediction='jack'
else:
    prediction='zach'
print("The input image is: {0}".format(prediction))
    

        

