# Last amended: 21/09/2017
# My folder: E:/cats_and_dogs/data
# Objective: Predicting cats and dogs--Kaggle 
#            https://www.kaggle.com/c/dogs-vs-cats
# Use spyder
# Ref:
#  https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html


'''

A. Arrange your data first
==========================

    Download data from: https://www.kaggle.com/c/dogs-vs-cats/data .
    Unzip train.zip and arrange its files as follows:
        data/
            train/
                dogs/
                    dog001.jpg
                    dog002.jpg
                    ...
                    dog1000.jpg
                cats/
                    cat001.jpg
                    cat002.jpg
                    ...
                    cat1000.jpg
           validation/
               dogs/
                   dog1001.jpg
                   dog1002.jpg
                   ...
                   dog1400.jpg
              cats/
                  cat1001.jpg
                  cat1002.jpg
                  ...
                  cat1004.jpg

    So we are picking up only 1400 files of each category for training.
    Arrangement of (only) 'training' files in this fashion has an advantage that
    keras automatically knows which images are of cats and which images are
    of dogs. It does automatic labeling of images; we do not have to specify
    explicitly in the code for building training model.

B. Training steps are as follows:
=================================
    
    1. Arrange training files as above. (Our total samples: nb_train_samples = 2000)
    2. Arrange validation files as above (V Samples: nb_validation_samples = 800)
    3. Specify location of all train folder and all vaidation folder
    4. Depending upon your backend (tensorflow/theano) decide
       the shape/format of your input image arrays. This is needed in CNN modeling.
    5. Build the CNN model
    6. Use ImageDataGenerator to augment train images. This is in two steps:
        i)  Specify configuration of possible changes in any image
        ii) Feed this configuration information in '.flow_from_directory'
            In '.flow_from_directory', specify:
                i)   Where are your images
                ii)  Do you also want to resize, if so, specify these
                iii) What batch-size to augment at a time;depends upon your memory
                iv)  Is classification binary or categorical?
    7. Use ImageDataGenerator to augment validation images. Again two steps
       as above. But we only resize validation images.
    8. Begin training using fit_generator():
        CNN fit_generator() takes these arguments:
        i)   train data generator (source of train images)
        ii)  validation data generator (source of validation images)
        iii) no of epochs
        iv)  Per epoch, batch-size for training
       
   9. After training has finished, save model weights to a '.h5' file and also
      save model configuration to a json file.
      
   ------------   
   Later, maybe, after some time
   10.Unzip test data file in a folder (within another folder. This is impt.).
   11.Configure test Image Data Generator
   12.Use above configuration and test-folder address, to create a test generator
   13.Load saved cnn model and load network weights in this model from saved h5 file
   14.Use predict_generator() to make predictions on test_generator.
   15.Evaluate predictions
   
C. About keras backend:
=======================
    
    The default keras configuration file is in folder:
        C:\Users\ashokharnal\.keras.  It looks like as below.
        The configurtion is as per the installed backend on your machine:
        tensorflow, theano or CNTK
            
            {
                    "image_data_format": "channels_last",
                    "epsilon": 1e-07,
                    "floatx": "float32",
                    "backend": "tensorflow"
                    }
            
            For 2D data (e.g. image), "channels_last" assumes (rows,cols,channels)
            while "channels_first" assumes (channels,rows,cols)
            (channels stand for RGB colour channels)
            
D. Prerequisites:
=================    
    Before attempting this problem, pl study Image Augmentation in Moodle at
    http://203.122.28.230/moodle/course/view.php?id=11&sectionid=166#section-9 

E. Note
=======
    This is a full code from building model to making predictions for test data.
    AUC is very less as no. of training epochs are just 5. The training consumes
    time but very less memory (around 50%) on an 8GB machine. Vary batch size
    to control memory usage.
           

'''

#%%                                A. Call libraries

# 0. Release memory
%reset -f

# 1.0 Data manipulation library
import pandas as pd

# 1.1 Call libraries for image processing
from keras.preprocessing.image import ImageDataGenerator

# 1.2, Libraries for building CNN model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

# 1.3.Keras has three backend implementations available: the TensorFlow,
#    the Theano, and CNTK backend.
from keras import backend as K

# 1.4 Save CNN model configuration
from keras.models import model_from_json

# 1.5 OS related
import os

# 1.6 For ROC plotting
import matplotlib.pyplot as plt

# 1.7 
from sklearn import metrics


#%%                            B. Define constants

# 2. Our constants
# 2.1 Dimensions to which our images will be adjusted
img_width, img_height = 150, 150

# 2.2 Data folder containing all training images, maybe in folders: cats and dogs
train_data_dir = 'E:/cats_and_dogs/data/train'

# 2.3 What is the total number of training images?
nb_train_samples = 23448   #11880 + 11568 =    23448


# 2.4 Data folder containing all validation images
validation_data_dir = 'E:/cats_and_dogs/data/validation'

# 2.5 What is the total no of validation samples
nb_validation_samples = 1019   # 619 + 931 =  1550


# 2.6 Batch size to train at one go:
batch_size = 24             # No of batches = 23448/24 = 977
                            # So per epoch we have 644 batches

# 2.7 How many epochs of training?
epochs = 5            # For lack of time, let us make it just 5.

# 2.8 No of test samples
test_generator_samples = 12500

# 2.9 For test data, what should be batch size
test_batch_size = 25



# 3. About keras backend
# 3.1 Can get backend configuration values, as:
K.image_data_format()          # Read .keras conf file to findout 
K.backend()

# 3.2 What is our backend? Decide data shape as per that.
#     Depth goes last in TensorFlow back-end, first in Theano
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:                                           # So, Tensorflow!
    input_shape = (img_width, img_height, 3)


#%%                         C. Define CNN Model


# 4. Create convnet model
#    con->relu->pool->con->relu->pool->con->relu->pool->flatten->fc->fc
    
# 4.1   Call model constructor and then pass on a list of layers    
model = Sequential()
# 4.2 2-D convolution layers with 32 filters and kernel-size 3 X 3
#         Default strides is (1, 1)
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
# 4.3 For each neuron in the convolved network, assign an activation function
model.add(Activation('relu'))
# 4.4 pool_size:  max pooling window size: (2,2)
model.add(MaxPooling2D(pool_size=(2, 2)))


# 4.5
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


# 4.6
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 4.7 Flattens the input. Does not affect the batch size.
model.add(Flatten())

# 4.8 Dense layer having 64 units 
#     dimensionality of the output space.
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# 4.9 Dense layer having 1 unit
#     dimensionality of the output space.
model.add(Dense(1))
model.add(Activation('sigmoid'))

# 4.10 Compile model
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])



#%%                            D. Create Data generators


## 5. Image augmentation
# 5.1 This is the augmentation configuration for training samples
train_datagen = ImageDataGenerator(
    rescale=1. / 255,           # Normalize colour intensities in 0-1 range
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)


# 5.1 Augmentation configuration we will use for testing:
#     only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

# 5.3 Image generation: flow_from_directory(directory) takes the path to a
#                       directory, and generates batches of augmented/normalized
#                       data. Yields batches indefinitely, in an infinite loop.

# 5.3.1 train data
train_generator = train_datagen.flow_from_directory(
    train_data_dir,                         # Data folder containing cats and dogs folders
    target_size=(img_width, img_height),    # Resize images
    batch_size=batch_size,                  # Pick images in batches
    class_mode='binary')                    # Data labels are binary in nature (dogs and cats)

# 5.3.2 validation data
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,                   # Which folder has validation data
    target_size=(img_width, img_height),   # Resize images
    batch_size=batch_size,                 # batch size to augment at a time
    class_mode='binary')                   # Data has binary classes



#%%                           E. Fit model & save CNN network weights


## 6. Model fitting
# 6.1 Fit the model as also evaluate in batches
model.fit_generator(
    generator = train_generator,          # First argument is always training data generator
    steps_per_epoch=nb_train_samples // batch_size, # How many batches per epoch?
    epochs=epochs,                        # No of epochs
    validation_data=validation_generator, # Get validation data from validation generator
    verbose = 1,                          # Do not be silent
    validation_steps=nb_validation_samples // batch_size) 


## 7. Model saving
# 7.1 Install h5py using Anaconda 
# 7.2 Save CNN model weights to a file
#     The h5py package is a Pythonic interface to HDF5 binary data format.
#     It lets you store huge amounts of numerical data, and easily manipulate
#     that data from NumPy. For example, you can slice into multi-terabyte
#     datasets stored on disk, as if they were real NumPy arrays. 
#     Thousands of datasets can be stored in a single file, categorized and
#     tagged however you want.
model.save_weights('first_try.h5')
os.getcwd()   # Where are these saved: 'C:\\Users\\ashokharnal'


# 7.3 Save your CNN model structure to a file, cnn_model.json
#  Get your model in json format
cnn_model = model.to_json()
cnn_model


# 7.4 Now save this json formatted data to a file on hard-disk
#     File name: cnn_model.json. File path: check with setwd()
# 7.4.1. Open/create file in write mode
json_file = open("cnn_model.json", "w")
# 7.4.2 Write to file
json_file.write(cnn_model)
# 7.4.3 Close file
json_file.close()



#%%                          F. Load model and model weights


## 8. Later

# 8.1 Open saved model file in read only mode
#     Just
os.chdir("C:\\Users\\ashokharnal")
json_file = open('cnn_model.json', 'r')
loaded_model_json = json_file.read()
loaded_model_json            # Model structure in file: loded_model_json  
json_file.close()

# 8.2 Create CNN model from the file: loaded_model_json
cnn_model = model_from_json(loaded_model_json)


# 8.3 load saved weights into new model
cnn_model.load_weights("first_try.h5")

 
# 8.4 Compile the model
cnn_model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])


#%%                         G. Make predictions on test data

# 9 Where is the directory which contains ANOTHER directory
#   containing your test images
test_data_dir = 'E:/cats_and_dogs/test'


# 9.1 Augmentation configuration for test dataset. 
#     Only rescaling as we did for validation data
test_datagen = ImageDataGenerator(rescale=1. / 255)


# 9.2 Create test data generator
test_generator = test_datagen.flow_from_directory(
        test_data_dir,                         # Which folder has validation data
        target_size=(img_width, img_height),   # Resize images
        batch_size=batch_size,                 # batch size to augment at a time
        class_mode='binary')                   # Data has binary classes



# 10. Make predictions
predictions = cnn_model.predict_generator(
        test_generator,
        steps=int(test_generator_samples/float(test_batch_size)), # all samples once
        verbose =1 
        )


# 10.1 See arrays of predictions
predictions
predictions[0:10]


# 11. Unfortunately Kaggle is not allowing submissions.
#     I have manually compiled a file looking at 300 images out of 12500 
#     images in the test folder.
actual=pd.read_csv("E:/cats_and_dogs/actual_result300.csv", header = 0)
actual.head()

# 11.1 Add predictions column to this data frame
actual['new'] = predictions[0:300]     

# 12. Evaluate accuracy
fpr, tpr, _ = metrics.roc_curve(actual['label'], actual['new'])

# 12.1 AUC
metrics.roc_auc_score(actual['label'], actual['new'])

# 12.2 ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange')
plt.show()

############ END ################
