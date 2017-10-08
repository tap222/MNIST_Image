"""
Last amended: 20th Sep, 2017
My Data folder: E:\cats_and_dogs
Data file: cat.0.jpg (any other file will also do)
    
Objective:
    Augmenting images:
      Randomly generate 21 augmented image files from a single image file
    
Using keras ImageDataGenerator
Ref: https://keras.io/preprocessing/image/  
     https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html  
    
"""

# 1. You need to install package, pillow, using Anaconda after running it as administrator
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# 2. glob module finds all the pathnames matching a specified pattern
import glob
# 2.1 OS related operations
import os

# 3. Where is my image placed
os.chdir("E:\\cats_and_dogs\\data\\train\\cats")

# 4. Instantiate an image data generator object
datagen = ImageDataGenerator(     # Generate more images from existing images
        rotation_range=40,        # Degree range for random rotations
        width_shift_range=0.2,    # Range for random horizontal shifts.
        height_shift_range=0.2,   # Range for random vertical shifts.
        shear_range=0.2,       # Shear angle in counter-clockwise direction as radians
        zoom_range=0.2,        #  Range for random zoom 
        horizontal_flip=True,  # Randomly flip inputs horizontally
        fill_mode='nearest'    # fill_mode is the strategy used for filling 
        )                      #  in newly created pixels, which can appear after 
                               #  a rotation or a width/height shift. 
                               
# 5. Load only one image of cat. Can be any image of a building, scenery etc.
img = load_img('cat.0.jpg')  # 'img' is a Python Image Library(PIL) image
img                          #    Display actual imqge of cat
# 5.1
type(img)

# 6. Transform PIL 'img' to arrays of colour components, pixel-by-pixel
x = img_to_array(img) 
# 6.1 
x                      # this is a Numpy array with shape (3, 150, 150)
x.shape                # 374 X 500 X 3
                       # For each one of the R,G, B colours, we have
                       #  374 rows or img_ht. of 500 pixel values or img_width. 
                       #   That is img size is 374 X 500 pixels= 187000 pixles
# 6.2                       
x.ndim
# 6.3 'Add' two tuples: Try this: (2,) + (3,5,7)
x = x.reshape((1,) + x.shape)  
x.shape                # this is a Numpy array with shape (1, 374, 500, 3)
                       # That is 1 image having shape: (374, 500, 3)         

# 7. We will store all transformed files in folder 'preview' 
#     under current folder
#      First delete existing files, if any, in 'preview' folder
# glob.glob(pathname): Return possibly-empty list of path names that match pathname
files = glob.glob("E:\\cats_and_dogs\\data\\train\\cats\\preview\\*")
# 7.1
files

# 8
for f in files:          # For every such file
    os.remove(f)         # Delete it


# 9. .flow() randomly picks up a batch from generator and augments it 
#      and yields a batch, at a time    
abc = datagen.flow(x,
             batch_size=16,          # Augment in batches. Default is 32
             save_to_dir='preview',  # Save to this folder
             save_prefix='cat',      #  While saving apply this prefix to file
             save_format='jpeg')     
# 9.1
type(abc)           # It is a numpy array iterator

# 9.2 Following will create images indefinitely
list(abc)

# 9.3 Picks-up one element (image) at a time
abc.next()

# 10. The .flow() command below generates batches of
#      randomly transformed images and saves the results
#       to the `preview directory. 
#        .flow() command takes numpy data & label arrays,
#          and generates batches of augmented/normalized data.
#           Yields batches indefinitely, in an infinite loop.
i = 0
# 10.1
for item in abc:        # Extracts one image-item first from iterator
                         #  and saves it to 'preview' folder
    input("See your preview folder. Press a key to proceed further")
    i += 1
    if i > 8:
        break  # otherwise the generator would loop indefinitely

# 10.2 See generated images in folder: E:\cats_and_dogs\data\train\cats\preview     
        
####### END 
