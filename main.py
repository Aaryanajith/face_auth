import tensorflow
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import pickle
from keras.models import Sequential # type: ignore
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense # type: ignore
from keras.callbacks import EarlyStopping # type: ignore
import time
import numpy as np
from keras.preprocessing import image # type: ignore
import os
import glob

training_images = r"/home/interstellar/face_auth/Face Images/Final Training Images"

train_gen = ImageDataGenerator(
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

test_gen = ImageDataGenerator()

# Generating training data
training_data = train_gen.flow_from_directory(
    training_images, 
    target_size = (100,100),
    batch_size = 30,
    class_mode = 'categorical'
)

# generating test data
testing_data = test_gen.flow_from_directory(
    training_images, 
    target_size = (100,100),
    batch_size = 30,
    class_mode = 'categorical'
)

testing_data.class_indices

Train_class = training_data.class_indices

Result_class = {}
for value_tag, face_tag in zip(Train_class.values(),Train_class.keys()):
    Result_class[value_tag] = face_tag

with open(r'/home/interstellar/face_auth/Face Images/ResultMap.pkl','wb') as Final_mapping:
    pickle.dump(Result_class,Final_mapping)

print("Mapping of Face and its numeric value",Result_class)

Output_Neurons=len(Result_class)

Model = Sequential()

Model.add(Conv2D(16, kernel_size=(5,5), strides=(1,1), input_shape = (100,100,3),activation='relu'))

Model.add(MaxPool2D(pool_size=(2,2)))

Model.add(Conv2D(32,kernel_size=(3,3),strides=(1,1),activation='relu'))
Model.add(MaxPool2D(pool_size=(2,2)))

Model.add(Flatten())

Model.add(Dense(64,activation='relu'))
Model.add(Dense(Output_Neurons,activation='softmax'))

Model.compile(loss='categorical_crossentropy',optimizer = 'adam',metrics = ['Accuracy'])

call = EarlyStopping(min_delta=0.005, patience=5, verbose=1)

StartTime=time.time()

Model.fit(training_data, epochs = 30, validation_data=testing_data, callbacks=call)

Endtime = time.time()
print('Total Training Time taken: ',round((Endtime-StartTime)/60),'Minutes')

image_path = r"/home/interstellar/face_auth/Face Images/Final Training Images/face4/image_0054_Face_1.jpg"

test_image=image.load_img(image_path,target_size=(100, 100))
test_image=image.img_to_array(test_image)

test_image=np.expand_dims(test_image,axis=0)

result=Model.predict(test_image,verbose=0)

print('####'*10)
print('Prediction is: ',Result_class[np.argmax(result)])

pics = r"/home/interstellar/face_auth/Face Images/Final Testing Images"
img_paths = glob.glob(os.path.join(pics,'**','*.jpg'))

print(img_paths[0:5])
print('*'*50)

for path in img_paths:
    test_image = image.load_img(path,target_size=(100,100))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image,axis =0)
    result = Model.predict(test_image,verbose=0)
    print('Prediction: ',Result_class[np.argmax(result)])