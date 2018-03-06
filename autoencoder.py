from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D,Flatten,Reshape,Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras import  backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint


import numpy as np
from models import sketh_model


img_size=(225,225)
b_size=32

train_dir='./data/train'
validation_dir='./data/valid'

file_path='./autoencoder.hdf5'

def apply_mean(image_data_generator):
    """Subtracts the dataset mean"""
    image_data_generator.mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape((3, 1, 1))

def get_train_datagen(*args, **kwargs):
    idg = ImageDataGenerator(*args, **kwargs)
    return idg.flow_from_directory(train_dir, color_mode="grayscale",target_size=img_size,batch_size=b_size, class_mode='input',shuffle=True)


def get_validation_datagen(*args, **kwargs):
    idg = ImageDataGenerator(*args, **kwargs)
    return idg.flow_from_directory(validation_dir, color_mode="grayscale",target_size=img_size,batch_size=b_size, class_mode='input',shuffle=True)



s_model=sketh_model()


autoencoder=s_model.get_sketch_autoencoder()
#autoencoder.load_weights('autoencoder.hdf5')
i=0
for layer in autoencoder.layers:
    print("layer number:{}".format(i))
    print(layer.get_config())
    print(layer.output_shape)
    i+=1

autoencoder.compile(optimizer=Adam(lr=1e-6),loss='binary_crossentropy')
checkpoint = ModelCheckpoint(file_path,monitor='val_loss',verbose=1,save_best_only=True,mode='min')

x_train=get_train_datagen(rescale=1./255.0, horizontal_flip=True, vertical_flip=True)
x_valid=get_validation_datagen(rescale=1./255.0)
image_numbers=x_train.samples
autoencoder.fit_generator(x_train, steps_per_epoch=image_numbers/b_size, epochs=1000, \
                validation_data=x_valid,validation_steps=10,callbacks=[TensorBoard(log_dir='./tmp'),checkpoint])
