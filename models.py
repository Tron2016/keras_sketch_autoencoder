from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D,Flatten,Reshape,Dropout,BatchNormalization,Activation,Conv2DTranspose
from keras.models import Model
from keras import regularizers
from keras import  backend as K

class sketh_model(object):

    def __init__(self):
        input_img = Input(shape=(225, 225, 1))
        x = Conv2D(64, (15, 15), strides=(3,3),padding='valid')(input_img)
        x= BatchNormalization(axis=-1)(x)
        x= Activation('relu')(x)
        x = MaxPooling2D((3, 3),strides=(2,2))(x)
        ##################################
        x = Conv2D(128, (5, 5),padding='valid')(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)
        ##################################
        x= Conv2D(256, (3, 3),padding='same')(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)

        x = Conv2D(256, (3, 3), padding='same')(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)

        x = Conv2D(256, (3, 3),padding='same')(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)
        ##################################
        x = Conv2D(512,(7,7),padding='valid')(x)
        x = BatchNormalization(axis=-1)(x)
        x = Flatten()(x)
        x = Activation('relu')(x)
        ##################################
        x = Dense(512)(x)
        x = Activation('relu')(x)
        ##################################
        x=Dense(32)(x)
        encoded = Activation('relu',name='encoded')(x)
        ##################################
        x= Dense(512)(x)
        x = Activation('relu')(x)
        ##################################
        x = Reshape((1, 1, 512))(x)
        x = Conv2DTranspose(256,(7,7),padding='valid')(x)
        x = Activation('relu')(x)

        x = Conv2DTranspose(256, (3, 3),strides=(2,2), padding='valid')(x)
        x = Activation('relu')(x)
        ##################################
        x = Conv2D(256, (3, 3), padding='same')(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)

        x = Conv2D(256, (3, 3), padding='same')(x)
        x = Activation('relu')(x)

        x = Conv2D(128, (3, 3), padding='same')(x)
        x = Activation('relu')(x)

        x = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='valid')(x)
        x = Activation('relu')(x)
        ##################################
        x = Conv2DTranspose(64, (5, 5), padding='valid')(x)
        x = Activation('relu')(x)

        x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='valid')(x)
        x = Activation('relu')(x)
        ##################################
        x = Conv2DTranspose(1, (15, 15), strides=(3, 3), padding='valid')(x)
        decoded = Activation('sigmoid')(x)
        self.autoencoder = Model(input_img, decoded)
        self.encoder = Model(input_img, encoded)



    def get_sketch_autoencoder(self):
        return self.autoencoder

    def get_sketch_encoder(self):
        return self.encoder


