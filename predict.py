import numpy as np
import time
import cv2
from models import sketh_model
from keras.preprocessing import image
from read_activations import get_one_activation

img_size=(225,225)

def preprocess_input(x):
    x /= 255.
    return x


def load_img(img_path):
    img = image.load_img(img_path,grayscale=True, target_size=img_size)
    x = image.img_to_array(img)
    cv2.imshow('input', x)
    cv2.imwrite('a.png',x)
    x = np.expand_dims(x, axis=0)
    return preprocess_input(x)

def load_autoencoder():
    s_model = sketh_model()
    autoencoder = s_model.get_sketch_autoencoder()
    autoencoder.load_weights('./autoencoder.hdf5')
    return autoencoder

def predict_activation(model,file_path):
    input = load_img(file_path)
    start = time.clock()
    activation = get_one_activation(model, input, model.get_layer(name='encoded'))
    print(activation)
    end = time.clock()
    print('Prediction took: {}'.format(end - start))
    return activation

if __name__ == '__main__':
    s_model=sketh_model()


    autoencoder=s_model.get_sketch_autoencoder()


    autoencoder.load_weights('./autoencoder.hdf5')

    input=load_img('./test.png')

    start = time.clock()
    out = autoencoder.predict(input)
    activation=get_one_activation(autoencoder,input,autoencoder.get_layer(name='encoded'))
    print(activation)


    end = time.clock()
    out_img=(out[0]*255.).astype('uint8')



    print('Prediction took: {}'.format( end - start))
    cv2.imshow('output',out_img)
    cv2.waitKey(5000)
