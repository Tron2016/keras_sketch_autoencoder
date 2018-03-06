# keras_sketch_autoencoder

This is a implementation of sketch image autoencoder based on tensorflow and keras. The encoder model comes from [Sketch-a-Net](http://homepages.inf.ed.ac.uk/thospeda/papers/yu2016sketchanet.pdf)

#For Train 

You can download sketch data set from [TU Berlin](http://cybertron.cg.tu-berlin.de/eitz/projects/sbsr/)

put the sketch images into:

data/train

data/valid

simply run:

python autoencoder.py

#For Predict

python predict.py