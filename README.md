# keras_sketch_autoencoder

This is a implementation of sketch image autoencoder based on tensorflow and keras. The encoder model comes from [Sketch-a-Net](http://homepages.inf.ed.ac.uk/thospeda/papers/yu2016sketchanet.pdf)

## For Train 

You can download sketch dataset from [TU Berlin](http://cybertron.cg.tu-berlin.de/eitz/projects/sbsr/)

Put the sketch images into:
```

data/train

data/valid

```
simply run:
```

python autoencoder.py

```
## For Predict
```

python predict.py

```
## Reference
[1] Yu Q, Yang Y, Song Y Z, et al. Sketch-a-net that beats humans[J]. arXiv preprint arXiv:1501.07873, 2015.
