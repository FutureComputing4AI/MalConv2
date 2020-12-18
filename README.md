# Classifying Sequences of Extreme Length with Constant Memory Applied to Malware Detection (a.k.a., MalConv2)

This is the PyTorch code implementing the approaches from our AAAI 2021 paper [Classifying Sequences of Extreme Length with Constant Memory Applied to Malware Detection](https://arxiv.org/abs/2012.09390). Using it, you can train the original MalConv model faster and using less memory 
than before. You can also train our new MalConv with “Global Channel Gating” (GCG), what allows MalConv to learn feature interactions from across the entire inputs. 

## Code Organization

This is research quality code that has gone through some quick edits before going online, and comes with no warranty. The rough outline of the files in this repo. 

### binaryLoader.py 

`binaryLoader.py` contains the functions we use for loading in a dataset of binaries, and supports un-gziping them on the fly to reduce IO costs. It also includes a sampler that is used to create batches of similarly sized files to minimize excess 
padding used during training. This assumes the input dataset is already in sorted order by file size. 

### checkpoint.py

This contains code used to perform gradient checkpointing for reduced memory usage. This is optional and generally not necessary for our MalConv* models now, but was used during experimentation. 

### LowMemConv.py 

LowMemConv is the base class that implementations extend to obtain the fixed-memory pooling we introduced. This is provided by `seq2fix` function, which does the work of applying the convolution in chunks, tracking the winners, and then grouping the 
winning slices to run over with gradient calculations on. 

The user extends `LowMemConvBase`, implementing the `processRange` function, which applies whatever convolutional strategy they desire to a range of bytes. The `determinRF` function is used to determine the receptive field size by iteratively testing 
for the smallest input size that does not error, so that we know how to size our chunk sizes later. 


### MalConvGCT_nocat.py & MalConvGCTTrain.py

MalConvGCT_nocat implements the new contribution of our paper, using the GCT attention. An older file, MalConvGCT uses this pooling but with a concatenation at the end. 

The `MalConvGCTTrain.py` is the sister file that will train a `MalConvGCT` object. 

The associated "*Train.py" functions allow for training these models. AvastTyleConv implements the max pool version of the Avast architecture, and MalConvML implement a multiple layer version of MalConv that were used in ablation testing. MalConv.py 
implements the original MalConv using our new low memory approach. 

### malconvGCT_nocat.checkpoint

This file contains the weights for the GCT model from our paper’s results. It has some extra parameters that were never used due to some lines left commented in durning model training. It also has an off-by-one “bug” that says its the 21’st epoch 
instead of the 20’th. 

To load this file, you want to have code that looks like:

```python
from MalConvGCT_nocat import MalConvGCT

mlgct = MalConvGCT(channels=256, window_size=256, stride=64,)
x = torch.load("malconvGCT_nocat.checkpoint.checkpoint")
mlgct.load_state_dict(x['model_state_dict'], strict=False)
```

### AvastStyleConv.py

This implements a network in the style of Avast’s CNN from 2018, but replacing average pooling with our temporal max pooling for speed. 

### MalConv.py

Implements the original MalConv network with our faster training/pooling. 


### MalConvML.py

This file contains an alternative experiment approach to training with more layers, but never worked well. 

### ContinueTraining.py

This file can be used to resume the training of models from a given checkpoint. 

### OptunaTrain.py 

This file is used to do training why a hyper-parameter search. 

### Non-Neg options

The non-negative training currently present is faulty, as it allows you to do such training with a softmax output, which is technically incorrect. Please do not use it. 


## Citations

If you use the MalConv GCT algorithm or code, please cite our work! 

```
@inproceedings{malconvGCT,
author = {Raff, Edward and Fleshman, William and Zak, Richard and Anderson, Hyrum and Filar, Bobby and Mclean, Mark},
booktitle = {The Thirty-Fifth AAAI Conference on Artificial Intelligence},
title = {{Classifying Sequences of Extreme Length with Constant Memory Applied to Malware Detection}},
year = {2021},
url={https://arxiv.org/abs/2012.09390},
}
```

## Contact 

If you have questions, please contact 

Mark Mclean <mrmclea@lps.umd.edu>
Edward Raff <edraff@lps.umd.edu>
Richard Zak <rzak@lps.umd.edu>

