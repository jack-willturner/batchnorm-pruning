Work in progress PyTorch implementation of [this paper](https://arxiv.org/abs/1802.00124).

## Log
**14-Feb**: Added initial batchnorm. Plan is to write autograd function that takes an ista penalty and add this to the layer update for gamma.
**15-Feb**: Subclassed SGD and added a line which includes ISTA penalty- now use two optimizers: one for batch norm layers and one for everything else. Need to alter batch norm to rescale gammas properly. Not sure how to get layer weight into running mean calculation but maybe it will be easy to figure out. 
