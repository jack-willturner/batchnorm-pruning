Work in progress PyTorch implementation of [this paper](https://arxiv.org/abs/1802.00124).  

ISTA pruning process:
- [x] compute ista penalties for each layer
- [ ] scale weights of batchnorm layers
- [x] end to end training - implement SGD, batchnorm layer
- [ ] mask channels
- [ ] rescale weights of batchnorm
- [ ] finetune

## Personal Log
**14-Feb**: Added initial batchnorm. Plan is to write autograd function that takes an ista penalty and add this to the layer update for gamma.  
**15-Feb**: Subclassed SGD and added a line which includes ISTA penalty- now use two optimizers: one for batch norm layers and one for everything else. Need to alter batch norm to rescale gammas properly. Not sure how to get layer weight into running mean calculation but maybe it will be easy to figure out.  
**16-Feb**: Wrote ISTA calculation and added rho to scale it. Need to adapt calculation to only calculate for conv-bn pairs since right now it will calculate for any conv layer. I think the gammas can be scaled without altering batch norm itself: it should be possible to iterate over layers and alter the weights in the same way as Deep Compression. Also would like a way to watch Lasso converge so should work on that later. Opening project board to keep track of all this.
