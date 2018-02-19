# [Rethinking the Smaller-Norm-Less-Informative Assumption in Channel Pruning of Convolution Layers](https://arxiv.org/abs/1802.00124).
Work in progress PyTorch implementation of [this paper](https://arxiv.org/abs/1802.00124).  

## Usage
I haven't included any code for transfer learning/ using pretrained models, so everything here must be done from scratch. You will have to rewrite your models to use my extended version of batch normalization, so any occurences of `nn.BatchNorm2d` should be replaced with `BatchNorm2dEx`. I have included a few examples in the `models` folder. 

I will add command line support for hyperparameters soon, but for now they will have to be altered in the `main` script itself. To run on your altered model.

```bash
python main.py --model=ResNet18 
```

## Results
Coming soon...
Things that could probably go here:
- [ ] table replicated ResNet20 results on CIFAR-10 
- [ ] graph of num params/ accuracy tradeoff for different sparse penalties

## Left to do 
ISTA pruning process:
- [x] compute ista penalties for each layer
- [x] scale weights of batchnorm layers
- [x] end to end training - implement SGD, batchnorm layer
- [ ] mask channels
- [x] rescale weights of batchnorm
- [x] finetune

## Personal Log
**14-Feb**: Added initial batchnorm. Plan is to write autograd function that takes an ista penalty and add this to the layer update for gamma.  
**15-Feb**: Subclassed SGD and added a line which includes ISTA penalty- now use two optimizers: one for batch norm layers and one for everything else. Need to alter batch norm to rescale gammas properly. Not sure how to get layer weight into running mean calculation but maybe it will be easy to figure out.  
**16-Feb**: Wrote ISTA calculation and added rho to scale it. Need to adapt calculation to only calculate for conv-bn pairs since right now it will calculate for any conv layer. I think the gammas can be scaled without altering batch norm itself: it should be possible to iterate over layers and alter the weights in the same way as Deep Compression. Also would like a way to watch Lasso converge so should work on that later. Opening project board to keep track of all this.  
**19-Feb**: Would be good to have hyperband support since there are two extra hyperparams `alpha` and `rho`. Everything done now except for post-processing to remove channels. 
