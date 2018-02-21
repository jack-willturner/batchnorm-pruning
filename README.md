# [Rethinking the Smaller-Norm-Less-Informative Assumption in Channel Pruning of Convolution Layers](https://arxiv.org/abs/1802.00124)
A PyTorch implementation of [this paper](https://arxiv.org/abs/1802.00124).  

## Usage
I haven't included any code for transfer learning/ using pretrained models, so everything here must be done from scratch. 
You will have to rewrite your models to use my extended version of batch normalization, so any occurences of `nn.BatchNorm2d`
should be replaced with `bn.BatchNorm2dEx`. I have included a few examples in the `models` folder. Note that in the forward pass
you need to provide the `weight` from the last convolution to the batchnorm (e.g. `out = self.bn1(self.conv1(x), self.conv1.weight)`.  

I will add command line support for hyperparameters soon, but for now they will have to be altered in the `main` script itself. Currently the default is set to train ResNet-18; this can easily be swapped out for another model.

```bash
python main.py
```

## Results
Coming soon...  
Things that could probably go here:
- [ ] table replicated ResNet20 results on CIFAR-10 
- [ ] graph of num params/ accuracy tradeoff for different sparse penalties
