# BOAT
Implementation for paper "BOAT: Binary Optimizer with Adaptive Thresholds".

The code is based on https://github.com/itayhubara/BinaryNet.pytorch and https://github.com/gushan/SGDAT.  
Please install torch and torchvision by following the instructions at: http://pytorch.org/.   

### Run Command
#### On the BinaryNet
#### SGD in cifar10
python main_binary.py --model vgg_cifar10_binary --save vgg_cifar10_SGD --dataset cifar10 --bin_regime "{0: {'optimizer': 'SGD','lr':1e-4}}"  --binarization det --input_size 32 --epochs 200 -b 256
#### SGDM in cifar10
python main_binary.py --model vgg_cifar10_binary --save vgg_cifar10_SGDM --dataset cifar10 --bin_regime "{0: {'optimizer': 'SGD','lr':1e-4,'momentum':0.9}}"  --binarization det --input_size 32 --epochs 200 -b 256
#### Bop in cifar10
python main_binary.py --model vgg_cifar10_binary --save vgg_cifar10_Bop --dataset cifar10 --bin_regime "{0: {'optimizer': 'Bop','gamma':1e-4,'threshold':1e-8}}" --binarization det --input_size 32 --epochs 200 -b 256
#### Bop2ndOrder in cifar10
python main_binary.py --model vgg_cifar10_binary --save vgg_cifar10_Bop2ndOrder --dataset cifar10 --bin_regime "{0: {'optimizer': 'Bop2ndOrder','gamma':1e-7,'sigma':1e-3,'threshold':1e-6}}" --binarization det --input_size 32 --epochs 200 -b 256
#### SGDT in cifar10
python main_binary.py --model vgg_cifar10_binary --save vgg_cifar10_SGDT --dataset cifar10 --bin_regime "{0: {'optimizer': 'SGD','lr':1e-4}}"  --binarization threshold --threshold 1e-8 --input_size 32 --epochs 200 -b 256
#### SGDAT in cifar10
python main_binary.py --model vgg_cifar10_binary --save vgg_cifar10_SGDAT --dataset cifar10 --bin_regime "{0: {'optimizer':'SGDAT','lr':1e-4,'threshold':1e-7}}" --binarization det --input_size 32 --epochs 200 -b 256
#### SGDAT in cifar100
python main_binary.py --model vgg_cifar100_binary --save vgg_cifar100_SGDAT --dataset cifar100 --bin_regime "{0: {'optimizer':'SGDAT','lr':1e-4,'threshold':1e-7}}" --binarization det --input_size 32 --epochs 200 -b 256
#### SGDAT in tiny_imagenet
python main_binary.py --model vgg_tiny_imagenet_binary --save vgg_tiny_imagenet_SGDAT --dataset tiny_imagenet --bin_regime "{0: {'optimizer':'SGDAT','lr':1e-4,'threshold':1e-7}}" --binarization det --input_size 64 --epochs 100 -b 256
#### BOAT(ours) in cifar10
python main_binary.py --model vgg_tiny_imagenet_binary --save vgg_cifar10_BOAT --dataset cifar10 --bin_regime "{0: {'optimizer':'BOAT','eta':0.1,'weight_decay':1e-3}}" --binarization det --input_size 32 --epochs 200 -b 256
#### BOAT(ours) in cifar100
python main_binary.py --model vgg_tiny_imagenet_binary --save vgg_cifar100_BOAT --dataset cifar100 --bin_regime "{0: {'optimizer':'BOAT','eta':0.1,'weight_decay':1e-3}}" --binarization det --input_size 32 --epochs 200 -b 256
#### BOAT(ours) in tiny_imagenet
python main_binary.py --model vgg_tiny_imagenet_binary --save vgg_tiny_imagenet_BOAT --dataset tiny_imagenet --bin_regime "{0: {'optimizer':'BOAT','eta':0.13,'weight_decay':1e-3}}" --binarization det --input_size 64 --epochs 100 -b 256

#### On the Resnet-18
#### SGDAT in cifar10
python main_binary.py --model resnet_binary --save resnet_cifar10_SGDAT --dataset cifar10 --bin_regime "{0: {'optimizer':'SGDAT','lr':1e-4,'threshold':1e-7}}" --binarization det --input_size 32 --epochs 200 -b 256
#### SGDAT in cifar100
python main_binary.py --model resnet_binary --save resnet_cifar100_SGDAT --dataset cifar100 --bin_regime "{0: {'optimizer':'SGDAT','lr':1e-4,'threshold':1e-7}}" --binarization det --input_size 32 --epochs 200 -b 256
#### SGDAT in tiny_imagenet
python main_binary.py --model resnet_binary --save resnet_tiny_imagenet_SGDAT --dataset tiny_imagenet --bin_regime "{0: {'optimizer':'SGDAT','lr':1e-4,'threshold':1e-7}}" --binarization det --input_size 64 --epochs 100 -b 256
#### BOAT(ours) in cifar10
python main_binary.py --model resnet_binary --save resnet_cifar10_BOAT --dataset cifar10 --bin_regime "{0: {'optimizer':'BOAT','eta':0.1,'weight_decay':1e-3}}" --binarization det --input_size 32 --epochs 200 -b 256
#### BOAT(ours) in cifar100
python main_binary.py --model resnet_binary --save resnet_cifar100_BOAT --dataset cifar100 --bin_regime "{0: {'optimizer':'BOAT','eta':0.1,'weight_decay':1e-3}}" --binarization det --input_size 32 --epochs 200 -b 256
#### BOAT(ours) in tiny_imagenet
python main_binary.py --model resnet_binary --save resnet_tiny_imagenet_BOAT --dataset tiny_imagenet --bin_regime "{0: {'optimizer':'BOAT','eta':0.13,'weight_decay':1e-3}}" --binarization det --input_size 64 --epochs 100 -b 256
### Experimental Result
<img width="866" alt="image" src="https://github.com/user-attachments/assets/95fc4938-b0c7-4e3c-8b2d-eb0a3ab08c50" />

