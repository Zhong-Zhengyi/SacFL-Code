
# SacFL: Self-Adaptive Federated Continual Learning for Resource-Constrained End Devices
Welcome to the repository! This is the detailed implementation of our project, SacFL. We hope this code will serve as a valuable resource for understanding our work and its application. Thank you for your interest and support!
![img_1.png](img_1.png)
## Dependencies
```
torch==2.2.1+cu121
numpy==1.24.3
scikit-learn==1.3.2
objgraph==3.6.1
pandas==2.0.2
torchvision==0.17.0+cu121
joblib==1.3.2
transformers==4.37.2
```
## Datasets
### Image Datasets
-[THUCNews](http://thuctc.thunlp.org/)  
-[Cifar10](https://www.cs.toronto.edu/~kriz/cifar.html)  
-[Cifar100](https://www.cs.toronto.edu/~kriz/cifar.html)  
-[FashionMNIST](https://www.worldlink.com.cn/en/osdir/fashion-mnist.html)


## Quick start
```angular2html
python federated_run.py --model='LeNet_FashionMNIST' --paradigm='sacfl' --scenario='class'  --global_epoch=3
```
