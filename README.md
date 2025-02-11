# SVAttack
SVAttack: Spatial-Viewpoint Transfer Attack on Graph Convolutional Skeleton Action Recognition

# Prerequisites
> python >= 3.9  
  torch >= 2.3.0

# Data Preparation
## NTU RGB+D 60 and 120
- Request dataset here: https://rose1.ntu.edu.sg/dataset/actionRecognition
- Download the skeleton-only datasets:  
  nturgbd_skeletons_s001_to_s017.zip (NTU RGB+D 60)  
  nturgbd_skeletons_s018_to_s032.zip (NTU RGB+D 120)

You can use the SVAttack file to generate adversarial examples and save them as files, which you will use to perform cross-model transfer attacks in the next step
# Training
```python
python SVAttack.py --config ./configs/svattack/stgcn-ntu60-cs.yaml
```

In this step, you need to set the configuration parameters for the migration test. The migration test will read the adversarial example path from the configuration file and set the mn parameter to the target model
# Testing
```python
python TransferAttack.py --config ./configs/svattack/stgcn-ntu60-cs.yaml --mn agcn
```