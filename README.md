# DatesNet: Facial Emotion Recognition 
Implementation of Facial Emotion Recognition model trained and evaluated on FER+ dataset. The DatesNet is a UNet with Residual blocks ending with a Self-Attention block.

## Model Architecture


**Link to FER+ dataset:** [FER+](https://github.com/microsoft/FERPlus/tree/master)

## Comparison with BaseLine and PAtt-Lite model
| Architecture             | Model File                                                     | Confusion Matrix                                                                  |
|--------------------------|----------------------------------------------------------|-----------------------------------------------------------------------------------|
| DatesNet (trained on FER+) | [datesnet_model_datesnet.pth](./checkpoints/datesnet_model_datesnet.pth)            | <img src ="./output/confusion_matrix_datesnet_8cls.png" width="600" height="550"> |
| PAtt-Lite               | [link](https://github.com/JLREx/PAtt-Lite)               | ![sota_cm](./output/patt_lite.png)                                                |
| Baseline Model on FER+  | [link](https://github.com/microsoft/FERPlus/tree/master) | ![ref conf mat](./output/ref_conf_mat.png)                                        | 

## Training
The training code needs the proper configuration set in the ***config/datesnet_config.py***. The 
model uses PyTorch. To create the environment, run
````commandline
create_and_set_environment.bat
````
If all your configurations are set in the ***config/datesnet_config.py***, then run
````python
python train.py
````
Training will log the loss from training and validation in the ***logdir*** folder. 
Since Tensorboard is to log the data, run the following command in another cmd inside the directory.
Make sure that you are inside the parent directory, i.e. inside ***/datesnet/***.
````commandline
conda activate datesnet
tensorboard --logdir logdir
````
The training code will save 2 models, one ***datesnet_model_cont.pth*** and ***datesnet_model.pth***. 
The ***datesnet_model.pth*** is the final mode saved when the best validation loss is achieved.

## Evaluation
To evaluate the model, run the following code
````python
python evaluate.py
````
This code will save the **confusion matrix** and the **accuracy, precision, recall, F1-score** in the ***/output/*** folder

Alternatively, the jupyter notebook can be used as well to evaluate the model. 
Open the jupyter notebook by running the following in a cmd while inside ***/datesnet/***,
````commandline
conda activate datesnet
jupyter notebook
````
