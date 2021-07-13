# experiments



## Code Structure
* [`MedMNIST2D/`](./MedMNIST2D/): training and evaluation scripts of MedMNIST2D
  * [`models.py`](./MedMNIST2D/models.py): *ResNet-18* and *ResNet-50* models
  * [`train_and_eval_pytorch.py`](./MedMNIST2D/train_and_eval_pytorch.py): training and evaluation script implemented with PyTorch
  * [`train_and_eval_autokeras.py`]('./MedMNIST2D/train_and_eval_autokeras.py'):  training and evaluation script of Autokeras
  * [`train_and_eval_autosklearn.py`](./MedMNIST2D/train_and_eval_autosklearn.py): training and evaluation script of auto-sklearn
  * [`eval_google_automl_vision.py`](./MedMNIST2D/eval_google_automl_vision.py): evaluation script of models trained by Google AutoML Vision

* [`MedMNIST3D/`](./MedMNIST3D/): training and evaluation scripts of MedMNIST3D

  * [`models.py`](./MedMNIST3D/models.py): *ResNet-18* and *ResNet-50* models

  * [`train_and_eval_pytorch.py`](./MedMNIST3D/train_and_eval_pytorch.py): training and evaluation script implemented with PyTorch

  * [`train_and_eval_autokeras.py`]('./MedMNIST3D/train_and_eval_autokeras.py'):  training and evaluation script of Autokeras

  * [`train_and_eval_autosklearn.py`](./MedMNIST3D/train_and_eval_autosklearn.py): training and evaluation script of auto-sklearn

    


## Requirements

The code requires only common Python environments for machine learning; Basicially, it was tested with

- Python 3 (Anaconda 3.6.3 specifically)
- PyTorch==1.6.0
- autokeras\==1.0.15
- auto-sklearn\==0.10.0
- tensorflow==2.3.0

Higher (or lower) versions should also work (perhaps with minor modifications).

For *MedMNIST3D*, the code requires [ACSConv](https://github.com/M3DV/ACSConv)

