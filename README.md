# MedMNIST Experiments 

Training and evaluation scripts to reproduce both 2D and 3D experiments in our [MedMNIST](https://github.com/MedMNIST/MedMNIST/) paper, including PyTorch, auto-sklearn, AutoKeras and Google AutoML Vision together with their weights ;)


# Code Structure
* [`MedMNIST2D/`](./MedMNIST2D/): training and evaluation scripts of MedMNIST2D
  * [`models.py`](./MedMNIST2D/models.py): *ResNet-18* and *ResNet-50* models (for small-image datasets like CIFAR-10/100)
  * [`train_and_eval_pytorch.py`](./MedMNIST2D/train_and_eval_pytorch.py): training and evaluation script implemented with PyTorch
  * [`train_and_eval_autokeras.py`](./MedMNIST2D/train_and_eval_autokeras.py):  training and evaluation script of Autokeras
  * [`train_and_eval_autosklearn.py`](./MedMNIST2D/train_and_eval_autosklearn.py): training and evaluation script of auto-sklearn
  * [`eval_google_automl_vision.py`](./MedMNIST2D/eval_google_automl_vision.py): evaluation script of models trained by Google AutoML Vision

* [`MedMNIST3D/`](./MedMNIST3D/): training and evaluation scripts of MedMNIST3D
  * [`models.py`](./MedMNIST3D/models.py): *ResNet-18* and *ResNet-50* models (for small-image datasets like CIFAR-10/100), basically same as [`MedMNIST2D/models.py`](./MedMNIST2D/models.py)
  * [`train_and_eval_pytorch.py`](./MedMNIST3D/train_and_eval_pytorch.py): training and evaluation script implemented with PyTorch
  * [`train_and_eval_autokeras.py`](./MedMNIST3D/train_and_eval_autokeras.py):  training and evaluation script of Autokeras
  * [`train_and_eval_autosklearn.py`](./MedMNIST3D/train_and_eval_autosklearn.py): training and evaluation script of auto-sklearn

    
# Installation and Requirements
This repository is working with [MedMNIST official code](https://github.com/MedMNIST/MedMNIST/) and PyTorch.

1. Setup the required environments and install `medmnist` as a standard Python package:

        pip install medmnist

2. Check whether you have installed the latest [version](https://github.com/MedMNIST/MedMNIST/tree/main/medmnist/info.py):

        >>> import medmnist
        >>> print(medmnist.__version__)

3. The code requires common Python environments for machine learning; Basically, it was tested with

- Python 3 (Anaconda 3.6.3 specifically)
- PyTorch==1.3.1
- autokeras\==1.0.15
- auto-sklearn\==0.10.0
- tensorflow==2.3.0

  Higher (or lower) versions should also work (perhaps with minor modifications).

4. For *MedMNIST3D*, our code additionally requires [ACSConv](https://github.com/M3DV/ACSConv). Install it through `pip` via the command bellow:

        pip install git+git://github.com/M3DV/ACSConv.git
    
    Then you can check use the cool `2.5D`, `3D` and `ACS` model convertor as follows:

    ```python
    from torchvision.models import resnet18
    from acsconv.converters import ACSConverter
    # model_2d is a standard pytorch 2D model
    model_2d = resnet18(pretrained=True)
    B, C_in, H, W = (1, 3, 64, 64)
    input_2d = torch.rand(B, C_in, H, W)
    output_2d = model_2d(input_2d)

    model_3d = ACSConverter(model_2d)
    # once converted, model_3d is using ACSConv and capable of processing 3D volumes.
    B, C_in, D, H, W = (1, 3, 64, 64, 64)
    input_3d = torch.rand(B, C_in, D, H, W)
    output_3d = model_3d(input_3d)
    ```

5. Download the model weights and predictions from [Zenodo](https://doi.org/10.5281/zenodo.7782113).
    -  `weights_*.zip`: 
        - PyTorch, AutoKeras and Google AutoML Vision are provided for MedMNIST2D.
        - PyTorch and AutoKeras are provided for MedMNIST3D. 
        - If you are using PyTorch model weights, please note that the ResNet18_224 / ResNet50_224 models are trained with images resized to 224 x 224 by `PIL.Image.NEAREST`. 
        - Snapshots for `auto-sklearn` are not uploaded due to the embarrassingly large model sizes (lots of model ensemble).
    -  `predictions.zip`: We also provide all standard prediction files by PyTorch, auto-sklearn, AutoKeras and Google AutoML Vision, which works with `medmnist.Evaluator`. Each file is named as `{flag}_{split}_[AUC]{auc:.3f}_[ACC]{acc:.3f}@{run}.csv`, e.g., `bloodmnist_test_[AUC]0.997_[ACC]0.957@autokeras_3.csv`.
