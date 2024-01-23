# Uncertainty Quantification for Failure Prediction in Autonomous Driving Systems

Codebase related to the master's thesis "Uncertainty Quantification for Failure Prediction in Autonomous Driving Systems" by Ruben Grewal. 

The repository also contains the artifacts accompanying the paper "*Predicting Safety Misbehaviours in Autonomous Driving Systems using Uncertainty Quantification*" by R. Grewal, P. Tonella and A. Stocco, published in the proceedings of the 17th IEEE International Conference on Software Testing, Verification and Validation (ICST 2024).  The codebase is based on previous work that is available [here](https://tsigalko18.github.io/assets/pdf/2022-Stocco-ASE.pdf).

A preprint of the paper can be found on [here](https://tsigalko18.github.io/assets/pdf/2024-Grewal-ICST.pdf).

## Reference

If you use our work in your research, or it helps it, or if you simply like it, please cite it in your publications. 
Here is an example BibTeX entry:

```
@inproceedings{2024-Grewal-ICST, 
  author = {Ruben Grewal and Paolo Tonella and Andrea Stocco},
  title = {Predicting Safety Misbehaviours in Autonomous Driving Systems using Uncertainty Quantification},
  booktitle = {Proceedings of 17th IEEE International Conference on Software Testing, Verification and Validation},
  series = {ICST '24},
  publisher = {IEEE},
  abbr = {ICST},
  year = {2024}
}
```

## Dependencies

**Software setup:** We adopted Visual Studio Code.

First, you need [anaconda](https://www.continuum.io/downloads) or [miniconda](https://conda.io/miniconda.html) installed on your machine. Then, you can create and install all dependencies on a dedicated virtual environment, by running one of the following commands, depending on your platform.

```python
# Windows
conda create --name <env> --file requirements.txt
```

Alternatively, you can manually install the required libraries (see the contents of the requirements.txt files) using ```pip```.

**Hardware setup:** Training the DNN models (self-driving cars and autoencoders) on our datasets is computationally expensive. Therefore, we recommend using a machine with a GPU. In our setting, we ran our experiments on a machine equipped with a AMD Ryzen 5 processor, 32 GB of RAM, and an NVIDIA GPU GeForce RTX 3070 with 8 GB of dedicated video memory.


## Training models

For MCD Models:
* changes in the ``the config_my.py`` file:
    * ``USE_MC = True``
    * ``USE_DE = False``
* run the file ``self_driving_car_train.py``. This will train mcd models with dropout rates: 5%, 10%, 15%, 20%, 25%, 30% and 35%.

For DE models:
* changes in the ``config_my.py`` file:
    * ``USE_MC = False``
    * ``USE_DE = True``  
* run the file ``self_driving_car_train.py``. This will train 120 different dave2 models from which the different ensembles are built. To change the numbers of models trained modify NUM_ENSEMBLE_MODELS

## Calculate offline uncertainty
Run the ``offline_uncertainty_calculation_all.py`` file. It calculates the uncertainty scores for deep ensembles and MC dropout models. It uses the simulations files (recorded laps in the Udacity Simulator under the different conditions). The plots and csv files will be saved in the uncertainty folderThe simulation files should be saved in the simulations/ directory. 

## Evaluate results:
Set the confidence threshold level in the ``config_my.py`` file through ``CONFIDENCE_LEVEL = 0.XXX`` (set it to the desired confidence level [0.95,0.99,0.9990,0.9999,0.99999]).

* For UQ: run ``evaluate_uq.py``
* For SelfOracle: run ``evaluate_failure_prediction_selforacle_compute_all.py``
* For ThirdEye: run ``evaluate_failure_prediction_heatmaps_scores_compute_all.py``

All results are now saved in the *results* folder

## Datasets & Simulator

Driving datasets, self-driving car models, and simulator have a combined size of several GBs. We will share them on demand.
