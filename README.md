# EE660 Final Project - Group 8

# Reproducing results

* Install the required packages before running the project

```bash
pip install -r requirements.txt
```

* Run the pre-run bash script to create requisite directories and download model files

```bash
bash prerun.sh
python main.py
```

* `main.py` file accepts other command line arguments to generate other results such as plots and tables given shown in the report
    * `training_mode`: Select from sl, ssl or both to generate results accordingly (Default: sl)
    * `plots`: Set to True to generate plots for the selected training mode (Default: True)
    * `tables`: Set to True to generate latex formatted tables for the selected training mode (Default: False)

**Note:** To generate results other than those of the final model, please clone this repository by running the following command

```bash
git clone https://github.com/aditya-anulekh/ee660-project.git
```

Example command to generate plots for supervised learning

```bash
python main.py --training_mode sl
```


* Training supervised learning models

```bash
python supervised_learning.py
```

* Training semi-supervised learning models

```bash
python semi_supervised_learning.py
```

# Members

* Aditya Anulekh Mantri: adityaan@usc.edu
* Surya Teja Chilumukuru Venkata Naga: suryatej@usc.edu
