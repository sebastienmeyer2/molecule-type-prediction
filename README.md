# Molecule Type Prediction

The goal of this project is to study the presence or absence of a certain property in molecules. More specifically, we have several thousands of molecules in the form of graphs and we would like to predict a binary label attesting of whether they have a property of interest. We followed a supervised approach and we implemented both kernel algorithms and kernel for graphs by hand.

A more detailed explanation of our results can be found in our report under *report/report.pdf*

Please refer to the following sections for more information about the package usage:

1. [Some results](#some-results)
2. [Installation](#installation-instructions)
3. [Description](#package-description)
4. [Usage via command lines](#package-usage)
5. [Documentation](#documentation)

## Some Results

In the following table, we present the model which allowed us to reach the best ROC AUC score of 0.89480 on the test set. It was trained using Kernel logistic regression with a regularization parameter of $\lambda = 1.73 \times 10^{-5}$.

| Kernel | Label enrichment | Parameters |
| --- | --- | --- |
| Counting | $\emptyset$ | $\sigma = 4.96$ |
| Edge histogram | $\emptyset$ | $\sigma = 4.96$ |
| Node histogram | $\emptyset$, 1 $\times$ Morgan and 1 $\times$ Weisfeiler-Lehman | $\sigma = 4.96$, $\sigma_{M} = 0.04$ and $\sigma_{WL} = 1.18$ |
| Geometric walk | $\emptyset$, 1 $\times$ Morgan and 1 $\times$ Weisfeiler-Lehman | $p = 4$; $\beta = 2.34$, $\beta_{M} = 1.89$ and $\beta_{WL} = 0.23$ |
| Shortest path | $\emptyset$, 1 $\times$ Morgan and 1 $\times$ Weisfeiler-Lehman | $\emptyset$ |

## Installation instructions

In order to use our package and run your own experiments, we advise you to set up a virtual environment.

You will need Python 3 and the *virtualenv* package:

    pip3 install virtualenv

Then, create your virtual environment and switch to it:

    python3 -m venv venv

    source venv/bin/activate (Linux)
    .\venv\Scripts\Activate.ps1 (Windows PowerShell)

Finally, install all the requirements:

    pip3 install -r requirements.txt (Linux)
    pip3 install -r .\requirements.txt (Windows PowerShell)

*Note*: Tested on Linux with Python 3.10.9 and on Windows.

## Package description

Below, we give a brief tree view of our package.

    .
    ├── doc  # contains a generated documentation of src/ in html
    ├── report  # contains our complete report in pdf format
    ├── src  # source code
    |   ├── engine
    |   |   ├── __init__.py
    |   |   ├── gridsearch.py
    |   |   ├── hub.py
    |   |   └── objective.py
    |   ├── kernels
    |   |   ├── __init__.py
    |   |   ├── base.py  # abstract class
    |   |   ├── count.py
    |   |   ├── edge_histogram.py
    |   |   ├── geometric_walk.py
    |   |   ├── node_histogram.py
    |   |   ├── order_walk.py
    |   |   ├── shortest_path.py
    |   |   └── sum.py
    |   ├── models
    |   |   ├── __init__.py
    |   |   ├── base.py  # abstract class
    |   |   ├── logistic_regression.py
    |   |   ├── ridge_regression.py
    |   |   └── svc.py
    |   ├── utils
    |   |   ├── __init__.py
    |   |   ├── data.py
    |   |   ├── functions.py
    |   |   ├── graph_ops.py
    |   |   ├── kernel_ops.py
    |   |   └── misc.py
    |   ├── __init__.py
    |   └── start.py
    ├── README.md
    ├── data_analysis.ipynb  # simple analysis of the dataset
    └── requirements.txt  # contains the required Python packages to run our files

## Package usage

The main file to use for experimenting is *src/start.py*. The command is as follows:

    python3 src/start.py [options]

- `--gridsearch-subset`: Select a subset of the training set to fasten cross validation. Default: 0.2.
- `--kernel-name`: Name of the kernel following package usage. Default: "sum".
- `--model-name`: Name of the model following package usage. Default: "logreg".
- `--normalize`: If "True", kernels matrices are normalized to zero mean and unit variance. Default: "True".
- `--center`: If "True", kernels matrices are centered. Default: "False".
- `--trials`: Choose the number of gridsearch trials. Default: 0.
- `--folds`: Number of cross-validation splits. Default: 5.
- `--eval-metric`: Evaluation metric for gridsearch. Default: "roc_auc".
- `--verbose`: Verbosity. Default: "False".
- `--submission`: If "True", a submission file will be produced after hyperparameter optimization. Default: "True".
- `--results-dir`: Directory where submission files are stored. Default: "results".

*Example*: In order to reproduce our results, simply run:

```bash
python3 src/start.py
```

## Documentation

A complete documentation is available in the *doc/src/* folder. If it is not
generated, you can run from the root folder:

```bash
pip3 install pdoc3
python3 -m pdoc -o doc/ --html --config latex_math=True --force src/
```

Then, open *doc/src/index.html* in your browser and follow the guide!
