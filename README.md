![Python package](https://github.com/nipype/pydra-ml/workflows/Python%20package/badge.svg?branch=master)

# pydra-ml

Pydra-ML is a demo application that leverages [Pydra](https://github.com/nipype/pydra)
together with [scikit-learn](https://scikit-learn.org) to perform model comparison
across a set of classifiers. The intent is to use this as an application to make
Pydra more robust while allowing users to generate classification reports more
easily. This application leverages Pydra's powerful splitters and combiners to
scale across a set of classifiers and metrics. It will also use Pydra's caching
to not redo model training and evaluation when new metrics are added, or when
number of iterations (`n_splits`) is increased.

Upcoming features:
1. Improve output report containing [SHAP](https://github.com/slundberg/shap)
  feature analysis.
2. Allow for comparing scikit-learn pipelines.
3. Test on scikit-learn compatible classifiers

## CLI usage

This repo installs `pydraml` a CLI to allow usage without any programming.

To test the CLI copy the `pydra_ml/tests/data/breast_cancer.csv` and
`short-spec.json.sample` to a folder and run.

```
$ pydraml -s short-spec.json.sample
```

This will generate a `test-{metric}-{timestamp}.png` file for each metric in the
local folder together with a pickled results file containing all the scores from
the model evaluations.

```
$ pydraml --help
Usage: pydraml [OPTIONS]

Options:
  -s, --specfile PATH   Specification file to use  [required]
  -p, --plugin TEXT...  Pydra plugin to use  [default: cf, n_procs=1]
  -c, --cache TEXT      Cache dir  [default:
                        /Users/satra/software/sensein/pydra-ml/cache-wf]

  --help                Show this message and exit.
```

With the plugin option you can use local multiprocessing

```
$ pydraml -s ../short-spec.json.sample -p cf "n_procs=8"
```

or execution via dask.

```
$ pydraml -s ../short-spec.json.sample -p dask "address=tcp://192.168.1.154:8786"
```

## Current specification

The current specification is a JSON file as shown in the example below. It needs
to contain all the fields described here. For datasets with many features, you
will want to generate `x_indices` programmatically.

- *filename*: Absolute path to the CSV file containing data. Can contain a column,
  named `group` to support `GroupShuffleSplit`, else each sample is treated as a
  group.
- *x_indices*: Numeric (0-based) or string list of columns to use as input features
- *target_vars*: String list of target variable (at present only one is supported)
- *n_splits*: Number of shuffle split iterations to use
- *test_size*: Fraction of data to use for test set in each iteration
- *clf_info*: List of scikit-learn classifiers to use.
- *permute*: List of booleans to indicate whether to generate a null model or not
- *noshap*: Boolean indicating whether shap values are evaluated
- *nsamples*: Number of samples to use for shap estimation
- *l1_reg*: Type of regularizer to use for shap estimation
- *metrics*: scikit-learn metric to use

## `clf_info` specification

This is a list of classifiers from scikit learn and uses an array to encode:

```
- module
- classifier
- (optional) classifier parameters
- (optional) gridsearch param grid
```

when param grid is provided and default classifier parameters are not changed,
then an empty dictionary **MUST** be provided as parameter 3.

## Example specification:

```
{"filename": "breast_cancer.csv",
 "x_indices": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
 "target_vars": ["target"],
 "n_splits": 100,
 "test_size": 0.2,
 "clf_info": [
 ["sklearn.ensemble", "AdaBoostClassifier"],
 ["sklearn.naive_bayes", "GaussianNB"],
 ["sklearn.tree", "DecisionTreeClassifier", {"max_depth": 5}],
 ["sklearn.ensemble", "RandomForestClassifier", {"n_estimators": 100}],
 ["sklearn.ensemble", "ExtraTreesClassifier", {"n_estimators": 100, "class_weight": "balanced"}],
 ["sklearn.linear_model", "LogisticRegressionCV", {"solver": "liblinear", "penalty": "l1"}],
 ["sklearn.neural_network", "MLPClassifier", {"alpha": 1, "max_iter": 1000}],
 ["sklearn.svm", "SVC", {"probability": true},
  [{"kernel": ["rbf", "linear"], "C": [1, 10, 100, 1000]}]],
 ["sklearn.neighbors", "KNeighborsClassifier", {},
  [{"n_neighbors": [3, 5, 7, 9, 11, 13, 15, 17, 19],
    "weights": ["uniform", "distance"]}]]
 ],
 "permute": [true, false],
 "noshap": false,
 "nsamples": 100,
 "l1_reg": "aic",
 "metrics": ["roc_auc_score"]
 }
```

### Installation

pydraml requires Python 3.7+.

```
git clone https://github.com/nipype/pydra-ml.git
cd pydra-ml
pip install -e .
```

## Developer installation

Install repo in developer mode:

```
git clone https://github.com/nipype/pydra-ml.git
cd pydra-ml
pip install -e .[dev]
```

It is also useful to install pre-commit:

```
pip install pre-commit
pre-commit
```

### Project structure

- `tasks.py` contain the annotated Pydra tasks.
- `classifier.py` contains the Pydra workflow.
- `report.py` contains report generation code.
