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

### Installation

pydraml requires Python 3.7+.

```
pip install pydra-ml
```

## CLI usage

This repo installs `pydraml` a CLI to allow usage without any programming.

To test the CLI for a classification example, copy the `pydra_ml/tests/data/breast_cancer.csv` and 
`short-spec.json.sample` to a folder and run.

```
$ pydraml -s short-spec.json.sample
```
To check a regression example, copy `pydra_ml/tests/data/diabetes_table.csv` and `diabetes_spec.json`
to a folder and run.

```
$ pydraml -s diabetes_spec.json
```

For each case pydra-ml will generate a result folder with the spec file name that includes
`test-{metric}-{timestamp}.png` file for each metric together with a pickled results file 
containing all the scores from the model evaluations.

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
- *gen_shap*: Boolean indicating whether shap values are generated
- *nsamples*: Number of samples to use for shap estimation
- *l1_reg*: Type of regularizer to use for shap estimation
- *plot_top_n_shap*: Number or proportion of top SHAP values to plot (e.g., 16 or 0.1 for top 10%). Set to 1.0 (float) to plot all features or 1 (int) to plot top first feature.
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
 "gen_shap": true,
 "nsamples": 100,
 "l1_reg": "aic",
 "plot_top_n_shap": 16,
 "metrics": ["roc_auc_score"]
 }
```

## Output:
The workflow will output:
- `results-{timestamp}.pkl` containing 1 list per model used. For example, if assigned to variable `results`, it is accessed through `results[0]` to `results[N]`
(if `permute: [false,true]` then it will output the model trained on the labels first `results[0]` and the model trained on permuted labels second `results[1]`.
Each model contains:
    - `dict` accesed through `results[0][0]` with model information: `{'ml_wf.clf_info': ['sklearn.neural_network', 'MLPClassifier', {'alpha': 1, 'max_iter': 1000}], 'ml_wf.permute': False}`
    - `pydra Result obj` accesed through `results[0][1]` with attribute `output` which itself has attributes:
        - `feature_names`: from the columns of the data csv.
        And the following attributes organized in N lists for N bootstrapping samples:
        - `output`: N lists, each one with two lists for true and predicted labels.
        - `score`: N lists each one containing M different metric scores.
        - `shaps`: N lists each one with a list of shape (P,F) where P is the amount of predictions and F the different SHAP values for each feature. `shaps` is empty if `gen_shap` is set to `false` or if `permute` is set to true.
- One figure per metric with performance distribution across splits (with or without null distribution trained on permuted labels)
- `shap-{timestamp}` dir
    - SHAP values are computed for each prediction in each split's test set
    (e.g., 30 bootstrapping splits with 100 prediction will create (30,100) array). The mean is taken across predictions for each split (e.g., resulting in a (64,30) array for 64 features and 30 bootstrapping samples).
    - For binary classification, a more accurate display of feature importance obtained by splitting predictions into TP, TN, FP, and FN,
    which in turn can allow for error auditing (i.e., what a model pays attention to when making incorrect/false predictions)
        - `quadrant_indexes.pkl`: The TP, TN, FP, FN indexes are saved in  as a `dict` with one `key` per model (permuted models without SHAP values will be skipped automatically), and each key `values` being a bootstrapping split.
        - `summary_values_shap_{model_name}_{prediction_type}.csv` contains all SHAP values and summary statistics ranked by the mean SHAP value across bootstrapping splits. A sample_n column can be empty or NaN if this split did not have the type of prediction in the filename (e.g., you may not have FNs or FPs in a given split with high performance).
        - `summary_shap_{model_name}_{plot_top_n_shap}.png` contains SHAP value summary statistics for all features (set to 1.0) or only the top N most important features for better visualization.


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
