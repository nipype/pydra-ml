# pydra-ml

for developers:

after cloning the repo do:

1. pip install -e .[dev]
2. pre-commit install

# CLI usage

to test the CLI copy the `pydra_ml/tests/data/breast_cancer.csv` and
`short-spec.json.sample` to a folder and run.

```
$ pydraml -s short-spec.json.sample
```

# Current specification

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
 "l1_reg": "aic"
 }
```
