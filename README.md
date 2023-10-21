[![Python package](https://github.com/nipype/pydra-ml/workflows/Python%20package/badge.svg?branch=master)](https://github.com/nipype/pydra-ml/actions?query=workflow%3A%22Python+package%22)

# pydra-ml

Pydra-ML is a demo application that leverages [Pydra](https://github.com/nipype/pydra)
together with [scikit-learn](https://scikit-learn.org) to perform model comparison
across a set of classifiers. The intent is to use this as an application to make
Pydra more robust while allowing users to generate classification reports more
easily. This application leverages Pydra's powerful splitters and combiners to
scale across a set of classifiers and metrics. It will also use Pydra's caching
to:

1. Efficiently train models using nested bootstrapping (with k-fold cross-validation performed in inner loop for hyperparameter tuning)

2. **Compare *some* scikit-learn pipelines** in addition to base
  classifiers (i.e., showing the distribution of performance of different models side-by-side).

  ![alt text](https://github.com/danielmlow/pydra-ml/blob/master/examples/test-roc_auc_score-example.png?raw=true)
  The distribution of performance from models trained on true labels (blue) and trained on permuted labels (orange) over 50 bootstrapping splits.


  ![alt text](https://github.com/danielmlow/pydra-ml/blob/master/examples/test_performance_with_null_roc_auc_score.png?raw=true)
  Median performance across 50 bootstrapping splits (95% Confidence Interval; median performance of null model)


3. Save models and **not redo model training and evaluation** when new metrics are added, or when
number of iterations (`n_splits`) is increased. Just change spec file and it will use stored models to save time.

4. Output report three types of **feature importance** methods:
- (1) standard feature importance methods for some models form sklearn (e.g., `coef_` for linear models, `feature_importances_` for tree-based models), *NOT FULLY TESTED*
- (2) sklearn's [permutation_importance](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html) (model agnostic, available for all models), *NOT FULLY TESTED*
- (3) [Kernel SHAP](https://github.com/slundberg/shap) feature importance (model agnostic, available for all models)

    ![alt text](https://github.com/danielmlow/pydra-ml/blob/master/examples/shap_example.png?raw=true)

    Each bootstrapping split of the data may create its own model (e.g., different weights or best hyperparameters). For each split, we take the average of the absolute SHAP values across all test predictions. We then compute the average SHAP values across all splits.





### Installation

pydraml requires Python 3.7+.

```
pip install pydra-ml
```

## CLI usage

This repo installs `pydraml` a CLI to allow usage without any programming.

To test the CLI for a classification example, copy the `pydra_ml/tests/data/breast_cancer.csv` and
`examples/classification_cancer_short-spec.json` to a folder and run or run within in `examples/` folder.

```
$ pydraml -s classification_cancer_short-spec.json
```

For now, gen_feature_importance and gen_permutation_importance only are working with linear models. We need to test on other models, pipelines including ones with hyperparameter tuning:
```
$ pydraml -s classification_cancer_toy-spec.json
```


To check a regression example, copy the `pydra_ml/tests/data/diabetes_table.csv` and
`examples/regression_diabetes_spec.json` to a folder and run or run within in `examples/` folder.

```
$ pydraml -s regression_diabetes_spec.json
```

For each case pydra-ml will generate a result folder `out-{spec_file_name}-{timestamp}/` that contains figures and tables comparing each model and their important features together with a
pickled results file containing all the scores from the model evaluations (see **Output** section below)

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
$ pydraml -s ../classification_cancer_short-spec.json -p cf "n_procs=8"
```

or execution via dask.

```
$ pydraml -s ../classification_cancer_short-spec.json -p dask "address=tcp://192.168.1.154:8786"
```

## Current specification

The current specification is a JSON file as shown in the example below. It needs
to contain all the fields described here. For datasets with many features, you
will want to generate `x_indices` programmatically.

- *filename*: Absolute path to the CSV file containing data. Can contain a column,
  named `group` to support `GroupShuffleSplit`, else each sample is treated as a
  group.
- *x_indices*: Numeric (0-based) or string list of column names to use as input features. Careful with not including output or target variables here.
- *target_vars*: String list of target variable (at present only one is supported)
- *group_var*: String to indicate column to use for grouping
- *n_splits*: Number of shuffle split iterations to use
- *test_size*: Fraction of data to use for test set in each iteration
- *clf_info*: List of scikit-learn classifiers to use.
- *permute*: List of booleans to indicate whether to generate a null model with permuted labels aka permutation test (set to true) or not (set to false)
- *gen_feature_importance*: Boolean indicating whether unique feature importance method should be generated for each model if available (e.g., `coef_` for linear models, `feature_importances_` for tree-based models) *NOT FULLY TESTED: set to false*
- *gen_permutation_importance*: Boolean indicating whether permutation_importance values are generated (model agnostic, available for all models) *NOT FULLY TESTED: set to false*
- *gen_shap*: Boolean indicating whether shap values are generated (model agnostic, available for all models)
- *nsamples*: Number of samples to use for shap estimation, use integer or the "auto" setting uses `nsamples = 2 * X.shape[1] + 2048`.
- *l1_reg*: Type of regularizer to use for shap estimation
- *plot_top_n_shap*: Number or proportion of top shap values to plot (e.g., 16
or 0.1 for top 10%). Set to 1.0 (float) to plot all features or 1 (int) to plot
top first feature.
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

This can also be embedded as a list indicating a scikit-learn Pipeline. For
example:

```
 [ ["sklearn.impute", "SimpleImputer"],
   ["sklearn.preprocessing", "StandardScaler"],
   ["sklearn.tree", "DecisionTreeClassifier", {"max_depth": 5}]
  ]
```

## Example specification:

```
{"filename": "breast_cancer.csv",
 "x_indices": ["radius_mean", "texture_mean","perimeter_mean", "area_mean", "smoothness_mean",
       "compactness_mean", "concavity_mean", "concave points_mean",
       "symmetry_mean", "fractal_dimension_mean", "radius_se",
       "texture_se", "perimeter_se", "area_se", "smoothness_se",
       "compactness_se", "concavity_se", "concave points_se",
       "symmetry_se", "fractal_dimension_se", "radius_worst",
       "texture_worst", "perimeter_worst", "area_worst",
       "smoothness_worst", "compactness_worst", "concavity_worst",
       "concave points_worst", "symmetry_worst", "fractal_dimension_worst"],
 "target_vars": ["target"],
 "group_var": null,
 "n_splits": 100,
 "test_size": 0.2,
 "clf_info": [
 ["sklearn.ensemble", "AdaBoostClassifier"],
 ["sklearn.naive_bayes", "GaussianNB"],
 [ ["sklearn.impute", "SimpleImputer"],
   ["sklearn.preprocessing", "StandardScaler"],
   ["sklearn.tree", "DecisionTreeClassifier", {"max_depth": 5}]],
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
 "gen_feature_importance": false,
 "gen_permutation_importance": false,
 "permutation_importance_n_repeats": 5,
 "permutation_importance_scoring": "accuracy",
 "gen_shap": true,
 "nsamples": "auto",
 "l1_reg": "aic",
 "plot_top_n_shap": 16,
 "metrics": ["roc_auc_score", "f1_score", "precision_score", "recall_score"]
 }
```

## Output:
The workflow will output:
<<<<<<< HEAD
- `results-{timestamp}.pkl` containing 1 list per model used. For example, if the `pkl` file is
assigned to variable `results`, the models are accessed through `results[0]` to `results[N]`.
 If `permute: [false,true]` then it will output the model trained on the labels first (`results[0]`) and the model trained on the permuted labels second (`results[1]`). If there is an additional model, these will be accessed through `results[2]` (labels) and `results[3]` (permuted).

  Each model contains:
    - `dict` accessed through `results[0][0]` with model information:
        ```python
        import pickle as pk

        with open("results-20201208T010313.229190.pkl", "rb") as fp:
            results = pk.load(fp)

        print(results[0][0]) #1st model trained on labels
        ```

        `{'ml_wf.clf_info': ['sklearn.neural_network', 'MLPClassifier', {'alpha': 1, 'max_iter': 1000}], 'ml_wf.permute': False}`

        ```python
        print(results[3][0]) #2nd models trained on permuted labels
        ```

        `{'ml_wf.clf_info':['sklearn.linear_model', 'LogisticRegression', {'penalty': 'l2'}], 'ml_wf.permute': True}`

    - `pydra Result obj` accessed through `results[0][1].output`:
=======
- `results-{timestamp}.pkl` containing 1 list per model used. For example, if
assigned to variable `results`, it is accessed through `results[0]` to `results[N]`
(e.g., if `permute: [true,false]` then it will output the model trained on permuted labels first `results[0]` and the model trained on the labels
second `results[1]`. If there is an additional model, these will be accessed through `results[2]` and `results[3]`).
Each model contains:
    - `dict` accessed through `results[0][0]` with model information:
     `{'ml_wf.clf_info': ['sklearn.neural_network', 'MLPClassifier',
         {'alpha': 1, 'max_iter': 1000}], 'ml_wf.permute': False}`
    - `pydra Result obj` accessed through `results[0][1]` with attribute `output`
>>>>>>> ea2092bb5f199aa6ff83f25f863d3652f824f6af
      which itself has attributes:
        - `feature_names`: from the columns of the data csv.

          ```python
          print(results[1][1].output.feature_names)
          ```

          `['mean radius', 'mean texture', 'mean perimeter', 'mean area', ... ]`

          And the following attributes organized in *n_splits* lists for *n_splits* bootstrapping samples:
        - `output`: *n_splits* lists, each one with two lists for true and predicted labels.
        - `score`: *n_splits* lists each one containing M different metric scores.

          Three types of feature importance methods:

        - (1) `feature_importance`: standard feature importance method from *sklearn*. Limitation: not all models have standard methods and difficult to compare methods across models.
          - `pipeline.coef_` for linear models (coefficients of regression, SVC).
          - `pipeline.coefs_` for multi-layer perceptron, which returns `j` lists for `j` hidden nodes connections with each input
          - `pipeline.feature_importances_` for decision tree, Random Forest, or boosting algorithms

          ```python
          print(results[1][1].output.feature_importance)
          ```
        - (2) `permutation_importance`: the difference in performance from permutating the feature column as in [sklearn's permutation importance](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html).
        Advantage: works for all models (i.e., model agnostic). Limitation: measures decrease in performance, not magnitude of each feature.

            ```python
              print(results[1][1].output.permutation_importance)
            ```

        - (3) `shaps`: `n_splits` lists each one with a list of shape (P,F) where P is the
        amount of predictions and F the different SHAP values for each feature.
        `shaps` is empty if `gen_shap` is set to `false` or if `permute` is set
        to true. Advantage: model agnostic, produces magnitude for each feature.

            ```python
              print(results[1][1].output.shaps)
            ```

        - `model`: A pickled version of the model trained on all the input data.
        One can use this model to test on new data that has the exact same input
        shape and features as the trained model. For example:

          ```python
          import pickle as pk
          import numpy as np

          with open("results-20201208T010313.229190.pkl", "rb") as fp:
              results = pk.load(fp)

          trained_model = results[0][1].output.model
          trained_model.predict(np.random.rand(1, 30))
          ```

          Please make sure the value of `results[N][0].get('ml_wf.permute')` is `False` to ensure that you are not using
          a permuted model.


- One figure per metric with performance distribution across splits (with or
without null distribution trained on permuted labels)
- `performance_table-{timestamp}` folder:
  - `test-performance-table_{metric}_all-splits_{timestamp).csv` with the test performance of each the model/s trained on each bootstrapping split and median score
  - `test-performance-table_accuracy_score_with-95ci-and-median-null_20210702T223005.935447``test-performance-table_{metric}_all-splits_{timestamp).csv` with the median score and 95% confidence interval (CI) and median score of null model if available: `median score [95% CI; median null score]`
- `stats-{metric}-{timestamp}.png`: one figure per any metric with the word `score` in it, containing a one tailed statistical comparison(row > column) of models using an empirical p-value, a common and effective measure for evaluating classifier performance (see Definition 1 in Ojala & Garriga, 2010) as implemented in [sklearn](https://github.com/scikit-learn/scikit-learn/blob/2beed5584/sklearn/model_selection/_validation.py#L1062). Annotation = p-value, color = significant over alpha level of 0.05.  The p-value represents the fraction of column-model scores where the row-model classifier had a higher mean performance (e.g., a p-value of 0.02 indicates that the mean score of a row model is higher than 98% of column-model scores). Data model vs. null model is displayed on the diagonal. The actual numeric values are stored in a correspondingly named pkl file.
- `shap-{timestamp}` dir
    - SHAP values are computed for each prediction in each split's test set
    (e.g., 30 bootstrapping splits with 100 prediction will create (30,100) array).
     The mean is taken across predictions for each split (e.g., resulting in a
     (64,30) array for 64 features and 30 bootstrapping samples).
    - For binary classification, a more accurate display of feature importance
    obtained by splitting predictions into TP, TN, FP, and FN, which in turn can
    allow for error auditing (i.e., what a model pays attention to when making
    incorrect/false predictions)
        - `quadrant_indexes.pkl`: The TP, TN, FP, FN indexes are saved in  as a
        `dict` with one `key` per model (permuted models without SHAP values will
        be skipped automatically), and each key `values` being a bootstrapping split.
        - `summary_values_shap_{model_name}_{prediction_type}.csv` contains all
        SHAP values and summary statistics ranked by the mean SHAP value across
        bootstrapping splits. A sample_n column can be empty or NaN if this split
        did not have the type of prediction in the filename (e.g., you may not
        have FNs or FPs in a given split with high performance).
        - `summary_shap_{model_name}_{plot_top_n_shap}.png` contains SHAP value
        summary statistics for all features (set to 1.0) or only the top N most
        important features for better visualization.

## Debugging

You will need to understand a bit of pydra to know how to debug this application for
now. If the process crashes, the easiest way to restart is to remove the `cache-wf`
folder first. However, if you are rerunning, you could also remove any `.lock` file in the `cache-wf` directory.


## Developer installation

Install repo in developer mode:

```
git clone https://github.com/nipype/pydra-ml.git
cd pydra-ml
pip install -e .[dev]
```

It is also useful to install pre-commit, which takes care of styling when
committing code. When pre-commit is used you may have to run git commit twice,
since pre-commit may make additional changes to your code for styling and will
not commit these changes by default:

```
pip install pre-commit
pre-commit install
```

### Project structure

- `tasks.py` contain the Python functions.
- `classifier.py` contains the Pydra workflow and the annotated tasks.
- `report.py` contains report generation code.
