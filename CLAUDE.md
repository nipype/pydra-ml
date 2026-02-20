# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Install for development (using uv)
```bash
uv venv .venv
uv pip install --pre -e ".[dev]"
```

### Run tests
```bash
uv run pytest pydra_ml/tests/test_classifier.py -s
# Single test
uv run pytest pydra_ml/tests/test_classifier.py::test_classifier -s
uv run pytest pydra_ml/tests/test_classifier.py::test_regressor -s
```
The `-s` flag is required because pytest's default stdout capture can interfere with matplotlib/seaborn output.

### Linting and formatting
```bash
pre-commit run --all-files
```

### Run the CLI
```bash
pydraml -s specification.json -p cf "n_procs=1"
pydraml -s specification.json -p dask "address=tcp://host:8786"
```

## Architecture

**pydra-ml** compares multiple scikit-learn classifiers/regressors across bootstrap splits using Pydra as the dataflow execution engine. The pipeline builds a DAG, executes it with caching, then generates reports.

### Module responsibilities

- **`tasks.py`** — Pure Python functions wrapped as Pydra tasks: `read_file`, `gen_splits`, `train_test_kernel`, `calc_metric`, `get_feature_importance`, `get_permutation_importance`, `get_shap`, `create_model`.

- **`classifier.py`** — Builds and runs the Pydra workflow DAG (`gen_workflow`, `run_workflow`). The workflow splits on `clf_info` (one per classifier) and `permute` (true/false for null models), runs tasks in parallel, then combines results.

- **`report.py`** — Post-processing: generates violin plots, pairwise statistical comparison heatmaps (empirical p-values), performance tables (median + 95% CI), and SHAP summary plots/CSVs. Classification uses TP/TN/FP/FN quadrant analysis; regression uses quartile analysis.

- **`cli.py`** — Click-based CLI. Parses the JSON spec file, validates it, then calls `gen_workflow` / `run_workflow`.

### Data flow

1. **Input**: JSON spec file specifying CSV path, feature indices, target variable, classifiers, metrics, and options.
2. **Workflow**: `read_file → gen_splits → fit_clf → calc_metric` with optional `get_feature_importance / get_permutation_importance / get_shap / create_model` branches.
3. **Output**: Written to `out-{specfile}-{timestamp}/` — pickled results, PNG violin/heatmap plots, performance CSV tables, and SHAP CSVs/plots.

### Pydra 1.0 patterns used

- Tasks are defined with `@python.define` from `pydra.compose.python` with typed inputs/outputs.
- The workflow is defined with `@workflow.define` from `pydra.compose.workflow`.
- `.split("field", field=values)` creates parallel executions; outer split on `["clf_info", "permute"]` runs one workflow per combination, inner split on `split_index` parallelises bootstrap splits.
- `.combine("nodename.fieldname")` is a cross-node combiner: downstream tasks collect per-split results into a list.
- Splitting on a lazy output (`split_index=gensplit.split_indices`) works when the output type is `list[T]` or `ty.Any`.
- `workflow.add(task, name="node_name")` inside `@workflow.define` functions adds nodes and returns lazy output proxies; explicit names are required for cross-node combiners.
- `Submitter(worker="debug", cache_root=path)` runs synchronously for testing; `worker="cf"` for parallel production use.
- `gen_workflow()` returns a `WorkflowSpec` dataclass (not the pydra task directly) to bundle the cache dir and inputs metadata needed by `run_workflow`.

### Classifier specification format (JSON)

```json
{
  "clf_info": [
    ["sklearn.module", "ClassName"],
    ["sklearn.module", "ClassName", {"param": value}],
    [["sklearn.module1", "Step1"], ["sklearn.module2", "Step2"]]
  ],
  "permute": [true, false],
  "metrics": ["roc_auc_score", "accuracy_score"],
  "gen_shap": true,
  "gen_feature_importance": false,
  "gen_permutation_importance": false
}
```

Classifiers are dynamically instantiated via `__import__()`. A nested list means a scikit-learn `Pipeline`.

### Code style

- Formatting: **black** (line length from flake8 config: max 99 chars)
- Linting: **flake8** (excludes `__init__.py` and `tests/`)
- Imports: **isort** (Black profile)
- Logging: controlled via `PYDRAML_LOG_LEVEL` environment variable
