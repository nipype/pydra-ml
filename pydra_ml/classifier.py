#!/usr/bin/env python

import dataclasses
import itertools
import os
import typing as ty

from pydra.compose import python, workflow
from pydra.engine import Submitter

from .report import gen_report
from .tasks import (
    calc_metric,
    create_model,
    gen_splits,
    get_feature_importance,
    get_permutation_importance,
    get_shap,
    read_file,
    train_test_kernel,
)

# --- Pydra task definitions ---


@python.define(
    outputs={"X": ty.Any, "Y": ty.Any, "groups": ty.Any, "feature_names": ty.Any}
)
def ReadFile(
    filename: ty.Any,
    x_indices: ty.Any = None,
    target_vars: ty.Any = None,
    group: ty.Any = None,
) -> tuple[ty.Any, ty.Any, ty.Any, ty.Any]:
    return read_file(filename, x_indices, target_vars, group)


@python.define(outputs={"splits": ty.Any, "split_indices": ty.Any})
def GenSplits(
    n_splits: int,
    test_size: float,
    X: ty.Any,
    Y: ty.Any,
    groups: ty.Any = None,
    bootstrap_strategy: str = "group_shuffle",
    n_bins: int = 10,
) -> tuple[ty.Any, ty.Any]:
    return gen_splits(
        n_splits,
        test_size,
        X,
        Y,
        groups,
        bootstrap_strategy=bootstrap_strategy,
        n_bins=n_bins,
    )


@python.define(outputs={"output": ty.Any, "model": ty.Any})
def TrainTestKernel(
    X: ty.Any,
    y: ty.Any,
    train_test_split: ty.Any,
    split_index: ty.Any,
    clf_info: ty.Any,
    permute: ty.Any,
    balancing: ty.Any = None,
    balancing_bins: ty.Any = None,
    groups: ty.Any = None,
) -> tuple[ty.Any, ty.Any]:
    return train_test_kernel(
        X,
        y,
        train_test_split,
        split_index,
        clf_info,
        permute,
        balancing,
        balancing_bins,
        groups,
    )


@python.define(outputs={"score": ty.Any, "output": ty.Any})
def CalcMetric(output: ty.Any, metrics: ty.Any) -> tuple[ty.Any, ty.Any]:
    return calc_metric(output, metrics)


@python.define(outputs={"feature_importance": ty.Any})
def GetFeatureImportance(
    permute: ty.Any,
    model: ty.Any,
    gen_feature_importance: bool = True,
) -> ty.Any:
    return get_feature_importance(
        permute=permute, model=model, gen_feature_importance=gen_feature_importance
    )


@python.define(outputs={"permutation_importance": ty.Any})
def GetPermutationImportance(
    X: ty.Any,
    y: ty.Any,
    permute: ty.Any,
    model: ty.Any,
    permutation_importance_n_repeats: int = 5,
    permutation_importance_scoring: ty.Any = None,
    gen_permutation_importance: bool = True,
) -> ty.Any:
    return get_permutation_importance(
        X,
        y,
        permute,
        model,
        permutation_importance_n_repeats,
        permutation_importance_scoring,
        gen_permutation_importance,
    )


@python.define(outputs={"shaps": ty.Any})
def GetShap(
    X: ty.Any,
    permute: ty.Any,
    model: ty.Any,
    gen_shap: bool = False,
    nsamples: ty.Any = "auto",
    l1_reg: ty.Any = "aic",
) -> ty.Any:
    return get_shap(X, permute, model, gen_shap, nsamples, l1_reg)


@python.define(outputs={"output": ty.Any, "model": ty.Any})
def CreateModel(
    X: ty.Any,
    y: ty.Any,
    clf_info: ty.Any,
    permute: ty.Any,
    balancing: ty.Any = None,
    balancing_bins: ty.Any = None,
    groups: ty.Any = None,
) -> tuple[ty.Any, ty.Any]:
    return create_model(X, y, clf_info, permute, balancing, balancing_bins, groups)


# --- Workflow definition ---


@workflow.define(
    outputs={
        "output": ty.Any,
        "score": ty.Any,
        "feature_importance": ty.Any,
        "permutation_importance": ty.Any,
        "shaps": ty.Any,
        "feature_names": ty.Any,
        "model": ty.Any,
    }
)
def MLWorkflow(
    filename: ty.Any,
    x_indices: ty.Any,
    target_vars: ty.Any,
    group_var: ty.Any,
    n_splits: int,
    test_size: float,
    clf_info: ty.Any,
    permute: ty.Any,
    metrics: ty.Any,
    gen_feature_importance: bool,
    gen_permutation_importance: bool,
    permutation_importance_n_repeats: int,
    permutation_importance_scoring: ty.Any,
    gen_shap: bool,
    nsamples: ty.Any,
    l1_reg: ty.Any,
    plot_top_n_shap: ty.Any,
    balancing: ty.Any = None,
    bootstrap_strategy: str = "group_shuffle",
    n_bins: int = 10,
    balancing_bins: ty.Any = None,
) -> tuple[ty.Any, ty.Any, ty.Any, ty.Any, ty.Any, ty.Any, ty.Any]:
    readcsv = workflow.add(
        ReadFile(
            filename=filename,
            x_indices=x_indices,
            target_vars=target_vars,
            group=group_var,
        ),
        name="readcsv",
    )
    gensplit = workflow.add(
        GenSplits(
            n_splits=n_splits,
            test_size=test_size,
            X=readcsv.X,
            Y=readcsv.Y,
            groups=readcsv.groups,
            bootstrap_strategy=bootstrap_strategy,
            n_bins=n_bins,
        ),
        name="gensplit",
    )
    # Split TrainTestKernel over each bootstrap split index
    fit_clf = workflow.add(
        TrainTestKernel(
            X=readcsv.X,
            y=readcsv.Y,
            train_test_split=gensplit.splits,
            split_index=gensplit.split_indices,
            clf_info=clf_info,
            permute=permute,
            balancing=balancing,
            balancing_bins=balancing_bins,
            groups=readcsv.groups,
        ).split("split_index", split_index=gensplit.split_indices),
        name="fit_clf",
    )
    # Combine CalcMetric results across splits
    metric = workflow.add(
        CalcMetric(
            output=fit_clf.output,
            metrics=metrics,
        ).combine("fit_clf.split_index"),
        name="metric",
    )
    fi = workflow.add(
        GetFeatureImportance(
            permute=permute,
            model=fit_clf.model,
            gen_feature_importance=gen_feature_importance,
        ).combine("fit_clf.split_index"),
        name="feature_importance",
    )
    pi = workflow.add(
        GetPermutationImportance(
            X=readcsv.X,
            y=readcsv.Y,
            permute=permute,
            model=fit_clf.model,
            permutation_importance_n_repeats=permutation_importance_n_repeats,
            permutation_importance_scoring=permutation_importance_scoring,
            gen_permutation_importance=gen_permutation_importance,
        ).combine("fit_clf.split_index"),
        name="permutation_importance",
    )
    shap_node = workflow.add(
        GetShap(
            X=readcsv.X,
            permute=permute,
            model=fit_clf.model,
            gen_shap=gen_shap,
            nsamples=nsamples,
            l1_reg=l1_reg,
        ).combine("fit_clf.split_index"),
        name="shap",
    )
    cm = workflow.add(
        CreateModel(
            X=readcsv.X,
            y=readcsv.Y,
            clf_info=clf_info,
            permute=permute,
            balancing=balancing,
            balancing_bins=balancing_bins,
            groups=readcsv.groups,
        ),
        name="create_model",
    )
    return (
        metric.output,
        metric.score,
        fi.feature_importance,
        pi.permutation_importance,
        shap_node.shaps,
        readcsv.feature_names,
        cm.model,
    )


@dataclasses.dataclass
class WorkflowSpec:
    """Bundles the split workflow task with metadata needed by run_workflow."""

    wf: ty.Any
    cache_dir: ty.Any
    inputs: dict


def gen_workflow(inputs, cache_dir=None, cache_locations=None):
    """Build the ML workflow split over clf_info × permute combinations.

    Returns a WorkflowSpec that can be passed to run_workflow.
    """
    clf_infos = inputs["clf_info"]
    permutes = inputs["permute"]
    # Construct with a placeholder scalar for clf_info/permute; .split() overrides them.
    wf = MLWorkflow(
        filename=inputs["filename"],
        x_indices=inputs["x_indices"],
        target_vars=inputs["target_vars"],
        group_var=inputs["group_var"],
        n_splits=inputs["n_splits"],
        test_size=inputs["test_size"],
        clf_info=clf_infos[0],
        permute=permutes[0],
        metrics=inputs["metrics"],
        gen_feature_importance=inputs["gen_feature_importance"],
        gen_permutation_importance=inputs["gen_permutation_importance"],
        permutation_importance_n_repeats=inputs["permutation_importance_n_repeats"],
        permutation_importance_scoring=inputs["permutation_importance_scoring"],
        gen_shap=inputs["gen_shap"],
        nsamples=inputs["nsamples"],
        l1_reg=inputs["l1_reg"],
        plot_top_n_shap=inputs["plot_top_n_shap"],
        balancing=inputs.get("balancing", None),
        bootstrap_strategy=inputs.get("bootstrap_strategy", "group_shuffle"),
        n_bins=inputs.get("n_bins", 10),
        balancing_bins=inputs.get("balancing_bins", None),
    ).split(["clf_info", "permute"], clf_info=clf_infos, permute=permutes)
    return WorkflowSpec(wf=wf, cache_dir=cache_dir, inputs=inputs)


def _format_results(result, clf_infos, permutes):
    """Adapt pydra 1.0 SplitOutputs into the list-of-(params, result) format
    expected by gen_report."""

    class _Output:
        pass

    class _Result:
        output = None

    formatted = []
    for i, (clf_info, permute) in enumerate(itertools.product(clf_infos, permutes)):
        params = {"ml_wf.clf_info": clf_info, "ml_wf.permute": permute}
        r = _Result()
        r.output = _Output()
        r.output.score = result.outputs.score[i]
        r.output.output = result.outputs.output[i]
        r.output.feature_importance = result.outputs.feature_importance[i]
        r.output.permutation_importance = result.outputs.permutation_importance[i]
        r.output.shaps = result.outputs.shaps[i]
        r.output.feature_names = result.outputs.feature_names[i]
        r.output.model = result.outputs.model[i]
        formatted.append((params, r))
    return formatted


def run_workflow(spec, worker, worker_args, specfile="localspec"):
    """Execute the workflow and generate reports.

    Parameters
    ----------
    spec : WorkflowSpec
        Returned by gen_workflow.
    worker : str
        Pydra worker name (e.g. 'cf', 'debug', 'dask').
    worker_args : dict
        Arguments forwarded to the worker (e.g. {'n_procs': 4}).
    specfile : str
        Base name used for the output directory.

    Returns
    -------
    pydra.engine.result.Result
        The raw pydra result with SplitOutputs.
    """
    wf = spec.wf
    cache_dir = spec.cache_dir
    inputs = spec.inputs

    cwd = os.getcwd()
    with Submitter(
        worker=worker,
        cache_root=cache_dir,
        **worker_args,
    ) as sub:
        result = sub(wf)
    os.chdir(cwd)

    import datetime
    import pickle as pk

    timestamp = datetime.datetime.utcnow().isoformat()
    timestamp = timestamp.replace(":", "").replace("-", "")
    result_dir = f"out-{os.path.basename(specfile)}-{timestamp}"
    os.makedirs(result_dir)
    os.chdir(result_dir)

    with open(f"results-{timestamp}.pkl", "wb") as fp:
        pk.dump(result, fp)

    clf_infos = inputs["clf_info"]
    permutes = inputs["permute"]
    formatted = _format_results(result, clf_infos, permutes)
    gen_report(
        formatted,
        prefix="ml_wf",
        metrics=inputs["metrics"],
        gen_shap=inputs["gen_shap"],
        plot_top_n_shap=inputs["plot_top_n_shap"],
    )
    os.chdir(cwd)
    return result
