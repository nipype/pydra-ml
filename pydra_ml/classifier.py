#!/usr/bin/env python

import os
import typing as ty

import pydra
from pydra.mark import annotate, task
from pydra.utils.messenger import AuditFlag, FileMessenger

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

# Create pydra tasks
read_file_pdt = task(
    annotate(
        {
            "return": {
                "X": ty.Any,
                "Y": ty.Any,
                "groups": ty.Any,
                "feature_names": ty.Any,
            }
        }
    )(read_file)
)

gen_splits_pdt = task(
    annotate({"return": {"splits": ty.Any, "split_indices": ty.Any}})(gen_splits)
)

train_test_kernel_pdt = task(
    annotate({"return": {"output": ty.Any, "model": ty.Any}})(train_test_kernel)
)

calc_metric_pdt = task(
    annotate({"return": {"score": ty.Any, "output": ty.Any}})(calc_metric)
)

get_feature_importance_pdt = task(
    annotate({"return": {"feature_importance": ty.Any}})(get_feature_importance)
)

get_permutation_importance_pdt = task(
    annotate({"return": {"permutation_importance": ty.Any}})(get_permutation_importance)
)

get_shap_pdt = task(annotate({"return": {"shaps": ty.Any}})(get_shap))

create_model_pdt = task(
    annotate({"return": {"output": ty.Any, "model": ty.Any}})(create_model)
)


def gen_workflow(inputs, cache_dir=None, cache_locations=None):
    wf = pydra.Workflow(
        name="ml_wf",
        input_spec=list(inputs.keys()),
        **inputs,
        cache_dir=cache_dir,
        cache_locations=cache_locations,
        audit_flags=AuditFlag.ALL,
        messengers=FileMessenger(),
        messenger_args={"message_dir": os.path.join(os.getcwd(), "messages")},
    )
    wf.split(clf_info=inputs["clf_info"], permute=inputs["permute"])
    wf.add(
        read_file_pdt(
            name="readcsv",
            filename=wf.lzin.filename,
            x_indices=wf.lzin.x_indices,
            target_vars=wf.lzin.target_vars,
            group=wf.lzin.group_var,
        )
    )
    wf.add(
        gen_splits_pdt(
            name="gensplit",
            n_splits=wf.lzin.n_splits,
            test_size=wf.lzin.test_size,
            X=wf.readcsv.lzout.X,
            Y=wf.readcsv.lzout.Y,
            groups=wf.readcsv.lzout.groups,
        )
    )
    wf.add(
        train_test_kernel_pdt(
            name="fit_clf",
            X=wf.readcsv.lzout.X,
            y=wf.readcsv.lzout.Y,
            train_test_split=wf.gensplit.lzout.splits,
            split_index=wf.gensplit.lzout.split_indices,
            clf_info=wf.lzin.clf_info,
            permute=wf.lzin.permute,
        )
    )
    wf.fit_clf.split(split_index=wf.gensplit.lzout.split_indices)
    wf.add(
        calc_metric_pdt(
            name="metric", output=wf.fit_clf.lzout.output, metrics=wf.lzin.metrics
        )
    )
    wf.metric.combine("fit_clf.split_index")
    wf.add(
        get_feature_importance_pdt(
            name="feature_importance",
            permute=wf.lzin.permute,
            model=wf.fit_clf.lzout.model,
            gen_feature_importance=wf.lzin.gen_feature_importance,
        )
    )
    wf.feature_importance.combine("fit_clf.split_index")
    wf.add(
        get_permutation_importance_pdt(
            name="permutation_importance",
            X=wf.readcsv.lzout.X,
            y=wf.readcsv.lzout.Y,
            permute=wf.lzin.permute,
            model=wf.fit_clf.lzout.model,
            permutation_importance_n_repeats=wf.lzin.permutation_importance_n_repeats,
            permutation_importance_scoring=wf.lzin.permutation_importance_scoring,
            gen_permutation_importance=wf.lzin.gen_permutation_importance,
        )
    )
    wf.permutation_importance.combine("fit_clf.split_index")
    wf.add(
        get_shap_pdt(
            name="shap",
            X=wf.readcsv.lzout.X,
            permute=wf.lzin.permute,
            model=wf.fit_clf.lzout.model,
            gen_shap=wf.lzin.gen_shap,
            nsamples=wf.lzin.nsamples,
            l1_reg=wf.lzin.l1_reg,
        )
    )
    wf.shap.combine("fit_clf.split_index")
    wf.add(
        create_model_pdt(
            name="create_model",
            X=wf.readcsv.lzout.X,
            y=wf.readcsv.lzout.Y,
            clf_info=wf.lzin.clf_info,
            permute=wf.lzin.permute,
        )
    )
    wf.set_output(
        [
            ("output", wf.metric.lzout.output),
            ("score", wf.metric.lzout.score),
            ("feature_importance", wf.feature_importance.lzout.feature_importance),
            (
                "permutation_importance",
                wf.permutation_importance.lzout.permutation_importance,
            ),
            ("shaps", wf.shap.lzout.shaps),
            ("feature_names", wf.readcsv.lzout.feature_names),
            ("model", wf.create_model.lzout.model),
        ]
    )
    return wf


def run_workflow(wf, plugin, plugin_args, specfile="localspec"):
    cwd = os.getcwd()
    with pydra.Submitter(plugin=plugin, **plugin_args) as sub:
        sub(runnable=wf)
    results = wf.result(return_inputs=True)
    os.chdir(cwd)

    import datetime
    import pickle as pk

    timestamp = datetime.datetime.utcnow().isoformat()
    timestamp = timestamp.replace(":", "").replace("-", "")
    result_dir = f"out-{os.path.basename(specfile)}-{timestamp}"
    os.makedirs(result_dir)
    os.chdir(result_dir)
    with open(f"results-{timestamp}.pkl", "wb") as fp:
        pk.dump(results, fp)

    gen_report(
        results,
        prefix=wf.name,
        metrics=wf.inputs.metrics,
        gen_shap=wf.inputs.gen_shap,
        plot_top_n_shap=wf.inputs.plot_top_n_shap,
    )
    os.chdir(cwd)
    return results
