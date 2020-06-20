#!/usr/bin/env python

import pydra
from pydra.mark import task, annotate
from pydra.utils.messenger import AuditFlag, FileMessenger
import typing as ty
import os
from .tasks import read_file, gen_splits, train_test_kernel, calc_metric, get_shap
from .report import gen_report

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

get_shap_pdt = task(annotate({"return": {"shaps": ty.Any}})(get_shap))


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
    wf.split(["clf_info", "permute"])
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
    wf.fit_clf.split("split_index")
    wf.add(
        calc_metric_pdt(
            name="metric", output=wf.fit_clf.lzout.output, metrics=wf.lzin.metrics
        )
    )
    wf.metric.combine("fit_clf.split_index")
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
    wf.set_output(
        [
            ("output", wf.metric.lzout.output),
            ("score", wf.metric.lzout.score),
            ("shaps", wf.shap.lzout.shaps),
            ("feature_names", wf.readcsv.lzout.feature_names),
        ]
    )
    return wf


def run_workflow(wf, plugin, plugin_args, specfile="localspec"):
    cwd = os.getcwd()
    with pydra.Submitter(plugin=plugin, **plugin_args) as sub:
        sub(runnable=wf)
    results = wf.result(return_inputs=True)
    os.chdir(cwd)

    import pickle as pk
    import datetime

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
