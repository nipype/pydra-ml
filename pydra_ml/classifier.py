#!/usr/bin/env python

import pydra
import os
from .tasks import read_file, gen_splits, train_test_kernel, calc_metric, get_shap
from .report import gen_report


def gen_workflow(inputs, cache_dir=None, cache_locations=None):
    wf = pydra.Workflow(
        name="ml_wf",
        input_spec=list(inputs.keys()),
        **inputs,
        cache_dir=cache_dir,
        cache_locations=cache_locations,
    )
    wf.split(["clf_info", "permute"])
    wf.add(
        read_file(
            name="readcsv",
            filename=wf.lzin.filename,
            x_indices=wf.lzin.x_indices,
            target_vars=wf.lzin.target_vars,
        )
    )
    wf.add(
        gen_splits(
            name="gensplit",
            n_splits=wf.lzin.n_splits,
            test_size=wf.lzin.test_size,
            X=wf.readcsv.lzout.X,
            Y=wf.readcsv.lzout.Y,
            groups=wf.readcsv.lzout.groups,
        )
    )
    wf.add(
        train_test_kernel(
            name="fit_clf",
            X=wf.readcsv.lzout.X,
            y=wf.readcsv.lzout.Y,
            train_test_split=wf.gensplit.lzout.splits,
            split_index=wf.gensplit.lzout.split_indices,
            clf_info=wf.lzin.clf_info,
            permute=wf.lzin.permute,
            metrics=wf.lzin.metrics,
        )
    )
    wf.fit_clf.split("split_index")
    wf.add(
        calc_metric(
            name="metric", output=wf.fit_clf.lzout.output, metrics=wf.lzin.metrics
        )
    )
    wf.metric.combine("fit_clf.split_index")
    wf.add(
        get_shap(
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


def run_workflow(wf, plugin, plugin_args):
    cwd = os.getcwd()
    with pydra.Submitter(plugin=plugin, **plugin_args) as sub:
        sub(runnable=wf)
    results = wf.result(return_inputs=True)
    os.chdir(cwd)
    import pickle as pk
    import datetime

    timestamp = datetime.datetime.utcnow().isoformat()
    with open(f"results-{timestamp}.pkl", "wb") as fp:
        pk.dump(results, fp)

    gen_report(
        results,
        prefix=wf.name,
        metrics=wf.inputs.metrics,
        gen_shap=wf.inputs.gen_shap,
        plot_top_n_shap=wf.inputs.plot_top_n_shap,
    )
    return results
