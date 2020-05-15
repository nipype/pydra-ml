#!/usr/bin/env python

import pydra
import os
from .tasks import read_file, gen_splits, train_test_kernel, get_shap
from .report import gen_report


def gen_workflow(inputs, cache_dir=None, cache_locations=None):
    wf = pydra.Workflow(
        name="ml_wf",
        input_spec=list(inputs.keys()),
        **inputs,
        cache_dir=cache_dir,
        cache_locations=cache_locations
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
        )
    )
    wf.fit_clf.split("split_index").combine("split_index")
    wf.add(
        get_shap(
            name="shap",
            X=wf.readcsv.lzout.X,
            train_test_split=wf.gensplit.lzout.splits,
            split_index=wf.gensplit.lzout.split_indices,
            permute=wf.lzin.permute,
            model=wf.fit_clf.lzout.model,
            noshap=wf.lzin.noshap,
            nsamples=wf.lzin.nsamples,
            l1_reg=wf.lzin.l1_reg,
        )
    )
    wf.shap.split(("split_index", "model")).combine("split_index")
    wf.set_output(
        [
            ("output", wf.fit_clf.lzout.output),
            ("auc", wf.fit_clf.lzout.auc),
            ("shaps", wf.shap.lzout.shaps),
        ]
    )
    return wf


def run_workflow(wf, n_procs):
    with pydra.Submitter(plugin="cf", n_procs=n_procs) as sub:
        sub(runnable=wf)
    results = wf.result(return_inputs=True)
    gen_report(results, prefix=wf.name)
    print(results)
    return results
