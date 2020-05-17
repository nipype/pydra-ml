import os
import json
import click
from .classifier import gen_workflow, run_workflow


@click.command()
@click.option(
    "-s",
    "--specfile",
    help="Specification file to use",
    required=True,
    type=click.Path(exists=True, resolve_path=True),
)
@click.option(
    "-p",
    "--plugin",
    nargs=2,
    default=["cf", "n_procs=1"],
    help="Pydra plugin to use",
    show_default=True,
)
@click.option(
    "-c",
    "--cache",
    default=os.path.join(os.getcwd(), "cache-wf"),
    help="Cache dir",
    show_default=True,
)
def main(specfile, plugin, cache):
    with open(specfile) as fp:
        spec = json.load(fp)
    spec["filename"] = os.path.abspath(spec["filename"])
    if not os.path.exists(spec["filename"]):
        raise FileNotFoundError(f"{spec['filename']} does not exist.")
    if len(spec["target_vars"]) > 1:
        raise ValueError(
            f"At present only one target_vars "
            f"({len(spec['target_vars'])} provided) is supported."
        )
    wf = gen_workflow(spec, cache_dir=cache)
    plugin_args = dict()
    for item in plugin[1].split():
        key, value = item.split("=")
        if plugin[0] == "cf" and key == "n_procs":
            value = int(value)
        plugin_args[key] = value
    run_workflow(wf, plugin[0], plugin_args)
