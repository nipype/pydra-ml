import os
import json
import click
from .classifier import gen_workflow, run_workflow


@click.command()
@click.option("-s", "--specfile", help="Specification file to use", required=True)
@click.option(
    "-n", "--nprocs", default=1, help="Number of processors to use", show_default=True
)
@click.option(
    "-c",
    "--cache",
    default=os.path.join(os.getcwd(), "cache-wf"),
    help="Cache dir",
    show_default=True,
)
def main(specfile, nprocs, cache):
    with open(specfile) as fp:
        spec = json.load(fp)
    spec["filename"] = os.path.abspath(spec["filename"])
    if not os.path.exists(spec["filename"]):
        raise FileNotFoundError(f"{spec['filename']} does not exist.")
    wf = gen_workflow(spec, cache_dir=cache)
    run_workflow(wf, nprocs)
