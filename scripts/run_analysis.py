from __future__ import annotations

import os
import socket

import dask
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster, performance_report

from egamma_tnp.utils import runner_utils


def main():
    parser = runner_utils.get_main_parser()
    args = parser.parse_args()
    if args.executor != "distributed":
        if args.scaleout is None:
            args.scaleout = 100
        if args.cores is None:
            args.cores = 1
        if args.memory is None:
            args.memory = "4GB"
        if args.disk is None:
            args.disk = "4GB"
    if args.executor == "distributed":
        if args.memory is None:
            args.memory = "auto"

    config = runner_utils.load_json(args.config)
    settings = runner_utils.load_settings(args.settings)
    args = runner_utils.merge_settings_with_args(args, settings)
    if args.binning:
        runner_utils.set_binning(runner_utils.load_json(args.binning))
    fileset = runner_utils.load_json(args.fileset)
    instance = runner_utils.initialize_class(config, args, fileset)

    if args.voms is not None:
        _x509_path = args.voms
    else:
        _x509_path = runner_utils.get_proxy()

    cluster = None
    client = None
    scheduler = None
    if args.executor in ["multiprocessing", "processes", "single-threaded", "sync", "synchronous", "threading", "threads"]:
        scheduler = args.executor
    elif args.executor == "distributed":
        cluster = LocalCluster(n_workers=args.scaleout, threads_per_worker=args.cores, memory_limit=args.memory, dashboard_address=args.dashboard_address)
    elif args.executor == "dask/lpc":
        from lpcjobqueue import LPCCondorCluster

        cluster = LPCCondorCluster(
            ship_env=True,
            scheduler_options={"dashboard_address": args.dashboard_address},
            memory=args.memory,
            disk=args.disk,
            cores=args.cores,
        )
        scheduler = "distributed"
    elif args.executor == "dask/lxplus":
        from dask_lxplus import CernCluster

        cluster = CernCluster(
            cores=args.cores,
            memory=args.memory,
            disk=args.disk,
            image_type="singularity",
            worker_image="/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-egamma/egamma-tnp:lxplus-el9-latest",
            death_timeout="3600",
            scheduler_options={"port": args.port, "host": socket.gethostname(), "dashboard_address": args.dashboard_address},
            job_extra={
                "log": "dask_job_output.log",
                "output": "dask_job_output.out",
                "error": "dask_job_output.err",
                "should_transfer_files": "Yes",
                "when_to_transfer_output": "ON_EXIT",
                "+JobFlavour": f'"{args.jobflavour}"',
            },
            job_script_prologue=[
                "export XRD_RUNFORKHANDLER=1",
                f"export X509_USER_PROXY={_x509_path}",
                "export PYTHONPATH=$PYTHONPATH:$_CONDOR_SCRATCH_DIR",
            ],
        )
        scheduler = "distributed"
    elif args.executor == "dask/slurm":
        from dask_jobqueue import SLURMCluster

        cluster = SLURMCluster(
            queue=args.queue,
            cores=args.cores,
            memory=args.memory,
            walltime=args.walltime,
            job_script_prologue=[
                "export XRD_RUNFORKHANDLER=1",
                f"export X509_USER_PROXY={_x509_path}",
                f"export PYTHONPATH=$PYTHONPATH:{os.getcwd()}",
            ],
            scheduler_options={"dashboard_address": args.dashboard_address},
        )
        scheduler = "distributed"
    elif args.executor == "dask/condor":
        from dask_jobqueue import HTCondorCluster

        cluster = HTCondorCluster(
            cores=args.cores,
            memory=args.memory,
            disk=args.disk,
            job_script_prologue=[
                "export XRD_RUNFORKHANDLER=1",
                f"export X509_USER_PROXY={_x509_path}",
                f"export PYTHONPATH=$PYTHONPATH:{os.getcwd()}",
            ],
            scheduler_options={"dashboard_address": args.dashboard_address},
        )
        scheduler = "distributed"
    else:
        raise ValueError(f"Unknown executor `{args.executor}`")

    if cluster:
        if args.adaptive and args.executor != "distributed":
            cluster.adapt(minimum=0, maximum=args.scaleout)
        elif not args.adaptive and args.executor != "distributed":
            cluster.scale(args.scaleout)
        client = Client(cluster)

    print("Calculating task graph")  # noqa: T201
    print("METHODS:\n", config["methods"])  # noqa: T201
    to_compute = runner_utils.run_methods(instance, config["methods"])
    print("TO COMPUTE:\n", to_compute)  # noqa: T201
    to_compute = runner_utils.process_to_compute(to_compute, args.output)
    print("TO COMPUTE:\n", to_compute)  # noqa: T201
    print("Calculating necessary columns")  # noqa: T201
    # print("NECESSARY COLUMNS:\n", dak.necessary_columns(to_compute))
    print("Visualizing task graph")  # noqa: T201
    # dask.visualize(to_compute, filename="/tmp/graph-unoptimized.pdf", optimize_graph=False)
    # dask.visualize(to_compute, filename="/tmp/graph-optimized.pdf", optimize_graph=True)

    if client:
        with performance_report(filename="/tmp/dask-report.html"):
            (out,) = dask.compute(to_compute, scheduler="distributed")
    else:
        with ProgressBar():
            (out,) = dask.compute(to_compute, scheduler=scheduler)
    print(out)  # noqa: T201
    out = runner_utils.process_out(out, args.output)
    print(out)  # noqa: T201


if __name__ == "__main__":
    main()
