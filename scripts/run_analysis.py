from __future__ import annotations

import os
import socket

import dask
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster, performance_report

from egamma_tnp.utils import runner_utils


def main():
    parser = runner_utils.get_main_parser()
    parser.add_argument("--port", type=int, default=8786, help="Port for the Dask scheduler")
    parser.add_argument("--dashboard_address", type=str, default=":8787", help="Address for the Dask dashboard")
    parser.add_argument("--cores", type=int, default=1, help="Number of cores for each worker")
    parser.add_argument("--memory", type=str, default="4GB", help="Memory allocation for each worker")
    parser.add_argument("--disk", type=str, default="4GB", help="Disk allocation for each worker")
    parser.add_argument("--scaleout", type=int, default=100, help="Maximum number of workers")
    parser.add_argument("--adaptive", type=bool, default=True, help="Adaptive scaling")
    parser.add_argument("--queue", type=str, help="Queue for job submission")
    parser.add_argument("--walltime", type=str, help="Walltime for job execution")
    args = parser.parse_args()

    config = runner_utils.load_json(args.config)
    if args.binning:
        runner_utils.set_binning(runner_utils.load_json(args.binning))
    fileset = runner_utils.load_json(args.fileset)

    instance = runner_utils.initialize_class(config, args, fileset)

    cluster = None
    client = None
    scheduler = None
    if args.executor in ["multiprocessing", "processes", "single-threaded", "sync", "synchronous", "threading", "threads"]:
        scheduler = args.executor
    elif args.executor == "distributed":
        default_cores = 1
        default_memory = "4GB"
        default_scaleout = 100

        cluster = LocalCluster(
            n_workers=args.scaleout if args.scaleout != default_scaleout else None,
            threads_per_worker=args.cores if args.cores != default_cores else None,
            memory_limit=args.memory if args.memory != default_memory else None,
            dashboard_address=args.dashboard_address,
        )
    elif args.executor == "dask/lpc":
        from lpcjobqueue import LPCCondorCluster

        cluster = LPCCondorCluster(
            ship_env=True,
            scheduler_options={"dashboard_address": args.dashboard_address},
            job_script_prologue=[
                f"export PYTHONPATH=$PYTHONPATH:{os.getcwd()}",
            ],
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
                "transfer_input_files": "root://eosuser.cern.ch//eos/user/i/ikrommyd/voms/x509px",
                "+JobFlavour": '"longlunch"',
            },
            job_script_prologue=[
                "export XRD_RUNFORKHANDLER=1",
                "export X509_USER_PROXY=${_CONDOR_SCRATCH_DIR}/x509px",
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
                f"export PYTHONPATH=$PYTHONPATH:{os.getcwd()}",
            ],
            scheduler_options={"dashboard_address": args.dashboard_address},
        )
        scheduler = "distributed"
    else:
        raise ValueError(f"Unknown executor: {args.executor}")

    to_compute = runner_utils.run_methods(instance, config["methods"])

    if cluster:
        if args.adaptive and args.executor != "distributed":
            cluster.adapt(minimum=1, maximum=args.scaleout)
        elif not args.adaptive and args.executor != "distributed":
            cluster.scale(args.scaleout)
        client = Client(cluster)

    if client:
        with performance_report(filename="/tmp/dask-report.html"):
            out = dask.compute(to_compute, scheduler="distributed")
    else:
        with ProgressBar():
            out = dask.compute(to_compute, scheduler=scheduler)

    print(out)  # noqa: T201


if __name__ == "__main__":
    main()
