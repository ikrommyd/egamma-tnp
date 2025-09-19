from __future__ import annotations

import getpass
import gzip
import json
import os
import socket
import warnings

import dask
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster, performance_report, progress

from egamma_tnp.config import binning_manager
from egamma_tnp.utils import runner_utils
from egamma_tnp.utils.logger_utils import setup_logger
from egamma_tnp.utils.misc import check_port, get_proxy


def main():
    warnings.filterwarnings("ignore", category=FutureWarning, module="coffea.*")
    parser = runner_utils.get_main_parser()
    args = parser.parse_args()
    if args.debug:
        logger = setup_logger(level="DEBUG")
    else:
        logger = setup_logger(level="INFO")
    logger.info("Starting the E/Gamma Tag and Probe workflow")
    logger.info(f"Default binning is located at {os.path.join(os.path.dirname(binning_manager.__file__), 'default_binning.json')}")
    logger.info(f"Runtime binning is located at {os.path.join(os.path.dirname(binning_manager.__file__), f'/tmp/runtime_binning_{getpass.getuser()}.json')}")

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
    logger.info(f"Loaded config from {args.config}")
    settings = runner_utils.load_settings(args.settings)
    logger.info(f"Loaded settings from {args.settings}")
    args = runner_utils.merge_settings_with_args(args, settings)
    if args.binning:
        logger.info(f"Overwriting default binning with {args.binning}")
        runner_utils.set_binning(runner_utils.load_json(args.binning))
    fileset = runner_utils.load_json(args.fileset)
    logger.info(f"Loaded fileset from {args.fileset}")

    if args.limit == 0:
        raise ValueError("Limit cannot be 0")
    if args.limit is not None:
        logger.info(f"Limiting each dataset of the fileset to the first {args.limit} files")
        for dataset in fileset.keys():
            files = fileset[dataset]["files"]
            fileset[dataset]["files"] = dict(list(files.items())[: args.limit])

    if args.executor == "dask/casa" or args.executor.startswith("tls://"):
        # use xcache for coffea-casa
        xrootd_pfx = "root://"
        xrd_pfx_len = len(xrootd_pfx)
        for dataset in fileset.keys():
            files = fileset[dataset]["files"]
            newfiles = {}
            for path, value in files.items():
                newpath = path.replace(path[xrd_pfx_len : xrd_pfx_len + path[xrd_pfx_len:].find("/store")], "xcache/")
                newfiles[newpath] = value
            fileset[dataset]["files"] = newfiles

    if args.preprocess:
        from coffea.dataset_tools import preprocess

        client = Client(dashboard_address=args.dashboard_address)
        logger.info(f"Preprocessing the fileset with client: {client}")
        fileset = preprocess(fileset, step_size=100_000, skip_bad_files=True, scheduler=client)[0]
        logger.info("Done preprocessing the fileset")
        client.shutdown()

        with gzip.open("/tmp/preprocessed_fileset.json.gz", "wt") as f:
            logger.info("Saving the preprocessed fileset to /tmp/preprocessed_fileset.json.gz")
            json.dump(fileset, f, indent=2)

    instance = runner_utils.initialize_class(config, args, fileset)

    if args.port is not None and (not args.executor.startswith("tls://") and not args.executor.startswith("tcp://") and not args.executor.startswith("ucx://")):
        if not check_port(args.port):
            logger.error(f"Port {args.port} is occupied in this node. Try another one.")
            raise ValueError(f"Port {args.port} is occupied in this node. Try another one.")

    if args.voms is not None:
        _x509_path = args.voms
    else:
        _x509_path = get_proxy()

    cluster = None
    client = None
    scheduler = None
    if args.executor in ["multiprocessing", "processes", "single-threaded", "sync", "synchronous", "threading", "threads"]:
        logger.info(f"Running locally with {args.executor} scheduler")
        scheduler = args.executor
    elif args.executor == "distributed":
        logger.info("Running using LocalCluster")
        cluster = LocalCluster(
            n_workers=args.scaleout,
            threads_per_worker=args.cores,
            memory_limit=args.memory,
            dashboard_address=args.dashboard_address,
        )
    elif args.executor == "dask/lpc":
        from lpcjobqueue import LPCCondorCluster

        logger.info("Running using LPCCondorCluster")
        cluster = LPCCondorCluster(
            ship_env=True,
            scheduler_options={"dashboard_address": args.dashboard_address},
            memory=args.memory,
            disk=args.disk,
            cores=args.cores,
            log_directory=args.log_directory,
            job_script_prologue=["export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH"],
        )
        scheduler = "distributed"
    elif args.executor == "dask/lxplus":
        from dask_lxplus import CernCluster

        logger.info("Running using CernCluster")
        cluster = CernCluster(
            cores=args.cores,
            memory=args.memory,
            disk=args.disk,
            image_type="singularity",
            worker_image="/cvmfs/unpacked.cern.ch/registry.cern.ch/cms-egamma/egamma-tnp:lxplus-el9-latest",
            death_timeout="3600",
            scheduler_options={"port": args.port, "host": socket.gethostname(), "dashboard_address": args.dashboard_address},
            log_directory=args.log_directory,
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
    elif args.executor == "dask/casa":
        from coffea_casa import CoffeaCasaCluster

        logger.info("Running using CoffeaCasaCluster")
        cluster = CoffeaCasaCluster(
            cores=args.cores,
            memory=args.memory,
            disk=args.disk,
            scheduler_options={"port": args.port, "dashboard_address": args.dashboard_address},
            log_directory=args.log_directory,
        )
        scheduler = "distributed"
    elif args.executor == "dask/slurm":
        from dask_jobqueue import SLURMCluster

        logger.info("Running using SLURMCluster")
        cluster = SLURMCluster(
            queue=args.queue,
            cores=args.cores,
            memory=args.memory,
            walltime=args.walltime,
            log_directory=args.log_directory,
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

        logger.info("Running using HTCondorCluster")
        cluster = HTCondorCluster(
            cores=args.cores,
            memory=args.memory,
            disk=args.disk,
            log_directory=args.log_directory,
            job_script_prologue=[
                "export XRD_RUNFORKHANDLER=1",
                f"export X509_USER_PROXY={_x509_path}",
                f"export PYTHONPATH=$PYTHONPATH:{os.getcwd()}",
            ],
            scheduler_options={"dashboard_address": args.dashboard_address},
        )
        scheduler = "distributed"
    elif args.executor is not None and (args.executor.startswith("tls://") or args.executor.startswith("tcp://") or args.executor.startswith("ucx://")):
        logger.info(f"Will use dask scheduler at {args.executor}")
    elif args.executor is None:
        logger.info("Running with default dask scheduler")
    else:
        logger.error(f"Unknown executor `{args.executor}`")
        raise ValueError(f"Unknown executor `{args.executor}`")

    if cluster:
        if args.adaptive and args.executor != "distributed":
            cluster.adapt(minimum=0, maximum=args.scaleout)
        elif not args.adaptive and args.executor != "distributed":
            cluster.scale(args.scaleout)
        logger.info(f"Set up cluster {cluster}")
        client = Client(cluster)
        if args.executor == "dask/casa" or args.executor.startswith("tls://"):
            from dask.distributed import PipInstall

            plugin = PipInstall(packages=["egamma-tnp@git+https://${TOKEN}@github.com/ikrommyd/egamma-tnp.git@master"])
            client.register_plugin(plugin)
        logger.info(f"Set up client {client}")
    if args.executor is not None and (args.executor.startswith("tls://") or args.executor.startswith("tcp://") or args.executor.startswith("ucx://")):
        client = Client(args.executor)
        logger.info(f"Set up client {client}")

    logger.info(f"Calculating task graph for methods: {config['methods']} on workflow: {instance}")
    to_compute = runner_utils.run_methods(instance, config["methods"])
    to_compute = runner_utils.process_to_compute(to_compute, args.output, args.repartition_n_to_one, args.skip_report)
    logger.info(f"Object to compute is:\n{to_compute}")
    if args.print_necessary_columns:
        import dask_awkward as dak

        necessary_columns = dak.necessary_columns(to_compute)
        logger.info(f"The necessary columns are:\n{necessary_columns}")
    logger.info("Computing the task graph")
    if client:
        with performance_report(filename="/tmp/dask-report.html"):
            logger.info("The performance report will be saved in /tmp/dask-report.html")
            (futures,) = dask.persist(to_compute)
            progress(futures)
            (out,) = dask.compute(futures)
    else:
        with ProgressBar():
            (out,) = dask.compute(to_compute, scheduler=scheduler)
    logger.info(f"Computed object is:\n{out}")
    out = runner_utils.process_out(out, args.output)
    logger.info(f"Final output after post-processing:\n{out}")
    logger.info("Finished the E/Gamma Tag and Probe workflow")
    if cluster:
        logger.info("Shutting down the cluster")
        cluster.close()
    if client:
        logger.info("Shutting down the client")
        client.close()


if __name__ == "__main__":
    main()
