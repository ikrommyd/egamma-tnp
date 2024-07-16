from __future__ import annotations

from egamma_tnp.utils import runner_utils


def main():
    parser = runner_utils.get_main_parser()
    args = parser.parse_args()

    config = runner_utils.load_json(args.config)
    runner_utils.set_binning(config["binning"])
    fileset = runner_utils.load_json(args.fileset)

    filtered_args = runner_utils.filter_class_args(runner_utils.class_map[args.workflow], vars(args))
    instance = runner_utils.initialize_class(config, filtered_args, fileset)
    results = runner_utils.run_methods(instance, config["methods"])

    print(results)  # noqa: T201


if __name__ == "__main__":
    main()
