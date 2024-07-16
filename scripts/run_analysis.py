from __future__ import annotations

from egamma_tnp.utils import runner_utils


def main():
    parser = runner_utils.get_main_parser()
    args = parser.parse_args()

    config = runner_utils.load_json(args.config)
    if args.binning:
        runner_utils.set_binning(runner_utils.load_json(args.binning))
    fileset = runner_utils.load_json(args.fileset)

    instance = runner_utils.initialize_class(config, args, fileset)
    results = runner_utils.run_methods(instance, config["methods"])

    print(results)  # noqa: T201


if __name__ == "__main__":
    main()
