from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from wfl.configset import ConfigSet

from .utils import add_util_args

from wif.utils import gather_params, log_params
from wif.logging import setup_logging, reset_logging
from wif.standalone_fit import standalone_fit

def fit(args):
    """actually run fitting of a MLIP from previous runs of iterative fit

    Parameters
    ----------
    args: Namespace
        namespace with command line arguments, e.g. as returned by ArgumentParser
    """
    rng = np.random.default_rng(args.seed)

    input_dir = Path(args.input_dir)

    _, potential_params, _, fit_params, _, wfl_params, prelim_params_warnings = gather_params(input_dir)

    # clean up output dir and prefix
    output_prefix = args.potential_label
    if len(output_prefix) > 0 and output_prefix[-1] not in (".", "_"):
        output_prefix += "."
    output_dir = Path(args.output_dir)

    setup_logging(output_dir, output_prefix, "fit", prelim_params_warnings)

    # log all run paramaters
    log_params("args:", args)
    log_params("potential_params:", potential_params)
    log_params("fit_params:", fit_params)
    log_params("wfl_params:", wfl_params)

    fit_configs = ConfigSet(args.fitting_configs)
    valid_configs = ConfigSet(args.validation_configs)
    isolated_atoms = ConfigSet(args.isolated_atom_configs)

    standalone_fit(args.potential_label, fit_configs, valid_configs, isolated_atoms,
                   potential_params, fit_params, rng,
                   output_dir=output_dir, output_prefix=output_prefix,
                   wfl_params=wfl_params, download_cached_data=args.download_cached_data)

    reset_logging()


def main(*cli_args):
    """main program interface for fit from previous runs of iterative fitting of a MLIP
    """
    parser = ArgumentParser()
    parser.add_argument("--seed", "-s", type=int, help="random seed", required=True)
    parser.add_argument("--input_dir", "-i", help="input directory", default=".")
    parser.add_argument("--output_dir", "-o", help="output directory", default=".")
    parser.add_argument("--potential_label", "-p", help="label for potential", default="combined_fit")
    parser.add_argument("--fitting_configs", "-f", nargs="+", help="fitting config globs", required=True)
    parser.add_argument("--validation_configs", "-v", nargs="+", help="validation config globs", required=True)
    parser.add_argument("--isolated_atom_configs", "-a", nargs="+", help="isolated atom config globs", required=True)
    parser.add_argument("--download_cached_data", "-c", action="store_true",
                        help="Only download cached info that might be needed for fit and then exit")
    add_util_args(parser, sections=["potential", "fit"])

    fit(parser.parse_args(*cli_args))


if __name__ == "__main__":
    main()
