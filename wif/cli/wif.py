from argparse import ArgumentParser
from pathlib import Path

import logging

import numpy as np

from wfl.configset import ConfigSet

from .utils import add_util_args

from wif.utils import log_params, gather_params, gather_input_configs
from wif.dft.vasp import read_inputs, manual_calc
from wif.logging import setup_logging, reset_logging

from wif.iterative_md_fit import iterative_md_fit

def wif(args):
    """actually run iterative fitting of a MLIP with interface that looks like VASP

    Parameters
    ----------
    args: Namespace
        namespace with command line arguments, e.g. as returned by ArgumentParser
    """

    rng = np.random.default_rng(args.seed)

    input_dir = Path(args.input_dir)
    # create vasp calculator INCAR and read md params
    vasp_calc, incar_md_params, incar_data_params, prelim_inputs_warnings = read_inputs(input_dir)
    if vasp_calc is None:
        raise RuntimeError(f"Failed to get vasp_calc with warnings {prelim_inputs_warnings}")
    vasp_calc.command = args.vasp_exec

    # read params from toml files, with md params from INCAR taking lower
    # precedence than run toml file
    main_params, potential_params, data_params, fit_params, md_params, wfl_params, prelim_params_warnings = gather_params(input_dir,
            pre_params={'md': incar_md_params, 'data': incar_data_params})

    # clean up output dir and prefix
    output_prefix = main_params['output_prefix']
    if len(output_prefix) > 0 and output_prefix[-1] not in (".", "_"):
        output_prefix += "."
    output_dir = Path(main_params['output_dir'])

    setup_logging(output_dir, output_prefix, "vasp", prelim_inputs_warnings + prelim_params_warnings)

    # log all run parameters
    log_params("args:", args)
    log_params("main_params:", main_params)
    log_params("potential_params:", potential_params)
    log_params("data_params:", data_params)
    log_params("fit_params:", fit_params)
    log_params("md_params:", md_params)
    log_params("wfl_params:", wfl_params)

    # consistency checks, all could probably be allowed to fail later, but error
    # messages may not be as informative
    # NONE RIGHT NOW

    # default to POSCAR if it exists, otherwise "*.POSCAR"
    if main_params["input_configs"] is None:
        if (input_dir / "POSCAR").is_file():
            main_params["input_configs"] = "POSCAR"
        else:
            main_params["input_configs"] = "*.POSCAR"

    config_files = gather_input_configs(main_params['input_configs'], input_dir)
    logging.info(f"config_files: {[str(f) for f in config_files]}")
    configs = ConfigSet(config_files, file_root=input_dir)

    # start the iterative process
    iterative_md_fit(configs, vasp_calc, manual_calc,
                     main_params, potential_params, data_params, fit_params, md_params,
                     rng, output_dir=output_dir, output_prefix=output_prefix,
                     wfl_params=wfl_params, download_cached_data=args.download_cached_data)

    reset_logging()


def main(*cli_args):
    """main program interface for iterative fitting of a MLIP with interface that looks like VASP
    """
    parser = ArgumentParser()
    parser.add_argument("--seed", "-s", type=int, help="random seed", required=True)
    parser.add_argument("--input_dir", "-i", help="input directory", default=".")
    parser.add_argument("--download_cached_data", "-c", action="store_true",
                        help="Only download cached info that might be needed for fit and then exit")
    parser.add_argument("vasp_exec", help="vasp executable")
    add_util_args(parser, sections=["main", "potential", "data", "fit", "md"])

    wif(parser.parse_args(*cli_args))


if __name__ == "__main__":
    main()
