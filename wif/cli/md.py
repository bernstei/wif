from argparse import ArgumentParser
from pathlib import Path

import numpy as np

import ase.io
from ase.units import fs as units_fs
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.langevinbaoab import LangevinBAOAB

from .utils import add_util_args

from wif.dft.vasp import read_inputs
from wif.utils import get_pot_mod
from wif.utils import gather_params, log_params
from wif.logging import setup_logging, reset_logging
from wif.utils import get_logger_interval

def md(args):
    """actually run MD with interface that looks like VASP

    Parameters
    ----------
    args: Namespace
        namespace with command line arguments, e.g. as returned by ArgumentParser
    """

    rng = np.random.default_rng(args.seed)

    input_dir = Path(args.input_dir)
    # create vasp calculator INCAR and read md params
    _, incar_md_params, _, prelim_inputs_warnings = read_inputs(input_dir, incar_only=True)

    # read params from toml files, with md params from INCAR taking lower
    # precedence than run toml file
    main_params, potential_params, _, _, md_params, _, prelim_params_warnings = gather_params(input_dir,
        pre_params={'md': incar_md_params})

    # clean up output dir and prefix
    output_prefix = main_params['output_prefix']
    if len(output_prefix) > 0 and output_prefix[-1] not in (".", "_"):
        output_prefix += "."
    output_dir = Path(main_params['output_dir'])

    setup_logging(output_dir, output_prefix, "md", prelim_inputs_warnings + prelim_params_warnings)

    log_params("args:", args)
    log_params("main_params:", main_params)
    log_params("potential_params:", potential_params)
    log_params("md_params:", md_params)

    # import functions for this particular type of potential
    pot_mod = get_pot_mod(potential_params)

    md_calc = pot_mod.calc_triple(potential_params, new_model_path=args.potential_file)
    md_calc = md_calc[0](*md_calc[1], **md_calc[2])

    # set up temperature T
    if md_params['per_config']['T_end'] is not None:
        T_slope = (md_params['per_config']['T_end'] - md_params['per_config']['T']) / md_params['n_steps']
    else:
        T_slope = 0.0

    traj_file = output_dir / (output_prefix + "md_traj.extxyz")
    if args.continuation:
        if not traj_file.is_file():
            raise RuntimeError(f"Got continuation but no existing traj_file '{traj_file}'")
        config = ase.io.read(traj_file, -1)
        if "MD_step" not in config.info or "MD_time_fs" not in config.info:
            raise RuntimeError(f"got continuation=True but missing 'MD_step' or 'MS_time_fs' properties in final config of traj file {traj_file}")
        first_step = config.info["MD_step"]

        traj_file_mode = "a"

        dyn_kwargs = {'nsteps': first_step}
    else:
        first_step = 0

        if main_params['input_configs'] is None:
            main_params['input_configs'] = "POSCAR"
        config_file = input_dir / main_params['input_configs']
        if not config_file.is_file():
            raise ValueError(f"Input file '{config_file}' does not exist")
        config = ase.io.read(config_file, index=-1)
        if "MD_step" in config.info:
            del config.info["MD_step"]
        if "MD_time_fs" in config.info:
            del config.info["MD_time_fs"]

        if traj_file.is_file():
            raise RuntimeError(f"Found existing traj_file '{traj_file}' but got continuation=False. Refusing to overwrite.")
        traj_file_mode = "w"

        dyn_kwargs = {}

    traj_file_fmt = ase.io.formats.filetype(traj_file.name, read=traj_file.is_file())

    config.calc = md_calc
    with open(traj_file, traj_file_mode) as traj_file_obj:
        if first_step == 0:
            config.info["MD_step"] = 0
            config.info["MD_time_fs"] = 0.0

        externalstress = md_params['per_config'].get('P', None)

        if md_params['per_config']['T'] is None:
            cur_T = None
        else:
            cur_T = md_params['per_config']['T'] + first_step * T_slope

        # initialize T if not a continuation, T is specified, but no velocities are set
        if not args.continuation and cur_T is not None and config.get_temperature() == 0.0:
            MaxwellBoltzmannDistribution(config, temperature_K=cur_T, force_temp=True, rng=rng)
            Stationary(config, preserve_temperature=True)

        if md_params['T_tau'] is None:
            # T_tau None indicates no thermostat
            cur_T = None

        logger_interval = get_logger_interval(md_params['logger_interval'], md_params['n_steps'])
        dyn = LangevinBAOAB(config, timestep=md_params['dt'] * units_fs,
                temperature_K=cur_T, externalstress=externalstress,
                T_tau=md_params['T_tau'], P_tau=md_params['P_tau'],
                hydrostatic=md_params['var_cell_hydrostatic'],
                P_mass=md_params['P_mass'], P_mass_factor=md_params['P_mass_factor'],
                rng=rng, logfile="-", loginterval=logger_interval, **dyn_kwargs)

        def save_config():
            if first_step != 0:
                # continuation
                if dyn.nsteps == 0:
                    return
            config.info["MD_time_fs"] = dyn.nsteps * dyn.dt / units_fs
            config.info["MD_step"] = dyn.nsteps
            ase.io.write(traj_file_obj, config, format=traj_file_fmt)
            traj_file_obj.flush()

            if cur_T is not None and T_slope != 0.0:
                dyn.set_temperature(md_params['per_config']['T'] + dyn.nsteps * T_slope)

        dyn.attach(save_config, interval=md_params['traj_config_interval'])

        dyn.run(md_params['n_steps'])

    reset_logging()


def main(*cli_args):
    """main program interface for iterative fitting of a MLIP with interface that looks like VASP
    """
    parser = ArgumentParser()
    parser.add_argument("--seed", "-s", type=int, help="random seed", required=True)
    parser.add_argument("--input_dir", "-i", help="input directory", default=".")
    parser.add_argument("--potential_file", "-p", help="potential input file", required=True)
    parser.add_argument("--continuation", "-c", action="store_true", help="continuation of interrupted run")
    add_util_args(parser, sections=["main", "potential", "md"])

    md(parser.parse_args(*cli_args))


if __name__ == "__main__":
    main()
