import sys
import os
import logging
import json
from pathlib import Path
from glob import glob

import numpy as np

from tqdm import tqdm

from wfl.configset import ConfigSet, OutputSpec
from wfl.generate.md import md
from wfl.calculators.generic import calculate

from wif.utils import get_pot_mod
from wif.utils import fraction_REF_success, calc_isolated_atoms
from wif.utils import log_fit_errors, split_cumulative
from wif.utils import get_logger_interval
from wif.utils import WIFDebugAbort


def _md_traj_skip_first_config(atoms):
    return atoms.info.get('MD_time_fs', 1.0) > 0


def _gather_prev_configs(fit_params):
    """Gather previous fit, validation, and isolated atoms configs

    Parameters
    ----------
    fit_params: dict
        fitting parameters

    Returns
    -------
    prev_fit_configs, prev_valid_configs, prev_isolated_configs: ConfigSet with previously
        generated fitting validation, and isolated atoms configs
    """

    def _str_to_list(val):
        if isinstance(val, str):
            return [val]
        else:
            return val

    # get previous fit configs
    prev_fit_configs = []
    prev_valid_configs = []
    prev_isolated_configs = []
    for d_glob in _str_to_list(fit_params['prev_run_dirs']):
        # grab everything from previous run dirs
        for prev_dir in sorted(glob(d_glob)):
            prev_fit_configs.extend(sorted(Path(prev_dir).glob("*.step_020.fitting.extxyz")))
            prev_valid_configs.extend(sorted(Path(prev_dir).glob("*.step_020.validation.extxyz")))
            prev_isolated_configs.extend(sorted(Path(prev_dir).glob("isolated_atom.*.extxyz")))
    if len(fit_params['prev_run_dirs']) > 0 and (len(prev_fit_configs) == 0 or
                                                 len(prev_valid_configs) == 0):
        raise RuntimeError(f"Got prev_run_dirs '{fit_params['prev_run_dirs']}' but missing "
                           f"fitting ({len(prev_fit_configs)}) or validation ({len(prev_valid_configs)}) files")
    # grab individual fitting, validation, and isolated atoms configs
    for f_glob in _str_to_list(fit_params['prev_fitting_configs']):
        prev_fit_configs.extend([Path(f) for f in sorted(glob(f_glob))])
    for f_glob in _str_to_list(fit_params['prev_validation_configs']):
        prev_valid_configs.extend([Path(f) for f in sorted(glob(f_glob))])
    for f_glob in _str_to_list(fit_params['prev_isolated_atoms_configs']):
        prev_isolated_configs.extend([Path(f) for f in sorted(glob(f_glob))])

    prev_fit_configs = ConfigSet(prev_fit_configs)
    prev_valid_configs = ConfigSet(prev_valid_configs)
    prev_isolated_configs = ConfigSet(prev_isolated_configs)

    return prev_fit_configs, prev_valid_configs, prev_isolated_configs


def iterative_md_fit(configs, dft_calc, dft_manual_calc,
                     main_params, potential_params, data_params, fit_params, md_params,
                     rng, output_dir=Path(), output_prefix="",
                     wfl_params=None, download_cached_data=False):
    """Do an iterative MD - gather - fit cycle

    Parameters
    ----------
    configs: ConfigSet
        configurations to start MD trajectories from
    dft_calc: wfl.wfl_fileio_calculator.WFLFileIOCalculator
        wfl-wrapped DFT calculator
    potential_params: dict
        params controlling creation of potential
    main_params: dict
        params controlling main aspects e.g. manual_dft
    data_params: dict
        params controlling gathering of fitting configs
    fit_params: dict
        params controlling fitting of potential
    md_params: dict
        params controlling MD trajectory
    rng: np.random.Generator
        random number generator to be used for any randomization, e.g. fitting code seed
    output_dir: Path, optional
        path to put all output in
    output_prefix: str, optional
        string to prefix to every output file
    wfl_params: dict, optional
        params controlling wfl autoparallelization
    download_cached_data: bool, default False
        download cached data needed for fit rather than doing real run
    """
    # in-memory
    configs = ConfigSet(list(configs))

    # store index of source
    for atoms_i, atoms in enumerate(configs):
        atoms.info["wif_source_config_i"] = atoms_i

    # import functions for this particular type of potential
    pot_mod = get_pot_mod(potential_params)

    if download_cached_data:
        if hasattr(pot_mod, "download_cached_data"):
            logging.info("Downloading cached data...")
            pot_mod.download_cached_data(potential_params)
            logging.info("Done downloading cached data, exiting")
            sys.exit(0)
        else:
            logging.error("No download_cached_data function defined for potential {potential_params['type']}")
            sys.exit(1)

    md_calc = pot_mod.calc_triple(potential_params, new_model_path=md_params["initial_model"])

    # pot_path may be passed to fit as source of restart if available, so make sure variable is defined
    pot_path = None

    prev_fit_configs, prev_valid_configs, prev_isolated_configs = _gather_prev_configs(fit_params)

    configs, md_params['per_config'] = _process_per_config_params(configs, md_params['per_config'])
    # turn into nested list, so ConfigSet.groups() works as expected
    configs = ConfigSet([[atoms] for atoms in configs])

    # determine MD traj length of each batch
    config_interval, n_steps_per_batch = _calc_interval_and_n_steps(len(list(configs)), data_params['n_configs_per_fit'],
                                                data_params['config_interval'], data_params['fit_interval'],
                                                fit_params['validation_fraction'])
    if config_interval % md_params['traj_config_interval'] != 0:
        raise ValueError(f"Got final config_interval {config_interval} not a multiple of "
                         f"md.traj_config_interval {md_params['traj_config_interval']}")

    # format of label for output files
    n_step_digits = int(np.floor(np.log10(max(1, md_params['n_steps'])))) + 1
    n_stage_digits = int(np.floor(np.log10(max(1, md_params['n_steps'] // n_steps_per_batch)))) + 1

    orig_config_types = [atoms.info.get('config_type', '') for atoms in configs]

    # start sampling
    cur_step = 0
    cur_stage = 0
    while cur_step < md_params['n_steps']:
        stage_n_steps = min(n_steps_per_batch, md_params['n_steps'] - cur_step)

        logging.info(f"Starting MD to gather configs, cur_step={cur_step}, n_steps={stage_n_steps}")
        stage_label_short = "stage_{cur_stage:0{n_stage_digits}}".format(
                                cur_stage=cur_stage, n_stage_digits=n_stage_digits)
        stage_label = "{stage_label_short}_md_step_{cur_step:0{n_step_digits}}".format(
                          stage_label_short=stage_label_short,
                          cur_step=cur_step, n_step_digits=n_step_digits)
        file_label = output_prefix + stage_label

        for config, md_params_per_config in zip(configs, md_params['per_config'], strict=True):
            _set_per_config_params(config, md_params, md_params_per_config, cur_step, stage_n_steps)

        # clean up Atoms.info['config_type'] so things like error table will look nice
        for atoms, orig_config_type in zip(configs, orig_config_types):
            atoms.info["config_type"] = (orig_config_type + (":" if len(orig_config_type) > 0 else "") +
                                         f"{stage_label_short}")

        # do sampling
        output = OutputSpec(f"{file_label}.step_000.md.extxyz", file_root=output_dir)

        logger_interval = get_logger_interval(md_params['logger_interval'], stage_n_steps, default=0.1)
        autopara_info = pot_mod.remote_info_input_files(md_calc, potential_params, wfl_params['fit_md'])
        stage_traj = md(configs, output,
                        md_calc, steps=stage_n_steps, dt=md_params['dt'],
                        integrator=md_params['integrator'],
                        temperature=None, temperature_tau=md_params['T_tau'],
                        pressure=None, pressure_tau=md_params.get('P_tau'),
                        hydrostatic=md_params['var_cell_hydrostatic'],
                        traj_step_interval=md_params['traj_config_interval'],
                        logger_interval=logger_interval,
                        rng=rng, autopara_info=autopara_info)

        if os.environ.get("WIF_DEBUG_ABORT") == "000": raise WIFDebugAbort # noqa: E701

        logging.info(f"Processing MD results, cur_step={cur_step}, n_steps={stage_n_steps}")

        # grab a config every `effective_interval` from each group (i.e. traj for each input
        # config), and also # last config from each group
        effective_interval = config_interval // md_params['traj_config_interval']
        stage_configs = []
        traj_final_configs = []
        for traj_grp in tqdm(stage_traj.groups(), total=len(list(configs)), desc="process MD traj group"):
            stage_configs.append([])
            stage_configs_grp = stage_configs[-1]
            for atoms_i, atoms in enumerate(traj_grp):
                if atoms_i % effective_interval == 0:
                    stage_configs_grp.append(atoms)
            _ = stage_configs_grp.pop(0)
            traj_final_configs.append(atoms)
        stage_configs = ConfigSet(stage_configs)
        traj_final_configs = ConfigSet(traj_final_configs)

        logging.info(f"Evaluating stage configs with DFT, cur_step={cur_step}, n_steps={stage_n_steps}")
        # prepare calculator for this stage
        dft_calc._wfl_workdir = output_dir / f"{file_label}.step_010.VASP_runs"

        # DFT evaluate sampled configs
        output = OutputSpec(f"{file_label}.step_010.DFT.extxyz", file_root=output_dir)
        if main_params['manual_dft']:
            eval_configs = dft_manual_calc(stage_configs, output, dft_calc, output_prefix="REF_")
        else:
            eval_configs = calculate(stage_configs, output, dft_calc, output_prefix="REF_", autopara_info=wfl_params['dft_eval'])

        if os.environ.get("WIF_DEBUG_ABORT") == "010": raise WIFDebugAbort # noqa: E701

        # check for large forces and enough successful evaluations
        large_forces = [at_i for at_i, at in enumerate(eval_configs) if
                        (not at.info.get("REF_calculation_failed", False) and
                         at.info.get("REF_converged", False) and
                         "REF_forces" in at.arrays and
                         np.max(np.linalg.norm(at.arrays["REF_forces"], axis=1) > data_params['max_force_warning']))]
        if len(large_forces) > 0:
            logging.warning("Got configurations with forces exceeding {data_params['max_force_warning']} {large_forces}")

        frac = fraction_REF_success(eval_configs)
        if frac < data_params['min_fraction_DFT_succeeded']:
            raise RuntimeError(f"Tried {len(list(eval_configs))} DFT evaluations, only "
                               f"{frac:.2f} < {data_params['min_fraction_DFT_succeeded']} suceeded "
                               "(REF_energy present and REF_converged True)")

        # separate output_prefix and stage_label needed here so it can find cumulative, i.e. all stages, set of files
        fit_configs, valid_configs = split_cumulative(eval_configs,
                                                      output_dir, output_prefix, f"{stage_label}.step_020",
                                                      fit_params['validation_fraction'], rng)

        if os.environ.get("WIF_DEBUG_ABORT") == "020": raise WIFDebugAbort # noqa: E701
        
        potential_params_fit = potential_params['fit'][potential_params['type']]
        # should we actually be allowing no isolated_atom_mark?
        isolated_atom_mark = potential_params_fit.get('isolated_atom_mark', {})

        logging.info(f"Evaluating isolated atoms with DFT, cur_step={cur_step}, n_steps={stage_n_steps}")

        # DFT evaluate isolated atoms (if needed)
        dft_calc._wfl_workdir = output_dir / f"{file_label}.step_025.isolated_atom_VASP_runs"
        isolated_atoms = calc_isolated_atoms(prev_fit_configs + fit_configs, prev_isolated_configs,
                                             output_dir, output_prefix, f"{stage_label}.step_025",
                                             dft_calc, fit_params['isolated_atoms']['box_size'],
                                             isolated_atom_mark,
                                             force_spin_pol=fit_params['isolated_atoms']['force_spin_pol'],
                                             autopara_info=wfl_params['dft_isolated_atoms'])
        # check for successful calculation and present mark
        frac_isolated_success = fraction_REF_success(isolated_atoms)
        if frac_isolated_success < 1.0:
            n_isolated = len(list(isolated_atoms))
            raise RuntimeError("Not all isolated_atoms DFT evaluations succeeded (REF_energy present and REF_converged true)"
                               f"{np.round(frac_isolated_success * n_isolated).astype(int)} / {n_isolated}")
        for k, v in isolated_atom_mark.items():
            if any([atoms.info.get(k) != v for atoms in isolated_atoms]):
                raise RuntimeError(f"Not all isolated_atoms info dicts contain isolated_atom_mark {k}={v}")

        logging.info(f"Starting fit, cur_step={cur_step}, n_steps={stage_n_steps}")

        # do fit
        pot_path = pot_mod.fit(prev_fit_configs + fit_configs + isolated_atoms,
                               prev_valid_configs + valid_configs,
                               fit_command_line_args=potential_params_fit['command_line_args'],
                               run_dir=output_dir, file_label=f"{file_label}.step_030",
                               property_prefix="REF_", rng=rng,
                               wfl_fit_kwargs=potential_params_fit.get('wfl_fit_kwargs', {}),
                               autopara_info=wfl_params['fit'])
                               ## NOT CLEAR REFITTING FROM PREV IS ACTUALLY A GOOD IDEA
                               ## init_model_path=pot_path)

        if os.environ.get("WIF_DEBUG_ABORT") == "030": raise WIFDebugAbort # noqa: E701

        md_calc = pot_mod.calc_triple(potential_params, new_model_path=pot_path)
        error_calc = pot_mod.calc_triple(potential_params, new_model_path=pot_path)

        # calculate error tables
        log_fit_errors("fitting errors:", prev_fit_configs + fit_configs, error_calc,
                       output_dir, f"{file_label}.step_040_fitting",
                       autopara_info=wfl_params['error_calc'])
        log_fit_errors("validation errors:", prev_valid_configs + valid_configs, error_calc,
                       output_dir, f"{file_label}.step_041_validation",
                       autopara_info=wfl_params['error_calc'])

        cur_step += n_steps_per_batch
        cur_stage += 1

        # update configs for next stage
        configs = traj_final_configs


def _process_per_config_params(configs, params):
    """prepare per-config quantities such as T, T_end and P

    Parameters
    ----------
    configs: ConfigSet
        input configurations
    params: dict
        parameters controlling per-config quantities of md run
    """

    all_combinations = params.pop("all_combinations")
    apply_to_each_config = params.pop("apply_to_each_config")

    subsection_names = [k for k in params.keys() if isinstance(params[k], dict)]
    try:
        subsection_names = [int(k) for k in subsection_names]
    except TypeError as exc:
        raise ValueError(f'per_config subsection names must be integers, got {subsection_names}') from exc

    if len(subsection_names) > 0:
        # subsections
        # ignore all_combinations
        # default + section specific
        default_params = {k: v for k, v in params.items() if not isinstance(v, dict)}
        params_new = []
        for subsection in sorted(subsection_names):
            params_new.append(default_params.copy())
            params_new[-1].update(params[str(subsection)])
    else:
        # no subsections

        # promote to lists
        for k, v in params.items():
            try:
                if isinstance(v, str):
                    raise TypeError
                _ = len(v)
            except TypeError:
                params[k] = [v]

        # check/fix T vs. T_end consistency
        if len(params['T']) != len(params['T_end']):
            if len(params['T']) == 1:
                params['T'] *= len(params['T_end'])
            elif len(params['T_end']) == 1:
                params['T_end'] *= len(params['T'])
            else:
                raise RuntimeError(f"Got inconsistent T and T_end lengths {len(params['T'])} != {len(params['T_end'])}")

        array_keys = ['T', 'P']
        if all_combinations:
            # make all combinations and turn into list of md_params['per_configs'] dicts
            arrays = [list(range(len(params[k]))) for k in array_keys]
            indices_array = np.array(np.meshgrid(*arrays)).T.reshape(-1, len(arrays))
        else:
            # assemble into list of md_params['per_config']
            # check lengths
            array_lengths = np.asarray([len(params[k]) for k in array_keys])
            if np.any(array_lengths != array_lengths[0]):
                raise RuntimeError(f"Got mismatched array lengths {list(zip(array_keys, array_lengths))}")
            indices_array = [[i] * len(array_keys) for i in range(array_lengths[0])]

        # create list of sections containing scalar values by selecting using correct indices from original lists
        params_new = []
        for indices in indices_array:
            params_new.append({})
            for (k, ind) in zip(array_keys, indices):
                params_new[-1][k] = params[k][ind]

    # combine T and T_end into tuple
    for p in params_new:
        p['T'] = (p.get('T'), p.pop('T_end', None))
        if p['T'][1] is None:
            p['T'] = p['T'][0]

    params = params_new

    if apply_to_each_config:
        # duplicate configs, one per param
        n_configs_orig = len(list(configs))
        configs = [atoms.copy() for atoms in configs for _ in range(len(params))]
        params *= n_configs_orig

    n_configs_use = len(list(configs))
    if len(params) != n_configs_use:
        raise ValueError(f"Got mismatched number of per-config param settings {len(params)} != number of (possibly duplicated) configs {n_configs_use}")

    return configs, params


def _set_per_config_params(config, md_params_global, md_params_per_config, cur_step, stage_n_steps):
    """Set wfl per-config md kwargs from keys `T`, `P`, and `var_cell_hydrostatic` via `atoms.info['WFL_MD_KWARGS']`
    fields `temperature`, `pressure`, and 'hydrostatic', respecively.
    """
    per_config_allowed_keys = ['T', 'P', 'var_cell_hydrostatic']
    if any([k not in per_config_allowed_keys for k in md_params_per_config.keys()]):
        raise ValueError(f"Got a per-config md section with unknown keys {list(md_params_per_config.keys())}, "
                         f"allowed {per_config_allowed_keys}")

    wfl_md_kwargs = {}
    try:
        # ramp - set to just this stage's portion
        # could perhaps refactor and use functions from inside wfl
        T_slope = (md_params_per_config['T'][1] - md_params_per_config['T'][0]) / md_params_global['n_steps']
        Ti = md_params_per_config['T'][0] + cur_step * T_slope
        Tf = md_params_per_config['T'][0] + (cur_step + stage_n_steps) * T_slope
        wfl_md_kwargs['temperature'] = (Ti, Tf)
    except TypeError:
        # constant or None, just copy
        wfl_md_kwargs['temperature'] = md_params_per_config['T']

    wfl_md_kwargs['pressure'] = md_params_per_config['P']

    if 'var_cell_hydrostatic' in md_params_per_config:
        wfl_md_kwargs['hydrostatic'] = md_params_per_config['var_cell_hydrostatic']

    config.info['WFL_MD_KWARGS'] = json.dumps(wfl_md_kwargs)


def _calc_interval_and_n_steps(n_traj, n_configs_per_fit, config_interval, fit_interval, validation_fraction):
    """Figure out n_steps_per_batch so that enough configs are generated to satisfy n_configs_per_fit, and also 1
    example of each class given validation_fraction. From this, figure out config_interval if not provided.

    Parameters
    ----------
    n_traj: int
        number of trajectories to be carried out
    n_configs_per_fit: int
        number of configs to gather (across all trajectories) for each fit
    config_interval: int or None
        interval (in MD steps) between selected configs. Must be compatible with fit_interval
    fit_interval: int or None
        interval (in MD steps) between fits. Must be compatible with config_interval
    validation_fraction: float
        fraction of selected configs reserved for validation

    Returns
    -------
    config_interval: int interval in MD steps between selected configs
    n_steps_per_batch: int number of steps to run between each fit
    """
    # initial number per traj (round up)
    n_configs_per_traj_per_fit = int(np.ceil(n_configs_per_fit / n_traj))
    # number needed for fitting-validation split
    n_needed_per_traj = max(int(np.ceil(1.0 / validation_fraction)), 2)
    # check if enough for stratified validation, fix and warn if not
    if n_configs_per_traj_per_fit < n_needed_per_traj:
        n_configs_per_traj_per_fit = n_needed_per_traj
        logging.warning("Not enough configs to have one of each class in validation "
                        "and one of each class in fitting, increasing n_configs_per_fit "
                        f"from {n_configs_per_fit} to {n_configs_per_traj_per_fit * n_traj}")

    # find config interval for a given fit_interval
    if config_interval is None:
        if fit_interval is None:
            raise ValueError("Need at least one of data.config_interval and data.fit_interval")
        if fit_interval % n_configs_per_traj_per_fit != 0:
            raise ValueError(f"Got fit_interval {fit_interval} not a multiple of "
                             f"n_configs_per_traj_per_fit {n_configs_per_traj_per_fit}")
        config_interval = fit_interval // n_configs_per_traj_per_fit
        logging.warning(f"Set config_interval {config_interval} based on fit_interval {fit_interval} "
                        f"// n_configs_per_traj_per_fit {n_configs_per_traj_per_fit}")

    # set actual fit_interval
    if fit_interval is None:
        fit_interval = config_interval * n_configs_per_traj_per_fit
        logging.warning(f"Got effective fit_interval {fit_interval}")

    # check that config_interval and fit_interval are compatible, e.g. if they were both specified manually
    if config_interval * n_configs_per_traj_per_fit != fit_interval:
        raise RuntimeError("Got incompatible config_interval * n_configs_per_traj_per_fit "
                           f"{config_interval} * {n_configs_per_traj_per_fit} != fit_interval {fit_interval}")

    return config_interval, config_interval * n_configs_per_traj_per_fit
