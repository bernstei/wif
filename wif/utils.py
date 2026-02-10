import importlib
import re
import logging
from copy import deepcopy
from pathlib import Path

from pprint import pformat

import toml
import json

import numpy as np

from sklearn.model_selection import train_test_split

from ase.atoms import Atoms

from wfl.configset import ConfigSet, OutputSpec
from wfl.calculators.generic import calculate
from wfl.fit.error import calc as error_calc, errors_dumps

from wif.term_symbols import term_symbols


class WIFDebugAbort(Exception):
    pass

def calc_isolated_atoms(configs, prev_isolated_atoms_configs, output_dir, output_prefix, stage_label,
                        calc, box_size, isolated_atom_mark, force_spin_pol=True, autopara_info={}):
    """do calculation of all isolated atoms

    Parameters
    ----------
    configs: iterable(Atoms)
        fitting configs to scan for all elements
    prev_isolated_atoms_configs: ConfigSet
        previously calculated isolated atoms configs
    output_dir: Path
        directory for all outputs
    output_prefix: str
        prefix for all output files
    stage_label: str
        label associated with this stage of iterative fit
    calc: Calculator
        reference value (DFT) calculator
    box_size: float
        size of box to do calculation in
    isolated_atom_mark: dict
        dict to update Atoms.info with to indicate to fitting code that this is an isolated atom
    force_spin_pol: True
        force DFT calculator into spin-polarized mode
    autopara_info: dict, optional
        dict with info for wfl AutoparaInfo

    Returns
    --------
    isolated_atoms_configs: ConfigSet with all evaluated isolated atom configs
    """
    species = sorted(set([sym for atoms in configs for sym in atoms.symbols]))

    if force_spin_pol:
        calc.force_spin_pol()

    prev_syms = [atoms.symbols[0] for atoms in prev_isolated_atoms_configs]

    new_isolated_atoms_configs = []
    for sym in species:
        if sym in prev_syms:
            continue
        atoms = Atoms(sym, cell=[box_size, box_size + 0.1, box_size + 0.2], pbc = [True] * 3)
        atoms.info.update(isolated_atom_mark)
        if calc.is_noncollinear():
            atoms.set_initial_magnetic_moments([[2 * term_symbols[atoms.numbers[0]]['S'], 0.0, 0.0]])
        else:
            atoms.set_initial_magnetic_moments([2 * term_symbols[atoms.numbers[0]]['S']])

        output = OutputSpec(output_prefix + "isolated_atom." + sym + ".extxyz", file_root=output_dir)
        atoms_eval = calculate(ConfigSet([atoms]), output, calc, output_prefix="REF_",
                               properties=["energy", "magmoms"],
                               autopara_info=autopara_info)
        new_isolated_atoms_configs.append(atoms_eval)

    if force_spin_pol:
        calc.restore_spin_pol()

    os = OutputSpec(output_prefix + stage_label + ".isolated_atoms.extxyz", file_root=output_dir)
    if not os.all_written():
        for atoms in prev_isolated_atoms_configs:
            atoms.info.update(isolated_atom_mark)
            os.store(atoms)
        for atoms in new_isolated_atoms_configs:
            os.store(atoms)
        os.close()
    return os.to_ConfigSet()


def nested_dict_update(orig_dict, update_dict, *, no_new_keys=True, debug=False):
    """update a possibly nested dict with items from another dict with
    the same nesting structure

    By default does not allow for new keys to be added, but individual subdicts
    can set `_new_keys_ok = True` to allow those dicts to accept new keys,
    or `_new_sections_ok = True` to allow new subsections (i.e. new keys that point to dicts)

    Parameters
    ----------
    orig_dict: dict
        dict to be updated
    update_dict: dict
        dict containing new values
    no_new_keys: bool, default True
        fail if new dict attempts to create new keys
    """
    for k in update_dict:
        if debug: print(debug, "nested_dict_update k", k) # noqa: E701
        if isinstance(orig_dict.get(k), dict):
            # nested dicts
            if isinstance(update_dict[k], dict):
                if debug: print(debug, "nested_dict_update descending") # noqa: E701
                nested_dict_update(orig_dict[k], update_dict[k], debug=debug + "  " if debug else debug)
            else:
                raise ValueError(f"Got key {k} in update_dict which exists with a dict value in orig_dict, "
                                 f"but in update_dict it is of type {type(update_dict[k])}")
        else:
            if k not in orig_dict and no_new_keys:
                if debug: print(debug, "nested_dict_update detected new key") # noqa: E701
                if orig_dict.get("_new_keys_ok") or (orig_dict.get("_new_sections_ok") and isinstance(update_dict[k], dict)):
                    pass
                else:
                    raise ValueError(f"Got key {k} in update_dict which does not exist in orig dict '{list(orig_dict.keys())}'")
            if debug: print(debug, "nested_dict_update copying") # noqa: E701
            orig_dict[k] = deepcopy(update_dict[k])


def toml_sentries(d):
    """convert sentry values "_NONE_" to `None` and "_DELETE_" to remove fields
    from toml file input

    Parameters
    ----------
    d: dict or list or ndarray
        dict output of toml.load to convert
    """
    if isinstance(d, dict):
        del_k = []
        for k, v in d.items():
            if isinstance(v, str):
                if v == "_NONE_":
                    d[k] = None
                elif v == "_DELETE_":
                    del_k.append(k)
            else:
                toml_sentries(v)
        for k in del_k:
            del d[k]
    elif not isinstance(d, str):
        try:
            for v_i, v in enumerate(d):
                if isinstance(v, str):
                    if v == "_NONE_":
                        d[v_i] = None
                    elif v == "_DELETE_":
                        raise ValueError("Cannot delete (sentry value '_DELETE_') from an array")
                else:
                    toml_sentries(v)
        except TypeError:
            pass


def clean_new_keys_sentry(d):
    for k, v in list(d.items()):
        if k in ["_new_keys_ok", "_new_sections_ok"]:
            del d[k]
        elif isinstance(v, dict):
            clean_new_keys_sentry(v)


def gather_params(input_dir, *, pre_params=None):
    """read wif parameters from global defaults and input directory toml files

    Parameters
    ----------
    input_dir: Path
        directory to search for "wif.toml"
    pre_params: dict, default None
        if present, dict to use for params that modify the global defaults,
        but are overwritten by what's provided in `input_dir / "wif.toml"`

    Returns
    -------
    main_params, potential_paramsl, data_params, fit_params, md_params, wfl_params: dicts containing parameters
    warnings: list(str) warning strings, to be outputted later once logging is set up
    """
    warnings = []
    ####################################################################################################
    # regular params

    # package defaults
    with open(Path(__file__).parent / "wif_defaults.toml") as fin:
        params = toml.load(fin)

    # passed into function (e.g. from INCAR)
    if pre_params is not None:
        nested_dict_update(params, pre_params) # , debug="DEBUG pre_params")

    # from specific config file
    if Path(input_dir / "wif.toml").is_file():
        with open(input_dir / "wif.toml") as fin:
            file_params = toml.load(fin)
        nested_dict_update(params, file_params) # , debug="DEBUG wif.toml")
    else:
        warnings.append(f"toml file '{input_dir / 'wif.toml'}' does not exist, continuing without overriding any defaults")

    # cleanup
    toml_sentries(params)
    clean_new_keys_sentry(params)

    ####################################################################################################
    # wfl_params

    # package defaults
    with open(Path(__file__).parent / "wif_wfl_defaults.toml") as fin:
        wfl_params = toml.load(fin)

    # from specific config file
    if Path(input_dir / "wif_wfl.toml").is_file():
        with open(input_dir / "wif_wfl.toml") as fin:
            wfl_file_params = toml.load(fin)
        nested_dict_update(wfl_params, wfl_file_params)

    # cleanup
    toml_sentries(wfl_params)
    clean_new_keys_sentry(wfl_params)

    # work around failure to pickle toml.load results, as discussed in
    # https://github.com/uiri/toml/issues/362#issuecomment-1309678944
    #
    # hope that we have no dicts with non-string keys, since json will
    # break those
    params = json.loads(json.dumps(params))
    wfl_params = json.loads(json.dumps(wfl_params))

    return params['main'], params['potential'], params['data'], params['fit'], params['md'], wfl_params, warnings


def show_default_params(sections):
    """Show content of default params file
    """
    filename = str(Path(__file__).parent / "wif_defaults.toml")
    print("# " + filename)
    print("#" * (len(filename) + 2))
    with open(Path(__file__).parent / "wif_defaults.toml") as fin:
        default_params = toml.load(fin)
    clean_new_keys_sentry(default_params)

    if sections is None:
        sections = default_params.keys()
    for k_i, k in enumerate(sections):
        v = default_params[k]
        if k_i > 0:
            print("")
        print(f"[{k}]")
        print("\n".join(["  " + re.sub(r"^(\s*)\[([^]]+)", r"\1[" + k + r".\2", toml_line) for toml_line in toml.dumps(v).splitlines()]))


def log_fit_errors(label, configs, calc, output_dir, file_label, autopara_info=None):
    """log errors as a pretty table

    Parameters
    ----------
    label: str
        label for lines
    configs: ConfigSet
        configurations with reference data, prefix "REF_"
    calc: Calculator or (Calculator_constructor, args, kwargs)
        calculator or triple to pass to wfl.calculators.generic.calculate
    output_dir: Path
        directory to run in
    file_label: str
        label for file that will contain calculated properties
    """
    os = OutputSpec(file_label + ".err_MLIP_calc.extxyz", file_root=output_dir)
    configs_pot = calculate(configs, os, calc, output_prefix="MLIP_", autopara_info=autopara_info)

    errors, _, _ = error_calc(configs_pot, calc_property_prefix="MLIP_", ref_property_prefix="REF_")
    error_lines = errors_dumps(errors)

    for error_line in error_lines.splitlines():
        logging.info(label + " " + error_line)


def log_params(label, obj):
    for msg_line in pformat(obj).splitlines():
        logging.info(label + " " + msg_line)


def fraction_REF_success(configs):
    successes = ["REF_energy" in atoms.info and atoms.info.get("REF_converged", False) for atoms in configs]
    n_configs = len(list(configs))
    n_success = sum(successes)

    return n_success / n_configs


def split_cumulative(new_configs, output_dir, output_prefix, stage_label, validation_fraction, rng):
    """train-validation split on cumulative set of configurations

    Newly split configs are saved to `output_dir / (output_prefix + stage_label + ".fitting.extxyz")`, and
    analogously for validation configs.

    Previous fitting configs are assumed match glob where `stage_label` is replaced by `*`.

    `rng` final state is guaranteed to be same whether or not fitting/validation files already
    exist and split does not have to be rerun

    Parameters
    ----------
    new_configs: ConfigSet
        configurations generated at this iteration
    output_dir: Path
        directory to run in
    output_prefix: str
        prefix for output files
    stage_label: str
        label associated with this stage of iterative fit
    validation_fraction: float
        fraction to assigne to validation set
    rng: np.random.Generator
        random number generator state to use

    Returns
    -------
    (fit_configs, validation_configs) ConfigSets with split configs
    """

    rng_use = rng.spawn(1)[0]

    # stratify by source of configs
    stratify = []
    for grp_i, grp in enumerate(new_configs.groups()):
        if isinstance(grp, Atoms):
            # not really groups, don't stratify
            stratify = None
            break
        for at in grp:
            assert isinstance(at, Atoms)
            stratify.append(grp_i)

    os_fit = OutputSpec(output_prefix + stage_label + ".fitting.extxyz", file_root=output_dir)
    os_valid = OutputSpec(output_prefix + stage_label + ".validation.extxyz", file_root=output_dir)

    if not os_fit.all_written() or not os_valid.all_written():
        new_fit, new_valid = train_test_split(list(new_configs), test_size=validation_fraction, stratify=stratify,
                                              random_state=rng_use.integers(2**32))

        for atoms in new_fit:
            os_fit.store(atoms)
        for atoms in new_valid:
            os_valid.store(atoms)

        os_fit.close()
        os_valid.close()

    return ConfigSet(output_prefix + "*.fitting.extxyz", file_root=output_dir), ConfigSet(output_prefix + "*.validation.extxyz", file_root=output_dir)


def gather_input_configs(input_configs, input_dir):
    """Gather input configs in a well-defined order.  Globs are expanded
    in order, and files in each glob are sorted alphabetically

    Parameters
    ----------
    input_configs: str / list(str)
        one or more globs to expand for files
    input_dir: Path
        leading path to interpret globs relative to

    Returns
    -------
    input_configs_files: list(Path) all input files
    """
    # gather input_configs files in well-defined order
    if isinstance(input_configs, str):
        input_configs = [input_configs]

    input_configs_files = []
    for input_configs_glob in input_configs:
        glob_files = sorted(list(input_dir.glob(input_configs_glob)))
        if len(glob_files) == 0:
            raise RuntimeError(f"Found no input_configs from {input_configs_glob}")
        input_configs_files.extend(glob_files)

    if len(input_configs_files) == 0:
        raise RuntimeError(f"Found no input_configs from {input_configs}")

    n_leading = len(input_dir.parts)
    input_configs_files = [Path().joinpath(*path.parts[n_leading:]) for path in input_configs_files]

    return input_configs_files


def get_pot_mod(potential_params):
    """get module that defines potential-specific functions:
    - calc_triple
    - fit
    - download_cached_data
    - remote_info_input_files
    """
    try:
        pot_mod = importlib.import_module(f"wif.potentials.{potential_params['type']}")
    except ImportError as exc:
        raise ValueError(f"potential type '{potential_params['type']}' unknown") from exc

    return pot_mod


def get_logger_interval(logger_interval, n_steps, default=1):
    """Get actual logger interval from value in params
    """
    if logger_interval is None:
        logger_interval = default

    if isinstance(logger_interval, (float, np.floating)):
        logger_interval = max(1, int(n_steps * logger_interval))
    elif not isinstance(logger_interval, (int, np.integer)):
        raise ValueError(f"Got logger_interval type {type(logger_interval)} not float or int")

    return logger_interval
