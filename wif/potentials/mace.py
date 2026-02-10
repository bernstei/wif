import logging
import subprocess

import torch

from copy import deepcopy
from pathlib import Path

import ase.io
from ase.atoms import Atoms

from mace.calculators import MACECalculator
from mace.calculators.foundations_models import download_mace_mp_checkpoint

from wfl.fit.mace import fit as mace_fit

import ase.data
try:
    from symmetrix import Symmetrix
except ImportError:
    Symmetrix = None


def calc_triple(potential_params, new_model_path=None):
    """Return a (constructor, args, kwargs) triple that wfl can use to create a
    MACE calculator

    Parameters
    ----------
    potential_params: dict
        potentials parameters
    new_model_path: Path, default None
        optional path for new model, overriding any kwargs that specify default,
        as returned by fit

    Returns
    --------
    (MACE_constructor, args, kwargs) triple for wfl calculator
    """
    kwargs = potential_params['kwargs']

    if new_model_path is not None:
        kwargs['model_paths'] = str(new_model_path)
    elif kwargs.get('model_paths') is None:
        kwargs['model_paths'] = potential_params['fit']['mace']['command_line_args']['foundation_model']

    if not Path(kwargs['model_paths']).is_file():
        kwargs['model_paths'] = download_mace_mp_checkpoint(kwargs['model_paths'])

    if kwargs.get('symmetrix'):
        if Symmetrix is None:
            raise ValueError("Got potential.kwargs.symmetrix but no Symmetrix ASE calculator")

        return (Symmetrix, [], {'model_file': kwargs['model_paths'], 'species': kwargs.get('species')})

    kwargs_use = kwargs.copy()

    if kwargs_use.get('device', None) == '_AUTO_CUDA_':
        logging.warning("MACE device set to _AUTO_CUDA_, selection will only be correct if top level "
                        "script is running on machine identical to where actual task will run")
        if torch.cuda.is_available():
            kwargs_use['device'] = 'cuda'
        else:
            del kwargs_use['device']
        logging.warning(f"MACE _AUTO_CUDA_ using MACE device '{kwargs_use.get('device')}'")

    return (MACECalculator, [], kwargs_use)


def fit(fit_configs, valid_configs, fit_command_line_args, run_dir, file_label, property_prefix, rng,
        wfl_fit_kwargs={}, init_model_path=None, autopara_info={}):
    """fit a MACE potential

    Parameters
    ----------
    fit_configs: ConfigSet
        fitting configurations (including isolated atom configs)
    valid_configs: ConfigSet
        validation configurations
    fit_command_line_args: dict
        keyword arguments passed to wfl.fit.mace in the mace_fit_params argument, to be
        added to mace_run_train command line
    run_dir: Path
        directory for all outputs
    file_label: str
        initial part of name of run directory name
    property_prefix: str
        string prefix for reference property names saved by reference calculator
    rng: np.random.Generator
        generator for making seed to make mace_run_train deterministic
    wfl_fit_kwargs: dict
        additional kwargs to pass to wfl.fit.mace()
    init_model_path: Path, default None
        optional initial model file to start from
    autopara_info: dict, optional
        dict with info for wfl AutoparaInfo object, but can only contain remote_info key

    Returns
    -------
    mace_model_path: path to final MACE model
    """
    mace_fit_params_use = deepcopy(fit_command_line_args)

    for key in ["energy", "forces", "stress"]:
        if key + "_key" in mace_fit_params_use:
            logging.warning(f"mace_fit_params already specifies {key + '_key'} = {mace_fit_params_use[key + '_key']}, not overriding")
        else:
            mace_fit_params_use[key + "_key"] = property_prefix + key

    if init_model_path is not None:
        mace_fit_params_use['foundation_model'] = str(init_model_path)

    if "seed" in mace_fit_params_use:
        logging.warning(f"mace_fit_params already specifies seed {mace_fit_params_use['seed']}, not overriding")
    else:
        mace_fit_params_use["seed"] = rng.integers(2**32)

    if mace_fit_params_use["atomic_numbers"] == "AUTO":
        atomic_numbers = set()
        for atoms in fit_configs:
            atomic_numbers |= set(atoms.numbers)
        mace_fit_params_use["atomic_numbers"] = str(sorted([int(Z) for Z in atomic_numbers]))

    if autopara_info is None or len(autopara_info) == 0:
        remote_info = None
    elif len(autopara_info) == 1 and "remote_info" in autopara_info:
        remote_info = autopara_info["remote_info"]
    else:
        # len > 1 or some other key
        raise ValueError("Only 'remote_info' is a valid key for autopara_info in opt_fit(), got {autopara_info}")

    mace_fit(fit_configs, "MACE", mace_fit_params=mace_fit_params_use,
             valid_configs=valid_configs, run_dir=run_dir / (file_label + ".MACE_fit"),
             remote_info=remote_info, **wfl_fit_kwargs)

    mace_model_path = run_dir / (file_label + ".MACE_fit") / "MACE.model"

    return mace_model_path


def remote_info_input_files(calc_triple, potential_params, autopara_info):
    """Modify remote_info to stage out potential's input file

    Parameters
    ----------
    calc_triple: tuple
        tuple that creates a potential for wfl
    potential_params: dict
        dict with params for potential
    autopara_info: dict
        dict with information on wfl autoparallelization

    Returns
    -------
    autopara_info_new: dict with information of wfl autoparallelization that stages out potential's input file
    """
    model_paths = potential_params["kwargs"].get("model_paths")

    if model_paths is None or Path(model_paths).is_absolute() or autopara_info is None or "remote_info" not in autopara_info:
        # no change
        return autopara_info

    if not isinstance(model_paths, (str, Path)):
        raise ValueError(f"can only handle a MACE model path that is a single path, got '{model_paths}'")

    autopara_info = deepcopy(autopara_info)
    if "input_files" not in autopara_info["remote_info"]:
        autopara_info["remote_info"]["input_files"] = []
    autopara_info["remote_info"]["input_files"].append(str(model_paths))

    return autopara_info


def download_cached_data(potential_params):
    with open("t.xyz", "w") as fout:
        atoms = Atoms('Si', cell=[10] * 3, pbc=[True] * 3)
        atoms.info["REF_energy"] = 0.0
        atoms.info["config_type"] = "IsolatedAtom"
        ase.io.write(fout, atoms, format="extxyz")
        atoms = Atoms('Si', cell=[2] * 3, pbc=[True] * 3)
        atoms.info["REF_energy"] = -1.0
        ase.io.write(fout, atoms, format="extxyz")

    foundation_model_arg = potential_params['fit']['mace']['command_line_args'].get('foundation_model', '')
    if len(foundation_model_arg) > 0:
        foundation_model_arg = f"--foundation_model={foundation_model_arg}"
    foundation_head_arg = potential_params['fit']['mace']['command_line_args'].get('foundation_head', '')
    if len(foundation_head_arg) > 0:
        foundation_head_arg = f"--foundation_head={foundation_head_arg}"

    cmd = ("mace_run_train --dry_run --name test "
           f"{foundation_model_arg} {foundation_head_arg} "
            "--multiheads_finetuning False --batch_size 1 --valid_batch_size 1 "
            "--train_file t.xyz --valid_file t.xyz").split()

    subprocess.run(cmd)
