import sys
import logging
from pathlib import Path

from wfl.configset import OutputSpec

from wif.utils import get_pot_mod
from wif.utils import log_fit_errors

def standalone_fit(potential_label, fit_configs, valid_configs, isolated_atoms,
                   potential_params, fit_params, rng, 
                   output_dir=Path(), output_prefix="",
                   wfl_params=None, download_cached_data=False):
    """actually run fitting of a MLIP from previous runs of iterative fit

    Parameters
    ----------
    potential_label: str
        label for fit potential file
    fit_configs: ConfigSet
        configurations to fit
    valid_configs: ConfigSet
        configurations for validation
    isolated_atoms: ConfigSet
        isolated atom reference energy configuration
    potential_params: dict
        parameters controlling creation of potential
    fit_params: dict
        parameters controlling potential fitting
    rng: np.random.Generator
        random number generator to be used for any randomization, e.g. fitting code seed
    output_dir: Path, optional
        path to put all output in
    output_prefix: str, optional
        string to prefix to every output file
    wfl_params: dict, optional
        dict controling wfl autoparallelization
    """

    pot_mod = get_pot_mod(potential_params)

    if download_cached_data:
        if hasattr(pot_mod, "download_cached_data"):
            pot_mod.download_cached_data(potential_params)
            logging.info("Done downloading cached data, exiting")
            sys.exit(0)
        else:
            logging.error("No 'download_cached_data' function defined for potential {potential_params['type']}")
            sys.exit(1)

    # make unique set of isolated atoms configs
    isolated_atoms_unique = {}
    n_isolated_atoms = 0
    for at in isolated_atoms:
        n_isolated_atoms += 1

        if len(at.numbers) != 1:
            raise ValueError(f"Got more than one atom {at.symbols} in isolated_atoms '{isolated_atoms}'")

        sym = at.symbols[0]
        E = at.info["REF_energy"]
        if sym in isolated_atoms_unique:
            E_prev = isolated_atoms_unique[sym].info["REF_energy"]
            if E != E_prev:
                raise ValueError(f"Got conflicting isolated atom energy values {E_prev} != {E}  for {at.symbols} in isolated_atoms '{isolated_atoms}'")
        isolated_atoms_unique[sym] = at

    potential_params_fit = potential_params['fit'][potential_params['type']]

    # rewrite to apply mark and combine ConfigSets
    combined_isolated_atoms_file = output_dir / (output_prefix + "isolated_atoms.combined.extxyz")
    logging.warning(f"Rewriting isolated atoms into a single file '{combined_isolated_atoms_file}' containing only unique instances")
    os = OutputSpec(combined_isolated_atoms_file)
    if not os.all_written():
        for at in isolated_atoms_unique.values():
            at.info.update(potential_params_fit.get('isolated_atom_mark', {}))
            os.store(at)
        os.close()
    isolated_atoms_unique = os.to_ConfigSet()

    pot_path = pot_mod.fit(fit_configs + isolated_atoms_unique, valid_configs,
                           fit_command_line_args=potential_params_fit.get('command_line_args', {}),
                           run_dir=output_dir, file_label=potential_label,
                           property_prefix="REF_", rng=rng,
                           wfl_fit_kwargs=potential_params_fit.get('wfl_fit_kwargs', {}),
                           autopara_info=wfl_params['fit']) ##mace_run_train bug , init_model_path=pot_path)

    calc = pot_mod.calc_triple(potential_params, new_model_path=pot_path)

    log_fit_errors("fitting errors:", fit_configs, calc, output_dir, output_prefix + potential_label + '.fitting', autopara_info=wfl_params['error_calc'])
    log_fit_errors("validation errors:", valid_configs, calc, output_dir, output_prefix + potential_label + '.validation', autopara_info=wfl_params['error_calc'])
