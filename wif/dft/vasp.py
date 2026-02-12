import os
import re
import logging
from pathlib import Path

import numpy as np

from ase.calculators import calculator
from wfl.calculators.vasp import Vasp
from wfl.utils.save_calc_results import save_calc_results


class VASPManualRuns(UserWarning):
    pass

def manual_calc(configs, output, calc, properties=None, output_prefix="REF_", ignore_failed_jobs=True):
    """Set up for manual submission of VASP runs, then exit if needed, or if they
    are all done, process outputs, like wfl.calculators.generic(..., wfl.calculators.vasp.Vasp, ...)

    Parameters
    ----------
    configs: ConfigSet
        configs to calculate
    output: OutputSpec
        where to write output
    calc: wfl.calculators.vasp.Vasp
        calculator to use
    properties: list(str), default None
        properties to calculate
    output_prefix: str, default "REF_"
        string to prefix every property key in Atoms.info/.arrays
    ignore_failed_jobs: bool, default True
        ignore failed jobs

    Returns
    -------
    output_configset ConfigSet containing output configs with calculated properties
    """
    if output.all_written():
        return output.to_ConfigSet()

    if properties is None:
        properties = ['energy', 'forces', 'stress']

    # should this be more general, like wfl.calculators.vasp.WIFVasp
    orig_VASP_PP_PATH = os.environ.get('VASP_PP_PATH')
    os.environ['VASP_PP_PATH'] = '.'

    runs_have_input = []
    runs_missing_output = []
    runs_failed_output = []

    def _check_have_input(rundir):
        return all([(rundir / filename).is_file() for filename in ["INCAR", "POSCAR", "POTCAR"]])
    def _check_missing_output(rundir):
        return not all([(rundir / filename).is_file() for filename in ["OUTCAR", "vasprun.xml"]])

    workdir = calc._wfl_workdir

    for atoms_i, atoms in enumerate(configs):
        rundir = (workdir / f"run_VASP_{atoms_i}")
        runs_have_input.append(_check_have_input(rundir))

        # set up rundir (whether or not input was already present, so calc is ready for reading)
        atoms.calc = calc
        rundir.mkdir(parents=True, exist_ok=True)
        atoms.calc.directory = rundir
        atoms.calc.initialize(atoms)
        atoms.calc.write_input(atoms, properties, system_changes=tuple(calculator.all_changes))

        # try to read output
        runs_missing_output.append(_check_missing_output(rundir))
        if runs_missing_output[-1]:
            continue

        # job appears to have run, try to read output
        runs_failed_output.append(True)
        try:
            atoms.calc.update_atoms(atoms)
            atoms.calc.read_results()
            save_calc_results(atoms, prefix=output_prefix, properties=properties)
            # WARNING: must set output.store(.., loc), otherwise resulting configs are grouped the
            # same as wfl.calculators.generic, and then cumul_split doesn't have the same stratify,
            # and resulting split is different
            output.store(atoms, atoms.info.get("_ConfigSet_loc", None))
            runs_failed_output[-1] = False
        except Exception as exc:
            logging.warning(f"WARNING: Got exception while reading VASP output {exc}")

    # check if we need to run any jobs because outputs are missing
    if any(runs_missing_output):
        rundirs_glob = f"{workdir}/vasp_run_*"
        if all(runs_missing_output):
            # all outputs are missing, input situation could be clean or dirty
            if not any(runs_have_input):
                # normal state: this was a clean dir before this run
                raise VASPManualRuns(f"Run all vasp jobs in '{rundirs_glob}' and then rerun script")
            elif all(runs_have_input):
                # all runs where already present, perhaps user forgot to run VASP and reran workflow anyway
                err_msg = "No outputs but all jobs were already set up.  Did you rerun workflow before running VASP in each " + \
                          f"dir '{rundirs_glob}' ?"
            else:
                err_msg = "No outputs, but only some inputs were set up {np.where(runs_have_input)[0]}.  Refusing to continue, " + \
                          f"clean up '{rundirs_glob}' and rerun."
        else:
            # some but not all outputs missing
            if not all(runs_have_input):
                err_msg = "Some outputs are missing, and so are some inputs. Clean up '{rundirs_glob}' and rerun."
            else:
                err_msg = f"Some but not all outputs are missing from '{workdir}/vasp_run_<I>' where " + \
                          f"I = {np.where(runs_missing_output)[0]}. Run those."
        raise RuntimeError(err_msg)

    # have all outputs, try to read them
    if any(runs_failed_output):
        # no need to set up runs, but some runs failed
        if ignore_failed_jobs:
            logging.warning(f"WARNING: Failed to read output from {workdir}/vasp_run_<I> where I = {np.where(runs_failed_output)[0]}")
        else:
            raise RuntimeError(f"Failed to read output from {workdir}/vasp_run_<I> where I = {np.where(runs_failed_output)[0]}")

    # all done successfully, write output
    output.close()

    if orig_VASP_PP_PATH is not None:
        os.environ['VASP_PP_PATH'] = orig_VASP_PP_PATH

    return output.to_ConfigSet()


class WIFVasp(Vasp):
    """Wrapper of wfl Vasp class that can force and undo spin polarized
    calculations, to be used for isolated atoms
    """
    def force_spin_pol(self):
        """force spin polarized mode
        """
        self.old_ispin = self.int_params["ispin"]
        self.old_lorbit = self.int_params["lorbit"]
        self.old_magmom = self.list_float_params['magmom']
        self.int_params["ispin"] = 2
        self.int_params["lorbit"] = 10
        # will use atoms.set_initial_magnetic_moments()
        self.list_float_params['magmom'] = None


    def restore_spin_pol(self):
        """restore spin polarized mode to what it was before forcing
        """
        self.int_params["ispin"] = self.old_ispin
        self.int_params["lorbit"] = self.old_lorbit
        self.int_params["magmom"] = self.old_magmom


    def is_noncollinear(self):
        return self.bool_params['lnoncollinear'] or self.bool_params['lsorbit']


def read_inputs(inputs_dir=Path(), incar_only=False):
    """read VASP input files (INCAR, POTCAR, KPOINTS) and parse for parameters
    that create Vasp calculator and set (some) MD run parameters

    Parameters
    ----------
    inputs_dir: Path
        path to search for files

    Returns
    -------
    calc: WIFVasp (derived from wfl.calculators.vasp.Vasp) calculator
    md_params: dict of MD params set in INCAR
    data_params: dict of data params set in INCAR
    warnings: list(str) warning messages, to be outputted later when logging is set up
    """
    warnings = []

    calc = WIFVasp()

    # must be read before INCAR for some reason
    if not incar_only:
        if (inputs_dir / "KPOINTS").is_file():
            calc.read_kpoints(inputs_dir / "KPOINTS")

    try:
        calc.read_incar(inputs_dir / "INCAR")
    except FileNotFoundError:
        warnings.append("WARNING: No INCAR found, skipping VASP inputs")
        return None, {}, {}, warnings

    # special tags that support multiple values
    with open(inputs_dir / "INCAR") as fin:
        for incar_line in fin:
            incar_line = incar_line.strip()
            for incar_tag, calc_dict in [("TEBEG", calc.float_params),
                                         ("TEEND", calc.float_params),
                                         ("PSTRESS", calc.float_params)]:
                if re.search(r"^" + incar_tag + r"\s*=", incar_line):
                    m = re.search(r"^" + incar_tag + r"\s*=([^#]+)", incar_line)
                    if m:
                        tag_values = m.group(1).strip().split()
                        if len(tag_values) == 1:
                            calc_dict[incar_tag.lower()] = float(tag_values[0]) if tag_values[0] != "_NONE_" else tag_values[0]
                        else:
                            calc_dict[incar_tag.lower()] = [float(t) if t != "_NONE_" else t for t in tag_values]
                    else:
                        raise RuntimeError(f"Got tag {incar_tag} in line '{incar_line}' but could not parse value")

    # extract and clear any INCAR tags that do not make sense for DFT calculator

    # type of dynamics
    ibrion = calc.int_params['ibrion']
    mdalgo = calc.int_params['mdalgo']
    calc.int_params['ibrion'] = None
    calc.int_params['mdalgo'] = None

    if ibrion not in [None, 0]:
        raise ValueError(f"Unsupported IBRION = {ibrion} not None or 0")
    if mdalgo is None:
        mdalgo = 0

    potim = calc.float_params['potim']
    calc.float_params['potim'] = None
    nsw = calc.int_params['nsw']
    calc.int_params['nsw'] = None

    T_tau = None
    P_tau = None
    smass = None
    if mdalgo in [0, 2]: # constant E or N-H thermostat
        smass = calc.float_params['smass']
        if smass is None: # VASP default, constant E
            smass = -3
        if smass == 0:
            T_tau = potim * 40
        elif smass != -3:
            raise ValueError(f"Only values of 0 or -3 allowed for SMASS, got {smass}")
    elif mdalgo == 1: # Anderson thermostat
        andersen_prob = calc.float_params['andersen_prob']
        if andersen_prob is not None and andersen_prob != 0:
            T_tau = potim / andersen_prob
    elif mdalgo == 3: # Langevin thermostat
        gamma = calc.list_float_params['langevin_gamma']
        if any([g != gamma[0] for g in gamma[1:]]):
            raise ValueError(f"Only single LANGEVIN_GAMMA is supported, got {gamma}")
        T_tau = 1.0 / (gamma[0] / 1000.0)
        gamma_l = calc.float_params['langevin_gamma_l']
        if gamma_l is None:
            P_tau = None
        else:
            P_tau = 1.0 / (gamma_l / 1000.0)
    elif mdalgo == 4: # NHC thermstat
        T_tau = potim * calc.float_params['nhc_period']
    elif mdalgo == 5: # CSVR thermostat
        T_tau = potim * calc.float_params['csvr_period']
    else:
        raise ValueError(f"No valid IBRION {ibrion} or MDALGO {mdalgo}")
    # warn about different constant T algorithm
    if T_tau is not None:
        msg = f"WARNING: Got IBRION {ibrion}, SMASS {smass}, and MDALGO {mdalgo}, which select constant T. "
        if mdalgo == 3:
            msg += "Selected algorithm is Langevin, but details will differ (using BAOAB " \
                   "implementation https://doi.org/10.1063/1.4802990)." 
        else:
            msg += "Selected algorithm is not Langevin, but only Langevin is implemented. " \
                   "Will use Langevin (BAOAB https://doi.org/10.1063/1.4802990), similar to VASP MDALGO=3."
        warnings.append(msg)
    calc.float_params['smass'] = None
    calc.float_params['andersen_prob'] = None
    calc.float_params['langevin_gamma'] = None
    calc.float_params['langevin_gamma_l'] = None
    calc.float_params['nhc_period'] = None
    calc.float_params['csvr_period'] = None

    isif = calc.int_params['isif']
    if isif is None:
        isif = 0
    if isif in [4, 5, 6, 7]:
        raise ValueError(f"Unsupported ISIF = {isif} in (4, 5, 6, 7)")

    var_cell = isif in [3, 8]
    var_cell_hydrostatic = isif in [8]
    if var_cell:
        P = calc.float_params['pstress']
        calc.float_params['pstress'] = None
        if P is None:
            P = 0.0
        # kB -> GPa
        if isinstance(P, list):
            P = [pp * 0.1 for pp in P]
        else:
            P *= 0.1
    else:
        P = None

    if calc.float_params['tebeg'] is not None:
        T = calc.float_params['tebeg']
        T_end = calc.float_params['teend']
    else:
        T = None
        T_end = None
    calc.float_params['tebeg'] = None
    calc.float_params['teend'] = None

    if var_cell and T is None:
        raise ValueError(f"variable cell ISIF={isif} requires a temperature TEBEG")

    md_params = {'dt': potim, 'n_steps': nsw,
                 'T_tau': T_tau, 'P_tau': P_tau,
                 'var_cell_hydrostatic': var_cell_hydrostatic,
                 'per_config': {'T': T, 'T_end': T_end, 'P': P}
                }

    if not incar_only:
        calc.input_params['setups'] = {}
        if (inputs_dir / "POTCAR").is_file:
            calc.input_params['pp'] = str(inputs_dir / "POTCARs")
            with open(inputs_dir / "POTCAR") as fin:
                fout = None
                for potcar_line in fin:
                    if potcar_line.strip().startswith("PAW_PBE "):
                        f = potcar_line.strip().split()
                        if fout is not None:
                            fout.close()
                        potcar_dir = (inputs_dir / "POTCARs" / f[1])
                        potcar_dir.mkdir(parents=True, exist_ok=True)
                        fout = open(potcar_dir / "POTCAR", "w")
                        elem_setup = f[1].split("_", maxsplit=1)
                        elem = elem_setup[0]
                        if len(elem_setup) == 1:
                            setup = ""
                        else:
                            setup = "_" + elem_setup[1]
                        if elem in calc.input_params["setups"] and calc.input_params["setups"][elem] != setup:
                            raise RuntimeError(f"Got multiple instances of elem {elem} in POTCAR with conflicting setups "
                                               f"'{calc.input_params['setups'][elem]}' != '{setup}'")
                        calc.input_params["setups"][elem] = setup
                    fout.write(potcar_line)
            fout.close()
        else:
            warnings.append(f"WARNING: Assuming that element-specific POTCARs are in '{inputs_dir}/<element_symbol>/POTCAR'")
            calc.input_params['pp'] = str(inputs_dir)

    data_params = {}
    with open(inputs_dir / "INCAR") as fin:
        for incar_line in fin:
            incar_line = incar_line.strip()
            for int_field in ["n_configs_per_fit", "config_interval", "fit_interval"]:
                incar_tag = "WIF_" + int_field.upper()
                if re.search(r"^" + incar_tag + r"\s*=", incar_line):
                    m = re.search(r"^" + incar_tag + r"\s*=\s*([0-9]+)\s*(|;.*|#.*)$", incar_line)
                    if m:
                        data_params[int_field] = int(m.group(1))
                    else:
                        raise ValueError(r"Failed to parse integer from {l}")

    return calc, md_params, data_params, warnings
