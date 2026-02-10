from pathlib import Path
import os
import shutil
import subprocess
import re

import pytest

import numpy as np

import ase.io

from wif.cli.wif import main
from wif.dft.vasp import VASPManualRuns
from wif.utils import WIFDebugAbort

def test_cli_wif(tmp_path, monkeypatch, vasp_exec):
    monkeypatch.chdir(tmp_path)

    inputs_dir = tmp_path / "input_files"
    shutil.copytree(Path(__file__).parent / "assets" / "test_cli_wif", inputs_dir)

    ####################################################################################################
    # run with normal (wfl evaluated) DFT

    print("WFL DFT RUN")

    with open(inputs_dir / "wif.toml.template") as fin, open(inputs_dir / "wif.toml", "w") as fout:
        for line in fin:
            fout.write(line.replace("_INPUTS_DIR_", str(inputs_dir)))

    args = ['--seed', '0', '--input_dir', str(inputs_dir), vasp_exec]
    print("testing with args", args)
    main(args)

    fit_dirs = list(tmp_path.glob("stage_*_md_step_*.step_030.MACE_fit"))
    # 0, 20, and 40
    assert len(fit_dirs) == 2, f"WFL number of fit dirs {len(fit_dirs)} != 2"
    for fit_dir in fit_dirs:
        assert (fit_dir / "MACE.model").is_file(), f"WFL model {fit_dir / 'MACE_model'} is missing"

    ####################################################################################################
    # again with manual DFT runs, in same test so they can be compared
    print("MANUAL DFT RUN")

    with open(inputs_dir / "wif_manual.toml.template") as fin, open(inputs_dir / "wif.toml", "w") as fout:
        for line in fin:
            fout.write(line.replace("_INPUTS_DIR_", str(inputs_dir)))

    # run DFT evalus of steps 00, 24, each followed by remaining workflow
    for stage, md_step in [('0', '00'), ('1', '24')]:
        with pytest.raises(VASPManualRuns):
            main(args)

        for vasp_run_dir in Path(f"manual.stage_{stage}_md_step_{md_step}.step_010.VASP_runs").glob("run_VASP_*"):
            os.chdir(vasp_run_dir)
            subprocess.run(vasp_exec.split())
            os.chdir(tmp_path)

    main(args)

    fit_dirs = list(tmp_path.glob("manual.stage_*_md_step_*.step_030.MACE_fit"))
    # 0, 20, and 40
    assert len(fit_dirs) == 2, f"MANUAL number of fit dirs {len(fit_dirs)} != 2"
    for fit_dir in fit_dirs:
        assert (fit_dir / "MACE.model").is_file(), f"MANUAL model {fit_dir / 'MACE_model'} is missing"

    # check some output
    # agreement will not be exact because of things like the wfl-wrapped calculator doing things
    # that should have no effect, like changing # the order of cell vectors to ensure triple product
    # is positive, but in fact do change the result at the level of roundoff
    for f_default, f_manual in zip(sorted(list(tmp_path.glob("stage_*_md_step_*.step_040_fitting.err_MLIP_calc.extxyz"))),
                                sorted(list(tmp_path.glob("manual.stage_*_md_step_*.step_040_fitting.err_MLIP_calc.extxyz")))):
        for a_default, a_manual in zip(ase.io.iread(f_default), ase.io.iread(f_manual)):
            assert np.allclose(a_default.positions, a_manual.positions), f"WFL-DEFAULT positions mismatch {f_default} {f_manual}"
            assert np.allclose(a_default.cell, a_manual.cell), f"WFL-DEFAULT cell mismatch {f_default} {f_manual}"


def test_cli_wif_var_cell(tmp_path, monkeypatch, vasp_exec):
    # check that variable cell runs: ISIF = 8, requires switching to [md] integrator = "LangevinBAOAB"
    monkeypatch.chdir(tmp_path)

    inputs_dir = tmp_path / "input_files"
    shutil.copytree(Path(__file__).parent / "assets" / "test_cli_wif", inputs_dir)

    ####################################################################################################
    # run with normal (wfl evaluated) DFT

    print("WFL DFT RUN")

    # [md]
    # n_steps 36
    # [data]
    # n_configs_per_fit = 4
    # config_interval = 6
    fit_interval = 32
    with open(inputs_dir / "wif.toml.template") as fin, open(inputs_dir / "wif.toml", "w") as fout:
        for line in fin:
            # replace or insert before line(s)
            if re.search(r'^\s*config_interval\b', line):
                line = "config_interval = \"_NONE_\"\n"
                fout.write(f"fit_interval = {fit_interval}\n")

            # write line
            fout.write(line.replace("_INPUTS_DIR_", str(inputs_dir)))

            # insert after lines(s)
            if '[md]' in line:
                fout.write("integrator = \"LangevinBAOAB\"\n")

    with open(inputs_dir / "INCAR") as fin:
        incar_lines = fin.readlines()
    with open(inputs_dir / "INCAR", "w") as fout:
        fout.write("ISIF = 3\n")
        fout.write("".join([line for line in incar_lines if "ISIF" not in line]))
        # fout.write("".join(incar_lines))

    args = ['--seed', '0', '--input_dir', str(inputs_dir), vasp_exec]
    print("testing with args", args)

    monkeypatch.setenv("WIF_DEBUG_ABORT", "000")
    with pytest.raises(WIFDebugAbort):
        main(args)

    # traj len should be fit_interval / traj_config_interval + 1
    assert len(ase.io.read(tmp_path / "stage_0_md_step_00.step_000.md.extxyz", ":")) == fit_interval // 2 + 1
