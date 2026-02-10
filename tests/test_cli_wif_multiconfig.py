from pathlib import Path
import shutil
import json

import pytest

import numpy as np

from wfl.configset import ConfigSet

from wif.cli.wif import main
from wif.utils import WIFDebugAbort


def test_cli_wif_multiconfig(tmp_path, monkeypatch, vasp_exec):
    do_test_cli_wif_multiconfig(tmp_path, monkeypatch, vasp_exec)


def do_test_cli_wif_multiconfig(tmp_path, monkeypatch, vasp_exec, toml_suffix=""):
    monkeypatch.chdir(tmp_path)

    inputs_dir = tmp_path / "input_files"
    shutil.copytree(Path(__file__).parent / "assets" / "test_cli_wif_multiconfig", inputs_dir)

    with open(inputs_dir / ("multiconfig.wif.toml.template" + toml_suffix)) as fin, open(inputs_dir / "wif.toml", "w") as fout:
        for line in fin:
            fout.write(line.replace("_INPUTS_DIR_", str(inputs_dir)))

    args = ['--seed', '0', '--input_dir', str(inputs_dir), vasp_exec]
    print("testing with args", args)
    main(args)

    fit_dirs = list(tmp_path.glob("stage_*_md_step_*.step_030.MACE_fit"))
    # 0
    assert len(fit_dirs) == 1, "wrong number of fit dirs"
    for fit_dir in fit_dirs:
        assert (fit_dir / "MACE.model").is_file(), f"no model file found {fit_dir}"

    # 8 traj: 2 configs x 2 temperatures x 2 pressures
    combined_traj = [list() for _ in range(8)]
    for traj_file in sorted(tmp_path.glob("stage_*_md_step_*.step_000.md.extxyz")):
        for traj_i, traj in enumerate(ConfigSet(traj_file).groups()):
            combined_traj[traj_i].append(list(traj))

    for traj_seq in combined_traj:
        for i in range(1, len(traj_seq)):
            assert np.all(traj_seq[i][0].positions == traj_seq[i-1][-1].positions), "traj continuity position mismatch"
            assert np.all(traj_seq[i][0].cell == traj_seq[i-1][-1].cell), "traj continuity cell mismatch"


def test_cli_wif_multiconfig_toml_sections(tmp_path, monkeypatch, vasp_exec):
    monkeypatch.chdir(tmp_path)

    inputs_dir = tmp_path / "input_files"
    shutil.copytree(Path(__file__).parent / "assets" / "test_cli_wif_multiconfig", inputs_dir)

    with open(inputs_dir / ("multiconfig_toml_sections.wif.toml.template")) as fin, open(inputs_dir / "wif.toml", "w") as fout:
        for line in fin:
            fout.write(line.replace("_INPUTS_DIR_", str(inputs_dir)))

    args = ['--seed', '0', '--input_dir', str(inputs_dir), vasp_exec]
    print("testing with args", args)
    monkeypatch.setenv("WIF_DEBUG_ABORT", "000")
    with pytest.raises(WIFDebugAbort):
        main(args)

    fit_dirs = list(tmp_path.glob("stage_*_md_step_*.step_030.MACE_fit"))
    # 0
    assert len(fit_dirs) == 0, "wrong number of fit dirs"

    # 8 traj: 2 configs x 2 temperatures x 2 pressures
    for traj_file in sorted(tmp_path.glob("stage_*_md_step_*.step_000.md.extxyz")):
        for traj_i, traj in enumerate(ConfigSet(traj_file).groups()):
            traj = list(traj)
            kw = json.loads(traj[0].info["WFL_MD_KWARGS"])
            try:
                T0 = kw['temperature'][0]
            except TypeError:
                T0 = kw['temperature']
            assert np.allclose(T0, traj[0].get_temperature(), rtol=0.01)
            if kw['pressure'] is None:
                assert np.all(traj[0].cell == traj[-1].cell)
            else:
                assert not np.all(traj[0].cell == traj[-1].cell)


def test_cli_wif_single_per_config(tmp_path, monkeypatch, vasp_exec):
    monkeypatch.chdir(tmp_path)

    inputs_dir = tmp_path / "input_files"
    shutil.copytree(Path(__file__).parent / "assets" / "test_cli_wif_multiconfig", inputs_dir)

    with open(inputs_dir / "single_config.wif.toml.template") as fin, open(inputs_dir / "wif.toml", "w") as fout:
        for line in fin:
            fout.write(line.replace("_INPUTS_DIR_", str(inputs_dir)))

    args = ['--seed', '0', '--input_dir', str(inputs_dir), vasp_exec]
    print("testing with args", args)
    main(args)

    fit_dirs = list(tmp_path.glob("stage_*_md_step_*.step_030.MACE_fit"))
    # 0
    assert len(fit_dirs) == 1, "wrong number of fit dirs"
    for fit_dir in fit_dirs:
        assert (fit_dir / "MACE.model").is_file(), f"no model file found {fit_dir}"

    # 4 traj: 1 configs x 2 temperatures x 2 pressures
    combined_traj = [list() for _ in range(4)]
    for traj_file in sorted(tmp_path.glob("stage_*_md_step_*.step_000.md.extxyz")):
        for traj_i, traj in enumerate(ConfigSet(traj_file).groups()):
            combined_traj[traj_i].append(list(traj))

    for traj_seq in combined_traj:
        for i in range(1, len(traj_seq)):
            assert np.all(traj_seq[i][0].positions == traj_seq[i-1][-1].positions), "traj continuity position mismatch"
            assert np.all(traj_seq[i][0].cell == traj_seq[i-1][-1].cell), "traj continuity cell mismatch"
