from pathlib import Path
import shutil

import pytest

import numpy as np

import ase.io
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from wif.cli.md import main


def test_cli_md(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    inputs_dir = Path(__file__).parent / 'assets' / 'test_cli_md'

    args = ['--seed', '0', '--input_dir', str(inputs_dir), '--potential_file', 'small']

    main(args)

    nsteps = 2500

    assert len(ase.io.read(tmp_path / 'md_traj.extxyz', ':')) == nsteps + 1, f'number of steps != {nsteps}'
    Ts = [atoms.get_temperature() for atoms in ase.io.iread(tmp_path / 'md_traj.extxyz')]

    init_T = np.mean(Ts[500:750])
    final_T = np.mean(Ts[-250:])
    coeffs = np.polyfit(np.arange(len(Ts) - 250), Ts[250:], deg=1)

    plot = False
    if plot:
        from matplotlib.figure import Figure
        fig = Figure()
        ax = fig.add_subplot()
        ax.plot(range(len(Ts)), Ts, '-')
        ax.plot([500, 750], [init_T, init_T], '--')
        ax.plot([len(Ts)-250, len(Ts)-1], [final_T, final_T], '--')
        ax.plot(np.arange(len(Ts)), coeffs[0] + np.arange(len(Ts)) * coeffs[1], '-', label='fit')
        ax.set_xlabel('timestep')
        ax.set_xlabel('T (K)')
        fig.savefig('test_cli_md.T.pdf', bbox_inches='tight')

    expected_slope = (1000 - 300) / len(Ts)
    print("CHECK slope", coeffs[0], "expected", expected_slope)
    assert coeffs[0] > expected_slope / 2 and coeffs[0] < expected_slope * 2

    n_log_lines = 0
    with open("wif_md.00.log") as fin:
        for line in fin:
            if n_log_lines > 0:
                n_log_lines += 1
            if " 0.0000 " in line:
                n_log_lines = 1
    # 2500, every 10, including both first and last
    assert n_log_lines == (nsteps // 10) + 1


def test_cli_md_initial_vel(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    inputs_dir = Path(__file__).parent / 'assets' / 'test_cli_md'
    shutil.rmtree(tmp_path / 'inputs', ignore_errors=True)
    shutil.copytree(inputs_dir, tmp_path / 'inputs')
    (tmp_path / 'inputs' / 'INCAR').rename(tmp_path / 'inputs' / 'INCAR.orig')
    (tmp_path / 'inputs' / 'POSCAR').rename(tmp_path / 'inputs' / 'orig.POSCAR')

    atoms = ase.io.read(tmp_path / 'inputs' / 'orig.POSCAR')
    MaxwellBoltzmannDistribution(atoms, temperature_K=400, force_temp=True, rng=np.random.default_rng(5))
    ase.io.write(tmp_path / 'inputs' / 'start.extxyz', atoms)

    args = ['--seed', '0', '--input_dir', str(tmp_path / 'inputs'), '--potential_file', 'small']

    def _set_output_dir(output_dir):
        Path(tmp_path / output_dir).mkdir()
        with open(tmp_path / 'inputs' / 'wif.toml', 'w') as fout:
            fout.write('\n[main]\n')
            fout.write(f'output_dir = "{output_dir}"\n')
            fout.write('input_configs = "start.extxyz"\n')

    #######
    output_dir = 'initial_vel_overrride_NVE'
    _set_output_dir(output_dir)

    with open(tmp_path / 'inputs' / 'INCAR', 'w') as fout:
        fout.write('IBRION = 0\n')
        fout.write('POTIM = 2.0\n')
        fout.write('NSW = 10\n')
        fout.write('TEBEG = 300\n')
        fout.write('TEEND = 1000\n')

    main(args)
    configs = ase.io.read(tmp_path / output_dir / 'md_traj.extxyz', ':')
    # initial 400
    assert np.allclose(configs[0].get_temperature(), 400.0, rtol=1e-4)
    # drops because not constant T
    assert configs[-1].get_temperature() < 260.0

    #######
    output_dir = 'initial_vel_overrride_NVT'
    _set_output_dir(output_dir)

    with open(tmp_path / 'inputs' / 'INCAR', 'w') as fout:
        fout.write('MDALGO = 1\n')
        fout.write('POTIM = 2.0\n')
        fout.write('NSW = 1000\n')
        fout.write('TEBEG = 300\n')
        fout.write('TEEND = 1000\n')
        fout.write('ANDERSEN_PROB = 0.1\n')

    main(args)
    configs = ase.io.read(tmp_path / output_dir / 'md_traj.extxyz', ':')
    # initial 400
    assert np.allclose(configs[0].get_temperature(), 400.0, rtol=1e-4)
    # increases because of constant T ramp
    assert np.mean([config.get_temperature() for config in configs[-100:]]) > 540.0


def test_cli_md_INCAR_NVE_NVT(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    inputs_dir = Path(__file__).parent / 'assets' / 'test_cli_md'
    shutil.rmtree(tmp_path / 'inputs', ignore_errors=True)
    shutil.copytree(inputs_dir, tmp_path / 'inputs')
    (tmp_path / 'inputs' / 'INCAR').rename(tmp_path / 'inputs' / 'INCAR.orig')

    args = ['--seed', '0', '--input_dir', str(tmp_path / 'inputs'), '--potential_file', 'small']

    def _set_output_dir(output_dir):
        Path(tmp_path / output_dir).mkdir()
        with open(tmp_path / 'inputs' / 'wif.toml', 'w') as fout:
            fout.write(f'\n[main]\n  output_dir = "{output_dir}"\n')

    ###############################################################
    # const E (no SMASS) v 1, set initial T
    output_dir = 'const_E_initial_T_v1'
    _set_output_dir(output_dir)

    with open(tmp_path / 'inputs' / 'INCAR', 'w') as fout:
        fout.write('IBRION = 0\n')
        fout.write('POTIM = 2.0\n')
        fout.write('NSW = 10\n')
        fout.write('TEBEG = 300\n')
        fout.write('TEEND = 1000\n')

    main(args)
    configs = ase.io.read(tmp_path / output_dir / 'md_traj.extxyz', ':')
    # initial 300
    assert np.allclose(configs[0].get_temperature(), 300.0, rtol=1e-4)
    # drops because not constant T
    assert configs[-1].get_temperature() < 160.0

    # no Langevin warning for const E
    with open(tmp_path / output_dir / 'wif_md.00.log') as fin:
        assert not any(["only Langevin is implemented" in line for line in fin])

    ###############################################################
    # const E (SMASS = -3) v 2, set initial T
    output_dir = 'const_E_initial_T_v2'
    _set_output_dir(output_dir)

    with open(tmp_path / 'inputs' / 'INCAR', 'w') as fout:
        fout.write('IBRION = 0\n')
        fout.write('POTIM = 2.0\n')
        fout.write('NSW = 10\n')
        fout.write('TEBEG = 300\n')
        fout.write('TEEND = 1000\n')
        fout.write('SMASS = -3\n')

    main(args)
    configs = ase.io.read(tmp_path / output_dir / 'md_traj.extxyz', ':')
    # initial 300
    assert np.allclose(configs[0].get_temperature(), 300.0, rtol=1e-4)
    # drops because not constant T
    assert configs[-1].get_temperature() < 160.0

    # no Langevin warning for const E
    with open(tmp_path / output_dir / 'wif_md.00.log') as fin:
        assert not any(["only Langevin is implemented" in line for line in fin])

    ###############################################################
    # const T (SMASS = 0), set initial T
    output_dir = 'const_T_NH_initial_T'
    _set_output_dir(output_dir)

    with open(tmp_path / 'inputs' / 'INCAR', 'w') as fout:
        fout.write('IBRION = 0\n')
        fout.write('POTIM = 2.0\n')
        fout.write('NSW = 10\n')
        fout.write('TEBEG = 300\n')
        fout.write('TEEND = 1000\n')
        fout.write('SMASS = 0\n')

    main(args)
    configs = ase.io.read(tmp_path / output_dir / 'md_traj.extxyz', ':')
    # initial 300
    assert np.allclose(configs[0].get_temperature(), 300.0, rtol=1e-4)
    # drops because not constant T
    assert configs[-1].get_temperature() > 200.0

    # Langevin algo warning for non-Langevin const T
    with open(tmp_path / output_dir / 'wif_md.00.log') as fin:
        assert sum(["only Langevin is implemented" in line for line in fin]) == 1

    ###############################################################
    # const T (Langevin), set initial T
    output_dir = 'const_T_Langevin_initial_T'
    _set_output_dir(output_dir)

    with open(tmp_path / 'inputs' / 'INCAR', 'w') as fout:
        fout.write('MDALGO = 3\n')
        # 1 / (gamma / 1000) = 80
        # gamma = 1000 * 1/80
        fout.write(f'LANGEVIN_GAMMA = {1000.0 / 80.0}\n')
        fout.write('POTIM = 2.0\n')
        fout.write('NSW = 10\n')
        fout.write('TEBEG = 300\n')
        fout.write('TEEND = 1000\n')

    main(args)
    configs = ase.io.read(tmp_path / output_dir / 'md_traj.extxyz', ':')
    # initial 300
    assert np.allclose(configs[0].get_temperature(), 300.0, rtol=1e-4)
    # drops because not constant T
    assert configs[-1].get_temperature() > 200.0

    # Langevin details warning for Langevin const T
    with open(tmp_path / output_dir / 'wif_md.00.log') as fin:
        assert sum(["Selected algorithm is Langevin" in line for line in fin]) == 1

    ###############################################################
    # invalid SMASS
    output_dir = 'invalid_SMASS'
    _set_output_dir(output_dir)

    with open(tmp_path / 'inputs' / 'INCAR', 'w') as fout:
        fout.write('IBRION = 0\n')
        fout.write('POTIM = 2.0\n')
        fout.write('NSW = 10\n')
        fout.write('TEBEG = 300\n')
        fout.write('TEEND = 1000\n')
        fout.write('SMASS = 1.0\n')

    with pytest.raises(ValueError):
        main(args)
