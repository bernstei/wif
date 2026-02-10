from pathlib import Path
from wif.cli.fit import main


def test_cli_fit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    inputs_dir = Path(__file__).parent / "assets" / "test_cli_fit"

    args = ['--seed', '0', '--input_dir', str(inputs_dir),
            "--fitting_configs", str(inputs_dir / "*.fitting.extxyz"),
            "--validation_configs", str(inputs_dir / "*.validation.extxyz"),
            "--isolated_atom_configs", str(inputs_dir / "isolated_atom.*.extxyz")]

    main(args)

    assert (tmp_path / "combined_fit.MACE_fit" / "MACE.model").is_file(), "no model file found"

    with open(tmp_path / "combined_fit.wif_fit.00.log") as fin:
        for line in fin:
            if 'fitting errors:' in line and '_ALL_' in line:
                n_energies = int(line.strip().split()[7])
                assert n_energies == 5
            elif 'validation errors:' in line and '_ALL_' in line:
                n_energies = int(line.strip().split()[7])
                assert n_energies == 4
