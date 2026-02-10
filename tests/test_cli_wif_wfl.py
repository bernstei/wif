from pathlib import Path
import shutil

import pytest

from wif.cli.wif import main

@pytest.mark.expyre
def test_cli_wif_wfl(tmp_path, monkeypatch, expyre, vasp_exec):
    monkeypatch.chdir(tmp_path)

    inputs_dir = tmp_path / "input_files"
    shutil.copytree(Path(__file__).parent / "assets" / "test_cli_wif_wfl", inputs_dir)

    with open(inputs_dir / "wif.toml.template") as fin, open(inputs_dir / "wif.toml", "w") as fout:
        for line in fin:
            fout.write(line.replace("_INPUTS_DIR_", str(inputs_dir)))

    args = ['--seed', '0', '--input_dir', str(inputs_dir), vasp_exec]
    print("testing with args", args)
    main(args)

    fit_dirs = list(tmp_path.glob("stage_*_md_step_*.step_030.MACE_fit"))
    # 0, 20, and 40
    assert len(fit_dirs) == 3, "wrong numbre of fit dirs"
    for fit_dir in fit_dirs:
        assert (fit_dir / "MACE.model").is_file(), f"no model file found {fit_dir}"
