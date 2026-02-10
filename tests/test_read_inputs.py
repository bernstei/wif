import shutil
from pathlib import Path

import pytest

from wif.dft.vasp import read_inputs

def test_read_inputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    inputs_dir = tmp_path / "input_files"
    shutil.copytree(Path(__file__).parent / "assets" / "test_read_inputs", inputs_dir)

    shutil.copyfile(inputs_dir / "POTCAR.good", inputs_dir / "POTCAR")

    read_inputs(inputs_dir)


def test_read_inputs_POTCAR_conflict(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    inputs_dir = tmp_path / "input_files"
    shutil.copytree(Path(__file__).parent / "assets" / "test_read_inputs", inputs_dir)

    shutil.copyfile(inputs_dir / "POTCAR.bad", inputs_dir / "POTCAR")

    with pytest.raises(RuntimeError):
        read_inputs(inputs_dir)
