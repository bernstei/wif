import re
import shutil

from pathlib import Path

from wif.cli.wif import main

def test_interrupted_rerun(tmp_path, monkeypatch, vasp_exec):
    monkeypatch.chdir(tmp_path)

    inputs_dir = tmp_path / "input_files"
    shutil.copytree(Path(__file__).parent / "assets" / "test_cli_wif", inputs_dir)

    with open(inputs_dir / "wif.toml.template") as fin, open(inputs_dir / "wif.toml", "w") as fout:
        for line in fin:
            fout.write(line.replace("_INPUTS_DIR_", str(inputs_dir)))

    args = ['--seed', '0', '--input_dir', str(inputs_dir), vasp_exec]

    print("testing with args", args)
    main(args)

    with open("wif_vasp.00.log") as fin:
        orig_lines = [re.sub(r"^\s*\d+-\d+-\d+\s+\d+:\d+:\d+,\d+ - ", "", line) for line in fin if ("fitting errors" in line or "validation errors" in line)]

    # remove final stage
    Path("wif_vasp.00.log").rename("wif_vasp.00.log.orig")
    for f in Path().glob("stage_*_md_step_40.*"):
        if f.is_dir():
            shutil.rmtree(f)
        else:
            f.unlink()

    print("rerunning test with args", args)
    main(args)

    with open("wif_vasp.00.log") as fin:
        new_lines = [re.sub(r"^\s*\d+-\d+-\d+\s+\d+:\d+:\d+,\d+ - ", "", line) for line in fin if ("fitting errors" in line or "validation errors" in line)]

    assert orig_lines == new_lines, "error table mismatch"
