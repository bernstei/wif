import os
from pathlib import Path

import shutil

import pytest

@pytest.fixture()
def expyre(tmp_path, monkeypatch):
    if not str(tmp_path).startswith(str(Path.home())):
        pytest.xfail(reason='expyre tests require tmp_path be under $HOME, pass "--basetemp $HOME/<some_path>"')

    expyre_root = (tmp_path / "_expyre")
    expyre_root.mkdir()

    expyre_mod = pytest.importorskip('expyre')

    # use @ to also read user's main ~/.expyre/config.json
    monkeypatch.chdir(tmp_path)
    expyre_mod.config.init("@", verbose=True)

@pytest.fixture()
def vasp_exec():
    vasp_exec_path = os.environ.get("PYTEST_VASP_EXEC", Path(__file__).parent / "assets" / "vasp.fake")

    if not shutil.which(vasp_exec_path):
        pytest.skip(reason=f"vasp_exec='{vasp_exec_path}' not found")

    return str(vasp_exec_path)

@pytest.fixture(scope="session", autouse=True)
def cached_data(tmpdir_factory):
    tmp_path = tmpdir_factory.mktemp("global_tmp")
    print("cache in tmp_path", tmp_path)

    tmp_path.mkdir("mace")
    os.environ["XDG_CACHE_HOME"] = str(tmp_path)
    cache_source = Path(__file__).parent / "assets" / "cache_mace"
    # too big for github - rethink tests
    # shutil.copyfile(cache_source / "descriptorsnpy", tmp_path / "mace" / "descriptorsnpy")
    os.system(f"gzip -cd < {cache_source / 'mp_traj_combinedxyz.gz'} > {tmp_path / 'mace' / 'mp_traj_combinedxyz'}")

# markers
wif_markers = [("expyre", "tests of wfl including remote execution with ExPyRe"),
               ("slow", "slow tests")]

def pytest_addoption(parser):
    for marker_name, marker_desc in wif_markers:
        parser.addoption(f"--run_{marker_name}", action="store_true", default=False, help="run " + marker_desc)


def pytest_configure(config):
    for marker_name, marker_desc in wif_markers:
        config.addinivalue_line("markers", f"{marker_name}: mark {marker_desc}")


def pytest_collection_modifyitems(config, items):
    for marker_name, _ in wif_markers:
        if not config.getoption(f"--run_{marker_name}"):
            skip = pytest.mark.skip(reason=f"need --run_{marker_name} option to run")
            for item in items:
                if marker_name in item.keywords:
                    item.add_marker(skip)
