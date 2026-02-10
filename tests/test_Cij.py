import pytest

import sys
import subprocess
from pathlib import Path

from tqdm import tqdm

import numpy as np

import ase.io
from ase.atoms import Atoms
from ase.calculators.lammpslib import LAMMPSlib
from ase.optimize import LBFGSLineSearch
from ase.filters import FrechetCellFilter
from ase.units import fs as u_fs

from matscipy.elasticity import fit_elastic_constants, full_3x3x3x3_to_Voigt_6x6

from ase.md.langevinbaoab import LangevinBAOAB
# from ase.md.npt import NPT
# from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

try:
    import lammps
except ImportError:
    lammps = None

def get_Cij_fluct(trajfile, fract_skip=0):
    trajfile = Path(trajfile)

    propag_trajfile = (trajfile.parent / (trajfile.stem + ".propag" + trajfile.suffix))
    if propag_trajfile.is_file():
        configs = ase.io.read(propag_trajfile, ":")[-200:]
        from matplotlib.figure import Figure
        fig = Figure()
        ax = fig.add_subplot()
        ax_c = ax.twinx()
        ax.plot(np.arange(len(configs)), [atoms.positions[0, 0] for atoms in configs], "-", c="C0", label="x")
        ax_c.plot(np.arange(len(configs)), [atoms.cell[0, 0] for atoms in configs], "-", c="C1", label="a0_x")
        fig.legend()
        fig.savefig(propag_trajfile.parent / (propag_trajfile.name + ".propag.pdf"), bbox_inches="tight")

    configs = [atoms for atoms in tqdm(ase.io.iread(trajfile))]
    configs = configs[int(len(configs) * fract_skip):]

    print("Cij_fluct traj len", len(configs))

    assert all(configs[0].numbers == 13)

    temperature_mean = np.mean([atoms.get_temperature() for atoms in configs])
    volume_mean = np.mean([atoms.get_volume() for atoms in configs])
    print("T", temperature_mean, "V", volume_mean)
    # a_traj has vectors as columns, opposite of atoms.cell
    ## print("traj len", len(configs))
    a_traj = np.asarray([atoms.cell.T for atoms in configs])
    ## print("a_traj.shape", a_traj.shape)

    ## print("a0", a_traj[0])

    # F maps from initial to current cell
    # F @ a0 = a
    # F = a(t) @ a(0)^-1
    a0_inv = np.linalg.inv(a_traj[0])
    F_traj = np.einsum('ijk,kl->ijl', a_traj, a0_inv)
    F_mean = np.mean(F_traj, axis=0)
    ## print("F_mean", F_mean)
    a0_mean = F_mean @ a_traj[0]
    ## print("a0_mean", a0_mean)
    a0_mean_inv = np.linalg.inv(a0_mean)
    # new F maps from mean to current cell
    F_traj = np.einsum('ijk,kl->ijl', a_traj, a0_mean_inv)
    # e = (F^T F - 1) / 2
    strain_traj = (np.einsum('ikj,ikl->ijl', F_traj, F_traj) - np.eye(3)) / 2.0
    ## print("strain mean", np.mean(strain_traj, axis=0))
    ## print("strain sqrt(var)", np.sqrt(np.var(strain_traj, axis=0)))

    strain_strain_mean = np.einsum('nij,nkl->ijkl', strain_traj, strain_traj) / strain_traj.shape[0]
    ## print("strain_strain_mean", strain_strain_mean)
    ## print("strain_strain_mean 9x9", strain_strain_mean.reshape((9,9)))
    strain_strain_mean_inv = np.linalg.pinv(strain_strain_mean.reshape((9,9)))
    Cij_fluct = (ase.units.kB * temperature_mean / volume_mean) * strain_strain_mean_inv # eV/A^3
    Cij_fluct = full_3x3x3x3_to_Voigt_6x6(Cij_fluct.reshape((3,3,3,3)))

    return Cij_fluct


@pytest.mark.skipif(lammps is None, reason="No lammps module imported")
@pytest.mark.slow
def test_Cij(tmp_path, monkeypatch):
    print("BOB tmp_path", tmp_path, "monkeypatch", monkeypatch)
    monkeypatch.chdir(tmp_path)

    with open(Path(__file__).parent / "assets" / "test_Cij" / "lammps.in") as fin, open("lammps.in", "w") as fout:
        for line in fin:
            fout.write(line)

    atoms = Atoms('Al' * 4, cell = [4] * 3, scaled_positions=[[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]], pbc=[True] * 3)

    atoms_sc = atoms * (3, 3, 3)
    atoms_sc.cell[2] += atoms.cell[0]
    atoms_sc.wrap()
    atoms = atoms_sc

    ####################################################################################################
    # FD
    calc = LAMMPSlib(lmpcmds=["pair_style    morse/smooth/linear 7.7895", "pair_coeff    * * 0.07 6.0 2.885 7.7895"], atom_types={"Al": 1})
    atoms.calc = calc
    print("initial volume", atoms.get_volume())

    at_cell = FrechetCellFilter(atoms)
    opt = LBFGSLineSearch(at_cell)
    opt.run(fmax=1e-6, steps=100)
    print("relaxed volume", atoms.get_volume())

    def print_Cij(Cij, label):
        with np.printoptions(precision=3, suppress=True):
            print(f"Cij {label} ", end="")
            print(f"\nCij {label} ".join([line for line in str(Cij).splitlines()]))
        sys.stdout.flush()

    Cij_FD, _ = fit_elastic_constants(atoms, verbose=False)
    print_Cij(Cij_FD, "FD")

    ####################################################################################################
    # fluctutations
    T = 10

    nsteps = 1000000

    # LAMMPS

    if not Path(f"traj.{T}.NPT.dump").is_file():
        ase.io.write("Al_108.data", atoms, format="lammps-data")

        subprocess.run(["lmp", "-var", "T", f"{T}", "-var", "nsteps", f"{nsteps}", "-in", "lammps.in"])

    Cij_lammps_NPT = get_Cij_fluct(f"traj.{T}.NPT.dump")
    print_Cij(Cij_lammps_NPT, "LAMMPS NPT fluct")

##     Cij_lammps_NVT_PRESS_LANG = get_Cij_fluct(f"traj.{T}.NVT_PRESS_LANG.dump")
##     print_Cij(Cij_lammps_NVT_PRESS_LANG, "LAMMPS NVT_PRESS_LANG fluct")

    # ASE LangevinBAOAB

    if not Path(f"traj.{T}.ASE_LangevinBAOAB.traj").is_file():

        rng = np.random.default_rng(10)
        dyn = LangevinBAOAB(atoms, timestep=2.5 * u_fs, temperature_K=T, externalstress=0.0, hydrostatic=False,
                            T_tau=200 * u_fs, P_tau=10000 * u_fs,
                            trajectory=f"traj.{T}.ASE_LangevinBAOAB.propag.traj", logfile="-", loginterval=1, rng=rng)
        dyn.run(1000)

        dyn = LangevinBAOAB(atoms, timestep=2.5 * u_fs, temperature_K=T, externalstress=0.0, hydrostatic=False,
                            T_tau=200 * u_fs, P_tau=10000 * u_fs,
                            logfile="-", loginterval=100, rng=rng)
        dyn.run(nsteps // 2)
        dyn = LangevinBAOAB(atoms, timestep=2.5 * u_fs, temperature_K=T, externalstress=0.0, hydrostatic=False,
                            T_tau=200 * u_fs, P_tau=10000 * u_fs,
                            trajectory=f"traj.{T}.ASE_LangevinBAOAB.traj", logfile="-", loginterval=100, rng=rng)
        dyn.run(nsteps // 2)

    Cij_ASE_LangevinBAOAB = get_Cij_fluct(f"traj.{T}.ASE_LangevinBAOAB.traj")
    print_Cij(Cij_ASE_LangevinBAOAB, "ASE LangevinBAOAB fluct")

##     # ASE NPT
## 
##     if not Path(f"traj.{T}.ASE_NPT.traj").is_file():
##         atoms_use = atoms.copy()
##         atoms_use.set_cell(atoms_use.cell.standard_form()[0], True)
##         atoms_use.calc = calc
## 
##         MaxwellBoltzmannDistribution(atoms_use, temperature_K=T * 2)
##         Stationary(atoms_use)
## 
##         rng = np.random.default_rng(10)
##         dyn = NPT(atoms_use, timestep=2.5 * u_fs, temperature_K=T, externalstress=0.0,
##                   ttime=200 * u_fs, pfactor=50,
##                   trajectory=f"traj.{T}.ASE_NPT.propag.traj", logfile=None, loginterval=1)
##         dyn.run(1000)
## 
##         dyn = NPT(atoms_use, timestep=2.5 * u_fs, temperature_K=T, externalstress=0.0,
##                   ttime=200 * u_fs, pfactor=50,
##                   logfile="-", loginterval=100)
##         dyn.run(nsteps // 2)
##         dyn = NPT(atoms_use, timestep=2.5 * u_fs, temperature_K=T, externalstress=0.0,
##                   ttime=200 * u_fs, pfactor=50,
##                   trajectory=f"traj.{T}.ASE_NPT.traj", logfile="-", loginterval=100)
##         dyn.run(nsteps // 2)
## 
##     Cij_ASE_NPT = get_Cij_fluct(f"traj.{T}.ASE_NPT.traj")
##     print_Cij(Cij_ASE_NPT, "ASE NPT fluct")

    assert np.allclose(Cij_FD, Cij_lammps_NPT, atol=0.2, rtol=0.4)
    assert np.allclose(Cij_FD, Cij_ASE_LangevinBAOAB, atol=0.2, rtol=0.25)
