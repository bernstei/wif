import sys
import os
from pathlib import Path

import ase.units

import numpy as np
from ase.atoms import Atoms

from atrajan.n_samples_eff import std_err_block_averaging
from wif.Langevin_BAOAB import Langevin_BAOAB


logfile = "/dev/null"

n_steps = 10000
interval = 20

def get_atoms(a0):
    atoms = Atoms('Al' * 4, cell=[a0] * 3, scaled_positions=[[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]], pbc=[True] * 3)
    atoms.cell[2] += atoms.cell[0]

    return atoms

####################################################################################################
# bulk modulus
####################################################################################################

def B(atoms, calc):
    atoms.calc = calc
    V_0 = atoms.get_volume()
    E_0 = atoms.get_potential_energy()
    s_0 = atoms.get_stress()

    atoms_p = atoms.copy()
    atoms_p.calc = calc
    atoms_p.set_cell(atoms_p.cell * 1.001, True)
    V_p = atoms_p.get_volume()
    E_p = atoms_p.get_potential_energy()

    atoms_m = atoms.copy()
    atoms_m.calc = calc
    atoms_m.set_cell(atoms_m.cell * 0.999, True)
    V_m = atoms_m.get_volume()
    E_m = atoms_m.get_potential_energy()
    # print("E", E_m, E_0, E_p)

    dE_dV_p = (E_p - E_0) / (V_p - V_0)
    dE_dV_m = (E_0 - E_m) / (V_0 - V_m)
    # print("stress", (dE_dV_p + dE_dV_m) / 2 / ase.units.GPa, s_0[0:3] / ase.units.GPa)
    d2E_dV2 = (dE_dV_p - dE_dV_m) / ((V_p + V_0) / 2 - (V_0 + V_m) / 2)

    return V_0, E_0, V_0 * d2E_dV2 / ase.units.GPa

a0 = 4.0

print("CALC EMT")
from ase.calculators.emt import EMT
calc_emt = EMT()
for a0 in np.linspace(3.5, 4.5, 11):
    print(a0, "V E B", B(get_atoms(a0), calc_emt))

sys.path.insert(0, str(Path() / 'tests' / 'assets' / 'modules'))
from morse_fast import MorsePotential

print("CALC Morse epsilon=0.07")
# epsilon, r0, rho0
morse = MorsePotential(r0 = 2.04 * np.sqrt(2), epsilon = 0.07)
for a0 in np.linspace(3.5, 4.5, 11):
    print(a0, "V E B", B(get_atoms(a0), morse))

print("CALC Morse epsilon=0.07 rho0=3")
# epsilon, r0, rho0
morse_soft = MorsePotential(r0 = 3.00 * np.sqrt(2), epsilon = 0.07, rho0=3)
for a0 in np.linspace(3.5, 4.5, 11):
    print(a0, "V E B", B(get_atoms(a0), morse_soft))

print("CALC Morse epsilon=0.07 r0=4.08 * sqrt(2)")
# epsilon, r0, rho0
morse_big = MorsePotential(r0 = 2 * 2.04 * np.sqrt(2), epsilon = 0.07)
for a0 in np.linspace(7, 9, 11):
    print(a0, "V E B", B(get_atoms(a0), morse_big))

a0 = 4.0
atoms_Al = get_atoms(a0)
timestep = 1.0 * ase.units.fs

####################################################################################################
# supercell MD
####################################################################################################

def do_run(atoms_prim, sc, P_mass, T, calc=morse, ax=None, traj_file=None, label_extra=''):
    label = f'sc={sc} P_mass={P_mass} T={T}' + label_extra
    filename = Path('data_' + label.replace(' ', '_') + '.npz')

    if not filename.is_file():
        atoms = atoms_prim * sc

        atoms.calc = calc
        rng = np.random.default_rng(seed=5)

        volumes = []
        temperatures = []
        cells = []
        stresses = []
        def track_vol():
            volumes.append(atoms.get_volume())
            temperatures.append(atoms.get_temperature())
            cells.append(atoms.cell.copy())
            stresses.append(atoms.get_stress(include_ideal_gas=True))

        dyn = Langevin_BAOAB(atoms, timestep=timestep, temperature_K=T, T_tau=50 * timestep, P_tau=500 * timestep,
                             externalstress=-0.01 * ase.units.GPa, P_mass=P_mass,
                             logfile=logfile, loginterval=interval, rng=rng, trajectory=traj_file)
        dyn.attach(track_vol)
        dyn.run(n_steps)

        np.savez(filename, volumes=np.asarray(volumes), temperatures=np.asarray(temperatures),
                           cells=np.asarray(cells), stresses=np.asarray(stresses))
    
    data = np.load(filename)
    volumes = data['volumes']
    temperatures = data['temperatures']
    cells = data['cells']
    stresses = data['stresses']

    n_local_maxima = np.sum(np.logical_and(volumes[1:-1] > volumes[0:-2], volumes[1:-1] > volumes[2:]))
    period = n_steps / n_local_maxima

    if ax is not None:
        ax.plot(range(len(volumes)), volumes, '-', label=label + f' period {period:.1f}')

    return period

# prelim vary M
from matplotlib.figure import Figure
fig = Figure()
ax = fig.add_subplot()

print(f'sc 3 P_mass 1000 T 300 period ', end=''); sys.stdout.flush()
period = do_run(atoms_Al, 3, 1000.0, 300, ax=ax)
print(period)

print(f'sc 3 P_mass 10000 T 300 period ', end=''); sys.stdout.flush()
period = do_run(atoms_Al, 3, 10000.0, 300, ax=ax)
print(period)

ax.set_xlabel('time step')
ax.set_ylabel('volume')
ax.legend()
fig.savefig('t_3_P_mass_VARY_T_300.pdf')

####################################################################################################
# vary T
print("VARY T")
from matplotlib.figure import Figure
fig = Figure()
ax = fig.add_subplot()

sc = 3
P_mass = 10000.0
# T
for T in [50, 100, 200, 400, 600, 700]:
    print(f'sc {sc} P_mass {P_mass} T {T} period ', end=''); sys.stdout.flush()
    period = do_run(atoms_Al, sc, P_mass, T, ax=ax, traj_file=f"vary_T_{T}.traj")
    print(period)

ax.set_xlabel('time step')
ax.set_ylabel('volume')
ax.legend()
fig.savefig(f't_sc_{sc}_P_mass_{P_mass}_T_VARY.pdf')

####################################################################################################
# vary B
print("VARY B")
from matplotlib.figure import Figure
fig = Figure()
ax = fig.add_subplot()

sc = 3
P_mass = 10000.0
T = 200

print(f'sc {sc} P_mass {P_mass} T {T} calc morse period ', end=''); sys.stdout.flush()
period = do_run(atoms_Al, sc, P_mass, T, calc=morse, label_extra=' calc morse', ax=ax, traj_file=f"vary_calc_morse.traj")
print(period)
print(f'sc {sc} P_mass {P_mass} T {T} calc morse_soft period ', end=''); sys.stdout.flush()
period = do_run(atoms_Al, sc, P_mass, T, calc=morse_soft, label_extra=' calc morse_soft', ax=ax, traj_file=f"vary_calc_morse_soft.traj")
print(period)

ax.set_xlabel('time step')
ax.set_ylabel('volume')
ax.legend()
fig.savefig(f't_sc_{sc}_P_mass_{P_mass}_T_200_calc_VARY.pdf')

####################################################################################################
# vary N
print("VARY N")
from matplotlib.figure import Figure
fig = Figure()
ax = fig.add_subplot()

# sc
P_mass = 10000.0
T = 400
for sc in [2, 3, 4]:
    print(f'sc {sc} P_mass {P_mass} T {T} period ', end=''); sys.stdout.flush()
    period = do_run(atoms_Al, sc, P_mass, T, calc=morse, label_extra=' calc morse', ax=ax, traj_file=f"vary_sc_{sc}.traj")
    print(period)

ax.set_xlabel('time step')
ax.set_ylabel('volume')
ax.legend()
fig.savefig(f't_sc_VARY_P_mass_{P_mass}_T_{T}.pdf')

####################################################################################################
# vary L
print("VARY L")
from matplotlib.figure import Figure
fig = Figure()
ax = fig.add_subplot()

# sc
P_mass = 10000.0
T = 400
sc = 3

L = 1
print(f'sc {sc} P_mass {P_mass} T {T} L {L} period ', end=''); sys.stdout.flush()
period = do_run(atoms_Al, sc, P_mass, T, calc=morse, label_extra=f' L {L}', ax=ax, traj_file=f"vary_L.traj")
print(period)

L = 2
print(f'sc {sc} P_mass {P_mass} T {T} L {L} period ', end=''); sys.stdout.flush()
atoms_use = atoms_Al.copy()
atoms_use.set_cell(atoms_use.cell * 2, True)
period = do_run(atoms_use, sc, P_mass, T, calc=morse_big, label_extra=f' L {L}', ax=ax, traj_file=f"vary_L.traj")
print(period)

ax.set_xlabel('time step')
ax.set_ylabel('volume')
ax.legend()
fig.savefig(f't_sc_{sc}_P_mass_{P_mass}_T_{T}_L_VARY.pdf')

####################################################################################################
# vary m
print("VARY m")
from matplotlib.figure import Figure
fig = Figure()
ax = fig.add_subplot()

# sc
P_mass = 10000.0
T = 400
sc = 3
for m_factor in [1.0, 4.0]:
    atoms_m = atoms_Al.copy()
    atoms_m.set_masses(atoms_m.get_masses() * m_factor)
    print(f'sc {sc} P_mass {P_mass} T {T} m_factor {m_factor} period ', end=''); sys.stdout.flush()
    period = do_run(atoms_m, sc, P_mass, T, calc=morse, label_extra=f' m_factor {m_factor}', ax=ax, traj_file=f"vary_m_{m_factor}.traj")
    print(period)

ax.set_xlabel('time step')
ax.set_ylabel('volume')
ax.legend()
fig.savefig(f't_sc_{sc}_P_mass_{P_mass}_T_{T}_m_VARY.pdf')

