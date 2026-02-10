# `wif` - *W*FL-based *I*terative *F*itting

Use [wfl](https://github.com/libAtoms/workflow) to fit a machine-learning
potential by iteratively running MD, sampling configs, evaluating them
with a reference calculator, and adding them to the fitting config list.

# INSTALLATION

clone git repo, then
```
python3 -m pip install .
```

# SUPPORTED FEATURES

## DFT codes

- [`vasp`](https://www.vasp.at/)
  - MD sampling, fixed or variable cell, constant T (or constant E, untested)

## Machine learning interatomic potentials

- [`MACE`](https://github.com/ACEsuit/mace)
  - default to multihead-stabilized fine tuning of the [mace-omat-0-medium](https://github.com/ACEsuit/mace-foundations/releases/tag/mace_omat_0)
    [universal model] (https://arxiv.org/abs/2401.00096)

## Parallelization

`wif` uses [wfl](https://github.com/libAtoms/workflow) and
[expyre](https://github.com/libatoms/ExPyRe) to (optionally) automatically
parallelize tasks, e.g. running multiple MLIP MD trajectories in parallel,
separate queued jobs for multiple DFT evaluations, and queued jobs for
MLIP fitting on GPU nodes.

# BASIC USAGE

## iterative MD and fitting with a VASP-like interface

Set up a VASP calculation:
 - `INCAR`
 - `POSCAR`
 - `POTCAR`
 - `KPOINTS` (not required if `INCAR` contains a `KSPACING` tag)

Additional non-standard `INCAR` tags are available for control of the
sampling of configurations for fitting
- `WIF_N_CONFIGS_PER_FIT` - number of configs to collect for each fitting pass
  (summed across all trajectories, i.e. number of configs multiplied by number of
  temperatures, pressures, etc.)
- `WIF_CONFIG_INTERVAL` - number of MD time steps between collecting configurations
  for fitting

In addition to the usual electronic-structure-related parameters, `INCAR` needs to be set up
for MD, i.e. `IBRION = 0`, `POTIM`, `NSW`, `MDALGO` and associated thermostat/barostat parameters,
and usually `TEBEG`.

If `POSCAR` does not exist, `*.POSCAR` will be used, each for a separate
MD trajectory, and configurations will be sampled from each.

Run with
```
wif -s seed [ -i inputs_dir ] vasp_executable
```

`seed` is a random seed, which is required to encourage you to make your simulations reproducible

`-i inputs_dir` is an optional location for all the input files, without which `.` is assumed (like VASP).

`vasp_executable` is the vasp executable command line you want to use. In principle this could include
`mpirun`, etc, but so far only tested with wrappers that hide all of that complexity.

Basic configuration is derived from `INCAR` and `POSCAR`, which define the usual VASP electronic structure
solution and MD parameters.

### non-standard `INCAR` tags

Some tags can take multiple values
- `TEBEG`
- `TEEND`
- `PSTRESS`

If there are multiple values they are used to run multiple MD trajectories
(optionally in parallel).  `TEBEG` and `TEEND`, if present, must match
in length.  By default, each combination of conditions (temperature
schedule, i.e. `TEBEG`/`TEEND` pair, and pressure) will be applied to
each configuration, so the number of MD trajectories will be number of
configs times number of temperatures times number of pressures.  This can
be modified with `wif.toml` keys.  See `wif --default_params`
for syntax.

There are also three additional possible `INCAR` tags:
- `WFL_N_CONFIGS_PER_FIT`: integer number of configuration to gather (total across all MD trajectories) before
   VASP-evaluating and re-fitting the potential (includes both fitting and validation configs)
- one of
    - `WFL_CONFIG_INTERVAL`: integer spacing (in time steps) between sampling MD configurations for inclusion
      in evaluation/fitting batch.  Trajectories will be interrupted once enough configs to satisfy
      `WIF_N_CONFIGS_PER_FIT` (i.e. total time steps equal to product of config interval and needed number of
      configs per trajectory).
    - `WFL_FIT_INTERVAL`: integer spacing (in time steps) between interrupting the trajectories to do
      a reference evaluation and fit.  Configs will be collected at equal intervals to satisfy 
      `WIF_N_CONFIGS_PER_FIT` (i.e. time steps between config selections will be equal to fit
      interval divided by needed number of configs per trajectory).

### `toml` configuration files

Additional configuration can be set via a [toml](https://toml.io) file named `wif.toml`, in `inputs_dir`.

The complete set of possible options can be printed with
```
wif --default_params
```

### combining multiple fits

If you are using radically different conditions, it may be useful to apply a subset, gather some configurations,
then another.  One example is small cells, which using k-point summation, and large supercells that can
be gamma-point only (in addition to changing the vasp executable you're likely to want more small cells per fit 
than large ones, because there is more information in the larger number of atomic forces).   This is most
easily done by setting up separate directories for each `wif` run.  The `wil.toml` can have a field 
`fit.prev_run_dirs = [ "dir_1", "dir_2", ... ]` that specifies the directories containing earlier runs,
and their fitting, validation, and isolated atoms files will be automatically added to the fits.

### using `Symmetrix` for faster MD (especially on CPU)

If you are running the MD for sampling on CPU (e.g. if you have many initial configs
and you have a lot more cores to parallelize over than GPUs), you can use the `Symmetrix`
package if it is installed.  To do this, you need to add
```
[potential]
kwargs = {symmetrix = true, device = "cpu", species = ["Sym1", "Sym2" ... ]}
```
where `SymN` are the chemical symbols in your system.  Note that to bootstrap this
process you need a symmetrix format version of the foundation model you are starting
from, which you can generate with
```
create_symmetrix_mace.py -Z Z1 Z2 ... -- $HOME/.cache/mace/maceomat0mediummodel
```
where `ZN` are the atomic numbers in your system.

### multi-GPU fitting

You can run the fit on more than one GPU, although it is a bit cumbersome.
At a minimum, the `wif.toml` needs
```
[potential.fit.mace]
    wfl_fit_kwargs = {mace_fit_cmd = "srun mace_run_train"}

[potential.fit.mace.command_line_args]
    distributed = "_NONE_"
```
and you need to make sure the fitting task runs in a job that requested
more than one GPU. If you are using `wfl` to spawn the fitting task
to a _single_ GPU node, you can do that using something like
```
[fit.remote_info]
    timeout = -1
    sys_name = "tin"
    job_name = "wif_CuAlNi_prim_fit"
    pre_cmds = ["module purge; module load compiler/gnu lapack/mkl python/conda cuda python_extras/torch/gpu python_extras/wif; module list"]
    # one gpu (also remove `mace_fit_cmd` from `wif.toml`)
    # resources = {num_nodes = 1, max_time = "24h", partitions = "gpu_1"}
    # two gpus
    partial_node = true
    resources = {num_cores = 2, max_time = "24h", partitions = "gpu_2"}

```
in `wif_wfl.toml`.  

** [BELOW IS UNTESTED] **

To use more than one node you need to either create custom
`~/.expyre/config.json` partitions that set a number of cores for each
node that is equal to the number of GPUs, or add command lines args to the
parallel startup in `mace_fit_cmd` string, namely something like 
`"srun --ntasks=<total N GPUs> --ntasks-per-node=<N GPUs per node> mace_run_train"`.

## standalone fitting

No VASP input files are required, just (optional) `wif.toml`

Run with
```
wif_fit -s seed [ -i inputs_dir ] -f fitting_files_glob.xyz -v validation_files_glob.xyz -a isolated_atoms_files_glob.xyz
```

The correct globs if you are redoing a fit from multiple previous `wif` runs, which were run in directories
named `run_1`, `run_2`, etc., would be something like
```
-f "run_*/md_step*.fit.extxyz" -v "run_*/md_step*.validation.extxyz" -a "run_*/isolated_atoms.*.extxyz"
```

## standalone md

VASP `INCAR` and `POSCAR` are required, but no electronic-structure specific files such as `KPOINTS`
or `POTCAR`.  Run with
```
wif_md -s seed -p potential_file [ -i inputs_dir ] [--continuation]
```
`--continuation` is required, and will assume that existing trajectory files should be used
as source of initial config and be appended to.

MD can be driven from VASP `INCAR` and `POSCAR` files (no electronic-structure related
files such as `POTCAR` or `KPOINTS` are used).  The usual `INCAR` tags are used
- `POTIM` - time step
- `IBRION` - only 0 (MD) is supported
- `MDALGO` - a subset of values are supported
  - `0` - const. E or const. T, depending on `SMASS`
  - `1` - const. T
    - `ANDERSEN_PROB` - converted to thermostat timescale
  - `3` - const. T
    - `LANGEVIN_GAMMA` - converted to thermostat timescale
    - `LANGEVIN_GAMMA_L` - converted to barostat timescale
  - `4` - const. T
    - `NHC_PERIOD` - converted to thermostat timescale
  - `5` - const. T
    - `CSVR_PERIOD` - converted to thermostat timescale
- `SMASS` - thermostat mass
  - `-3` - constant E
  - `0` - heuristic value, 40 * `POTIM`
- `ISIF` - fixed cell (0) or variable volume (8) (variable cell shape (3) not yet supported)
- `NSW` - number of MD steps
- `TEBEG` - temperature, activates NVT or NPT
- `TEEND` - final temperature ramp, optional
- `PSTRESS` - applied pressure, defaults to 0.0

## parallelization example

This is an example of a `wif_wfl.toml` file that activates the
parallelization several of the main iterative fit operations.
The fitting MD will use 2 python threads, the DFT will be done with
one queued job per configuration, and the fitting will be done with a
queued job directed to a GPU node.  Note that for the queued jobs (DFT
evaluation and MACE fitting) to work also requires a `~/.expyre/config.json`
file.  See the [wfl](https://libatoms.github.io/workflow) and
[expyre](https://libatoms.github.io/ExPyRe) documentation.

```
[fit_md]

    num_inputs_per_python_subprocess = 1
    num_python_subprocesses = 2

[dft_eval]

    [dft_eval.remote_info]

        sys_name = "tin"
        job_name = "wif_pytest_dft_eval"
        num_inputs_per_queued_job = 2
        input_files = ["input_files/POTCARs"]
        pre_cmds = [ "module unload dft/vasp; module load dft/vasp/6.4.3" ]
        # env_vars = ["ASE_VASP_COMMAND=vasp.para", "ASE_VASP_COMMAND_GAMMA=vasp.gamma.para"]
        resources = {num_cores = 16, max_time = "1h", partitions = "n2013,n2016"}

# [dft_isolated_atoms]

[fit]

    [fit.remote_info]

        sys_name = "tin"
        job_name = "wif_pytest_fit"
        pre_cmds = ["module unload cuda python_extra/torch/cpu python_extras/torch/gpu; module load cuda python_extras/torch/gpu"]
        resources = {num_nodes = 1, max_time = "1h", partitions = "gpu_1"}
```

# FAQ

- If you get a failure during the fitting with messages such as `Model download failed and no local model found`,
  or similar text about replay data or descriptors, the most likely cause is that you're fine-tuning a foundation
  model but `mace_run_train` failed to download the model itself or its replay data because compute nodes don't
  have access to the internet.  In this case, add the `--download_cached_data` command line flag to `wif` or `wif_fit`,
  and run it on the **head node**, which will download and cache the necessary data (to `~/.cache/mace/` or
  `${XDG_CACHE_HOME}/mace`).

# TODO

- [MEDIUM, STARTED] refactor into useful pieces - started by removing bits from cli and moving into more reusable functions.
  Can we at least reuse "standalone fit" in "iterative md"?
- [EASY] move isolated atoms to before loop, assuming that composition doesn't change during iteration.
  Note that it might even be better to move it to a separate script, since isolated atoms
  can be difficult to converge and might require therefore different DFT params
- [EASY?] see if we can get rid of wif having to deal with mace's checkpoint download - just pass non-constructor mace function.
  May not be possible, since kwargs are too different
- [MEDIUM] more clever fitting configuration selection method (CUR, FPS, something else?) from among sampled configs
- [UNCLEAR] Actual UQ-driven sampling

## DONE

- variable cell MD in `Langevin_BAOAB`
- `fit.prev_run_dirs` that automatically grabs all (fitting, validation. isolated atoms) evaluated configs
  from a previous `wif_vasp` run
- switch MACE fit default to small model (same as already used for MD)
- allow for manual submission of vasp runs
- support multiple MD trajectories with different parameters such as temperatures or pressures,
  and/or different starting configurations
- make `--default_params` only report relevant sections for each CLI script
- start each stage of iterative MD/fit from last config of previous stage
- implement temperature ramp in `wif_md`
- Add test of rerun of interrupted run
- check interrupted `vasp_wif` replay to confirm that error tables are same on second (replay) pass
- make error calculation store configs with calculated MACE quantities in files to avoid redoing it on rerun
- enable logging of sampling MD by default
- hooks for wfl autopara
- MD-only mode, no fitting (w/ vasp interface)
- check for repeats of a given element in a single POTCAR
- standalone fit mode
- report errors from fits
