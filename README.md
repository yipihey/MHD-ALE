# A structure-preserving, helicity-conserving high-order ALE finite element method for compressible ideal MHD systems 

* Use [this branch](https://github.com/ruijie-xi/mfem-develop) of MFEM 
* Example run script: `test_TaylorGreen.sh`

## Build

The required MFEM fork provides APIs used by this solver. The additions were
validated with MFEM commit `842fac6a3b038eaee848921f096e80ceb5f895fa`, double
precision, MPI, METIS 5, HYPRE and GSLIB, using GCC 14.2 and Open MPI 5.0.7.
GSLIB commit `375eda25b04205d1f749ab0e00a0bb3f22225b22` was used. Install those
dependencies and the MFEM fork, then build with:

```sh
make -j4 MFEM_INSTALL_DIR=/absolute/path/to/mfem-install
```

MFEM must be configured with `MFEM_USE_MPI=YES`, `MFEM_USE_METIS=YES`,
`MFEM_USE_METIS_5=YES` and `MFEM_USE_GSLIB=YES`. Its installed
`share/mfem/config.mk` supplies compiler and dependency flags. See
[build details](docs/periodic-remap.md#dependency-build-example) for a pinned
dependency build example.

## Periodic boundaries and magnetic remapping

Positive lengths in `ProblemData::SetPeriodic(px,py,pz,B0)` enable each periodic
axis independently. The density remapper now searches x images as well as y/z
images, including sources translated by multiple box lengths. Existing y/z
API calls remain valid.

For ALE with periodic x, use the initial-reference smoother (`-mst 1`). The
older periodic smoother (`-mst 3`) assumes x walls. `-crz` / `--comoving-rezone`
retains the reference mesh's bulk translation in enabled periodic directions.

**Enable `-pmf` / `--preserve-mean-field` to retain stretching of a uniform
periodic background magnetic field across remaps.** The correction converts
the transported background's fluctuation into a potential before applying the
supplied remapper. It supports both the 2-D DG and 3-D H(curl) paths. The new
options default to disabled; `-no-pmf` reproduces the original remap behavior.

The potential degree must be at least the mesh degree in 2-D, and at least
twice the mesh degree in 3-D. For quadratic geometry use `-oa 3` in 2-D or
`-oa 4` in 3-D. The implementation checks the uniform background and the
native pre-remap magnetic-field decomposition. Mathematical details and
limitations are in [the remapping note](docs/periodic-remap.md).

Two self-contained problems are available:

| Problem | Initial conditions | Boundaries |
| --- | --- | --- |
| `-p 14` | Weak-field sinusoidal shear; `-samp`, `-sboost`, `-sbx` set amplitude and y/x boosts | Periodic y/z; `-spx` also enables periodic x |
| `-p 15 -bic FILE` | Fourier velocity and magnetic modes, with a uniform seed field | Fully periodic unit square/cube |

An x-periodic shear example is:

```sh
mkdir -p output/periodic-shear
mpirun -np 1 ./MHD -p 14 -dim 2 -m mesh/Disk-4x4-quad.mesh \
  -rs 2 -ok 2 -ot 1 -or 1 -oa 3 -s 3 -tf 0.5 -cfl 0.5 -cgt 1e-12 \
  -spx -mst 1 -crz -pmf -ma -fsr -fsri 10 \
  -rmv 1 -rme 1 -rma 1 -rmr 3 -bpt 1 -no-pv -no-vis \
  -od output/periodic-shear
```

`-ma` writes native magnetic energy, total energy and mean magnetic flux in
`magnetic_audit.csv`, including paired measurements before and after remaps.
Diagnostic files are opened only by MPI rank zero. The Fourier file format
and a 3-D example are documented in the remapping note.

## Regression checks

Python 3 is needed for the test driver; it uses only the standard library.

```sh
make -j4 MFEM_INSTALL_DIR=/absolute/path/to/mfem-install check
python3 tests/regression.py --long
```

Checks cover geometric identities, x/y/full periodic projection over multiple
box lengths, both coordinate orderings, one/two MPI ranks, 2-D/3-D physical
shear growth, x boosts, uniform translation, a 3-D Fourier flow, unsupported
potential order, and the supplied Taylor-Green setup. `--long` also tests
order-unity magnetic growth through repeated remaps. An optional
`--reference-executable /path/to/original/MHD` compares final Taylor-Green
errors with an unmodified upstream binary.

Each run creates a new directory under `tests/output/` containing commands,
native diagnostics, logs, binary hashes and `summary.json`. These generated
files and binaries are ignored by git.

The [recorded validation results](docs/validation.md) include the long growth
control and the comparison with the unmodified Taylor-Green executable.
