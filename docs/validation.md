# Validation of periodic boundaries and mean-field remapping

The solver additions in commit `1a263343d489b1e4fb38c56cb09d06775ec92cbf` passed 130 driver checks across 28 executions. The insufficient-order execution was expected to fail; the other executions completed successfully. Tests used the pinned MFEM/GSLIB dependencies documented in the README, GCC 14.2 and Open MPI 5.0.7.

Command:

```sh
python3 tests/regression.py --long --reference-executable /path/to/unmodified/MHD
```

## Native magnetic-growth results

| Control | Final time | No-remap E_B/E_B0 | Corrected E_B/E_B0 | Remaps |
| --- | --- | --- | --- | --- |
| 2-D, x walls | 0.05 | 1.012333789764 | 1.012333789764 | 24 |
| 2-D, fully periodic | 0.05 | 1.012333789741 | 1.012333789741 | 24 |
| 3-D, x walls | 0.02 | 1.001966022125 | 1.001966023336 | 4 |
| 3-D, fully periodic | 0.02 | 1.001966022123 | 1.001966022968 | 4 |
| Long 2-D, x walls | 0.50 | 2.233676303519 | 2.233676303519 | 48 |
| Long 2-D, fully periodic, x boost 10 | 0.50 | 2.233676184588 | 2.233676184588 | 48 |

The long periodic comparison uses the same boundaries and x boost in the no-remap reference. Its final energy ratio differs from the corrected result by 1.625e-13.

The original 2-D wall remap gives E_B/E_B0 = 1.000001894061 at t=0.05, versus 1.012333789764 with the correction. The same growth-erasure control passes with periodic x.

The maximum relative mean-flux drift among measured physical controls is 1.399e-13; maximum native L2 divergence is 5.943e-14. These are diagnostics of this validation set, not universal error bounds.

## Other checks

- General quadratic geometry identities pass in both dimensions on one and two MPI ranks.
- Periodic density projection across several box lengths passes for x-only, y-only and fully periodic boundaries, both coordinate orderings and one/two ranks. The same controls check the comoving reference and removal of rigid displacement from the rezone criterion.
- Uniform x translation through five box lengths introduces no measurable magnetic-energy growth within the 1e-8 tolerance.
- Corrected physical shears pass one/two-rank and comoving x-boost comparisons.
- A coarse 3-D Fourier flow retains the same magnetic growth with x boosts 0 and 10; this is a regression control, not a turbulence convergence result.
- The stored Taylor-Green baseline passes, and its final rho/v/e/B L2 errors agree with an unmodified upstream executable to within 1e-12.
- Insufficient magnetic-potential order is rejected with the expected diagnostic.

Generated per-case commands, logs and native diagnostics remain under the ignored `tests/output/` directory. The transfer evidence archive includes that final directory and its binary hashes. All figures and large comparison-suite outputs remain in the independent experiment repository.

## Limits

The correction preserves the representation of a uniform harmonic background through fixed-connectivity rezoning. It does not demonstrate exact energy conservation for arbitrary remaps, a nonzero-net-flux helicity invariant, or convergence of general three-dimensional turbulence. No r3djl remapper is included. See [the mathematical and implementation note](periodic-remap.md).
