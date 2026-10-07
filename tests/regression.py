#!/usr/bin/env python3
"""Native-field regressions; Python standard library only, no plotting dependencies."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference-executable", type=Path,
                        help="Optional unmodified MHD binary for a Taylor-Green comparison")
    parser.add_argument("--long", action="store_true",
                        help="Also retain order-unity magnetic growth through many remaps")
    args = parser.parse_args()
    parent = ROOT / "tests/output"
    parent.mkdir(exist_ok=True)
    output = args.output.resolve() if args.output else Path(tempfile.mkdtemp(prefix="run-",dir=parent))
    output.mkdir(parents=True,exist_ok=True)
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS","1")
    env.setdefault("OPENBLAS_NUM_THREADS","1")
    launcher = shlex.split(env.get("MPIEXEC","mpirun --bind-to none"))
    records, checks = {}, []

    def check(name, value, tolerance):
        passed = math.isfinite(value) and value <= tolerance
        checks.append(dict(name=name,value=value,tolerance=tolerance,passed=passed))
        if not passed:
            print(f"FAIL {name}: {value:.9g} > {tolerance:.9g}",flush=True)

    def launch(name, executable, options=(), ranks=1, expected_error=None):
        dest = output / name
        dest.mkdir(exist_ok=False)
        command = launcher + ["-np",str(ranks),str(executable)] + list(options)
        (dest/"command.json").write_text(json.dumps(command,indent=2)+"\n")
        with (dest/"stdout.log").open("w") as stream:
            result = subprocess.run(command,cwd=ROOT,env=env,stdout=stream,
                                    stderr=subprocess.STDOUT,timeout=600)
        log = (dest/"stdout.log").read_text()
        records[name] = dict(command=command,exit_code=result.returncode)
        if expected_error:
            valid = result.returncode != 0 and expected_error in log
        else:
            valid = result.returncode == 0
        checks.append(dict(name=name+": process result",passed=valid))
        print(name,"PASS" if valid else "FAIL",flush=True)
        if not valid:
            raise RuntimeError(f"Unexpected process result; inspect {dest/'stdout.log'}")
        return dest

    def summarize(dest):
        with (dest/"magnetic_audit.csv").open() as stream:
            data = list(csv.DictReader(stream))
        for row in data:
            for key in row:
                if key != "phase":
                    row[key] = float(row[key])
                    if not math.isfinite(row[key]):
                        raise ValueError(f"Nonfinite native diagnostic in {dest}")
        first, last = data[0],data[-1]
        b0 = [first["mean_B"+d] for d in "xyz"]
        norm = math.sqrt(sum(b*b for b in b0))
        flux = max(math.sqrt(sum((row["mean_B"+d]-b0[i])**2
                                for i,d in enumerate("xyz"))) for row in data)/norm
        divergence = max(abs(float(line.split()[1]))
                         for line in (dest/"B_divergence_error.dat").read_text().splitlines())
        pairs = []
        for i,row in enumerate(data):
            if row["phase"] == "pre_remap":
                after = data[i+1]
                if after["phase"] != "post_remap" or row["time"] != after["time"]:
                    raise ValueError("Unpaired remap diagnostics")
                pairs.append(abs(after["magnetic_energy"]-row["magnetic_energy"])/first["magnetic_energy"])
        return dict(final_time=last["time"],energy_ratio=last["magnetic_energy"]/first["magnetic_energy"],
                    remaps=len(pairs),max_remap_jump_over_initial_energy=max(pairs,default=0.0),
                    relative_mean_flux_drift=flux,divergence_L2=divergence,
                    relative_total_energy_drift=(last["total_energy"]-first["total_energy"])/first["total_energy"])

    def shear(name,dim=2,ref=1,mode="corrected",px=False,ux=0,uy=0,
              amplitude=1,interval=1,final=None,ranks=1,order=None):
        final = final if final is not None else (.05 if dim == 2 else .02)
        dest = output/name
        options = ["-p","14","-dim",str(dim),"-m",str(ROOT/"mesh"/(
            "Disk-4x4-quad.mesh" if dim == 2 else "Cube-4x4x4-hex.mesh")),
            "-rs",str(ref),"-ok","2","-ot","1","-or","1",
            "-oa",str(order if order is not None else (3 if dim == 2 else 4)),
            "-s","3","-tf",str(final),"-cfl","0.5","-cgt","1e-12",
            "-mst","1" if px else "3","-no-ale" if mode == "lagrangian" else "-ale",
            "-pmf" if mode == "corrected" else "-no-pmf","-ma",
            "-fsr","-fsri",str(interval),"-samp",str(amplitude),"-sboost",str(uy),
            "-sbx",str(ux),"-spx" if px else "-no-spx"]
        if px: options += ["-crz"]
        options += ["-rmv","1","-rme","1","-rma","1" if dim == 2 else "2",
                    "-rmr","3","-bpt","1","-no-pv","-no-vis","-od",str(dest)]
        expected = "Mean-field correction requires A order" if order is not None else None
        launch(name,ROOT/"MHD",options,ranks,expected)
        if expected: return
        native = summarize(dest)
        records[name].update(native)
        check(name+": reaches final time",abs(native["final_time"]-final),1e-12)
        check(name+": mean flux",native["relative_mean_flux_drift"],1e-8)
        check(name+": divergence",native["divergence_L2"],1e-10)
        if mode != "lagrangian":
            checks.append(dict(name=name+": actually remaps",passed=native["remaps"]>0))
        return native

    try:
        for binary in ("test_mean_field_geometry","test_periodic_projection"):
            for ranks in (1,2):
                launch(binary+f"-mpi{ranks}",ROOT/binary,ranks=ranks)
        for dim in (2,3):
            ref = 1 if dim == 2 else 0
            for px,label in ((False,"walls"),(True,"torus")):
                prefix = f"shear-{dim}d-{label}"
                reference = shear(prefix+"-lagrangian",dim,ref,"lagrangian",px)
                corrected = shear(prefix+"-corrected",dim,ref,px=px)
                check(prefix+": retained growth",abs(corrected["energy_ratio"]-reference["energy_ratio"]),1e-7)
                if dim == 2:
                    original = shear(prefix+"-original",dim,ref,"original",px)
                    retained = corrected["energy_ratio"]-1.0
                    checks.append(dict(name=prefix+": exposes original growth erasure",
                                       passed=retained>1e-3 and original["energy_ratio"]-1.0 < .2*retained))
                if px:
                    boosted = shear(prefix+"-xboost10",dim,ref,px=True,ux=10)
                    check(prefix+": comoving boost agreement",abs(boosted["energy_ratio"]-corrected["energy_ratio"]),1e-7)
                else:
                    parallel = shear(prefix+"-mpi2",dim,ref,ranks=2)
                    check(prefix+": MPI agreement",abs(parallel["energy_ratio"]-corrected["energy_ratio"]),1e-7)
        translation = shear("translation-x",px=True,ux=25,amplitude=0,final=.2,ranks=2)
        check("translation: no magnetic injection",abs(translation["energy_ratio"]-1.0),1e-8)
        shear("reject-insufficient-potential-order",dim=3,ref=0,px=True,order=2)

        for boost in (0,10):
            name = f"vortex3d-xboost{boost}"
            dest = output/name
            text = (ROOT/"tests/inputs/vortex3d.ic").read_text().splitlines()
            text[1] = f"{boost} 0 0"
            icfile = output/f"{name}.ic"
            icfile.write_text("\n".join(text)+"\n")
            options = ["-p","15","-dim","3","-bic",str(icfile),"-m",str(ROOT/"mesh/Cube-4x4x4-hex.mesh"),
                       "-rs","0","-ok","2","-ot","1","-or","1","-oa","4",
                       "-s","3","-tf",".2","-cfl",".4","-cgt","1e-12",
                       "-mst","1","-crz","-pmf","-ma","-fsr","-fsri","5",
                       "-rmv","1","-rme","1","-rma","2","-rmr","3","-bpt","1",
                       "-no-pv","-no-vis","-od",str(dest)]
            launch(name,ROOT/"MHD",options)
            native = summarize(dest)
            records[name].update(native)
            check(name+": reaches final time",abs(native["final_time"]-.2),1e-12)
            check(name+": mean flux",native["relative_mean_flux_drift"],1e-8)
            check(name+": divergence",native["divergence_L2"],1e-10)
            checks.append(dict(name=name+": actually remaps",passed=native["remaps"]>0))
        check("3D vortex: comoving boost agreement",abs(records["vortex3d-xboost0"]["energy_ratio"]-
                                                       records["vortex3d-xboost10"]["energy_ratio"]),1e-8)

        # Distributed Taylor-Green setup, coarsest mesh, same t=0.5 as its script.
        tg_options = ["-p","0","-dim","2","-m",str(ROOT/"mesh/Disk-4x4-quad.mesh"),
                      "-rs","0","-ok","2","-ot","1","-or","1","-oa","3",
                      "-s","3","-tf",".5","-cfl",".5","-cgt","1e-12","-mst","1",
                      "-ale","-fsr","-fsri","10","-rmv","1","-rme","1","-rma","1",
                      "-rmr","3","-bpt","1","-no-pv","-no-vis"]
        tg = launch("taylor-green",ROOT/"MHD",tg_options+["-od",str(output/"taylor-green")])
        def errors(dest):
            pairs = re.findall(r"L_2\s+error of (rho|v|e|B):\s*(\S+)",(dest/"stdout.log").read_text())
            return {key:float(value) for key,value in pairs}
        current = errors(tg)
        checks.append(dict(name="Taylor-Green: finite final errors",
                           passed=len(current)==4 and all(math.isfinite(v) for v in current.values())))
        records["taylor-green"]["final_L2_errors"] = current
        baseline = json.loads((ROOT/"tests/inputs/taylor_green_reference.json").read_text())
        for key,value in baseline["final_L2_errors"].items():
            check("Taylor-Green reference: "+key,abs(current[key]-value),baseline["absolute_tolerance"])
        if args.reference_executable:
            reference = launch("taylor-green-upstream",args.reference_executable.resolve(),
                               tg_options+["-od",str(output/"taylor-green-upstream")])
            previous = errors(reference)
            for key in current:
                check("Taylor-Green unchanged: "+key,abs(current[key]-previous[key]),1e-12)

        if args.long:
            reference = shear("long-lagrangian",ref=2,mode="lagrangian",final=.5,interval=10)
            corrected = shear("long-corrected",ref=2,final=.5,interval=10)
            check("long: retained magnetic growth",abs(corrected["energy_ratio"]-reference["energy_ratio"]),1e-5)
            checks.append(dict(name="long: order-unity growth",passed=corrected["energy_ratio"]>2.0))
            torus = shear("long-torus-xboost10",ref=2,px=True,ux=10,final=.5,interval=10)
            check("long: x-periodic boosted growth",abs(torus["energy_ratio"]-reference["energy_ratio"]),1e-5)
    except BaseException as error:
        checks.append(dict(name="suite completion",passed=False,error=str(error)))
        raise
    finally:
        provenance = {}
        for binary in ("MHD","test_mean_field_geometry","test_periodic_projection"):
            provenance[binary] = hashlib.sha256((ROOT/binary).read_bytes()).hexdigest()
        summary = dict(cases=records,checks=checks,binary_sha256=provenance,
                       all_checks_passed=all(c["passed"] for c in checks))
        (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
        print("Evidence:",output,flush=True)
    if not summary["all_checks_passed"]:
        raise SystemExit("Regression checks failed; inspect summary.json")
    print(f"All {len(checks)} checks passed.",flush=True)


if __name__ == "__main__":
    main()
