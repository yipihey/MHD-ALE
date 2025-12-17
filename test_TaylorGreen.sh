#!/bin/bash

# ==============================================================================
# Parallel Configuration
# ==============================================================================
refine=(0 1 2 3)
NPROC=(1 1 1 4)

# ==============================================================================
# Problem Setup
# ==============================================================================
TESTCASE=0                  # -p: Problem type (0: Taylor-Green (MHD), 9: Taylor-Green (hydro case))
DIM=2                       # -dim: Dimension of the problem
MESH_FILE="./mesh/Disk-4x4-quad.mesh" # -m: Mesh file
T_FINAL=0.5                 # -tf: Final time

# ==============================================================================
# Finite Element Orders
# ==============================================================================
ORDER_VELOCITY=2            # -ok: Kinematic space order (velocity)
ORDER_ENERGY=$(($ORDER_VELOCITY-1))   # -ot: Thermodynamic space order (energy)
ORDER_MAGNETIC=$(($ORDER_VELOCITY+1)) # -oa: Magnetic potential space order
ORDER_DENSITY=$(($ORDER_VELOCITY-1))  # -or: Density space order

# ==============================================================================
# Time Integration
# ==============================================================================
ODE_SOLVER=3                # -s: ODE solver type (3=RK3SSP, 4=RK4, etc.)
CFL=0.5                     # -cfl: CFL number
GAMMA1=0.85                 # -g1: Time step decrease factor
GAMMA2=1.02                 # -g2: Time step increase factor

# ==============================================================================
# Linear Solver (Velocity)
# ==============================================================================
CG_TOL=1e-12                # -cgt: CG tolerance

# ==============================================================================
# ALE and Remeshing
# ==============================================================================
ALE="-ale"                  # -ale/-no-ale: Enable/Disable ALE
MESH_SMOOTH_TYPE=1          # -mst: Mesh smoothing type
FIX_STEP_REMESH="-fsr"      # -fsr/-no-fsr: Fixed step remeshing
FSRI=10                     # -fsri: Interval for fixed step remeshing

# ==============================================================================
# Remap Options
# ==============================================================================
REMAP_VELOCITY=1            # -rmv: Velocity remap (0=Interp, 1=DG)
REMAP_ENERGY=1              # -rme: Energy remap (0=Interp, 1=DG)
REMAP_MAGNETIC=1            # -rma: A-field remap (0=Interp, 1=DG, 2=Helicity)
REMAP_DENSITY=3             # -rmr: Density remap (3=L2Projection)
BOUND_PRES_TYPE=1           # -bpt: Bound preserving type (1=Positive, 2=Local, 3=Global)

# ==============================================================================
# Visualization and Output
# ==============================================================================
PARAVIEW="-pv"              # -pv/-no-pv: ParaView visualization
PARAVIEW_BASENAME="TaylorGreen" # -pvb: ParaView basename
PLOT_TIME_FILE="plot_time/plot_time_TaylorGreen.dat" # -ptf: Plot time file

# ==============================================================================
# Execution Loop
# ==============================================================================

for i in "${!refine[@]}"; do
    
    OUTPUTDIR=output/TaylorGreen-Dim${DIM}-Refine${refine[$i]}-Order${ORDER_VELOCITY}${ALE}-Remap-R${REMAP_DENSITY}-V${REMAP_VELOCITY}-E${REMAP_ENERGY}-A${REMAP_MAGNETIC}-FSRI${FSRI}

    mkdir -p $OUTPUTDIR

    export RUN_OPTS="-p $TESTCASE \
-rs ${refine[$i]} \
-ptf ${PLOT_TIME_FILE} \
-s ${ODE_SOLVER} \
-m ${MESH_FILE} \
${PARAVIEW} \
-od ${OUTPUTDIR} \
-pvb ${PARAVIEW_BASENAME} \
${ALE} \
-tf ${T_FINAL} \
-mst ${MESH_SMOOTH_TYPE} \
-ok ${ORDER_VELOCITY} \
-ot ${ORDER_ENERGY} \
-or ${ORDER_DENSITY} \
-oa ${ORDER_MAGNETIC} \
-cfl ${CFL} \
-dim ${DIM} \
-g1 ${GAMMA1} \
-g2 ${GAMMA2} \
${FIX_STEP_REMESH} \
-fsri ${FSRI} \
-rmv ${REMAP_VELOCITY} \
-rme ${REMAP_ENERGY} \
-rma ${REMAP_MAGNETIC} \
-rmr ${REMAP_DENSITY} \
-cgt ${CG_TOL} \
-bpt ${BOUND_PRES_TYPE}"

    mpirun -np ${NPROC[$i]} ./MHD $RUN_OPTS > ${OUTPUTDIR}/MHD.out 2> ${OUTPUTDIR}/MHD.err
    
    unset RUN_OPTS

done


unset Machine