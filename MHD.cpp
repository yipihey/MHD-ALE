#include "Problemdata.hpp"
#include "mesh_smoother.hpp"
#include "remap.hpp"
#include "tools.hpp"
#include "fstream"
#include <sys/stat.h>
#include <sys/types.h>

#include <unistd.h>   
#include "MHD_solver.hpp"

// useful macros
#define mfemPrintf(...) { \
    if (Mpi::Root()) { \
          printf(__VA_ARGS__); \
    } \
}

using std::cout;
using std::endl;
using namespace mfem;

const int filename_length = 512;

// Choice for the problem setup.
static Testcase problem = TAYLOR_GREEN;
static int dim;

void GetPlotTime(const char *time_file, Array<real_t> &plot_time)
{
    std::ifstream time_in(time_file);
    real_t t;
    while(time_in >> t){
        plot_time.Append(t);
    }
    time_in.close();
    
    mfemPrintf("time to plot: size = %d, plot_time list = [", plot_time.Size());
    for(int i=0; i<plot_time.Size(); i++){
        mfemPrintf(" %f, ", plot_time[i]);
    }
    mfemPrintf("]\n");
}

bool CheckPlot(const Array<real_t> &plot_time, real_t t)
{
    static int target = 0;
    real_t eps = 1e-7;
    
    if(target >= plot_time.Size()){
        return false;
    }
    
    if(t >= plot_time[target] - eps){
         mfemPrintf("Checkplot: t = %g, target = %d, plot_time[target] = %g, \n", t, target, plot_time[target]);
        target++;
        return true;
    }
    
    return false;
}

void PrintError(ProblemData *pd, real_t t,
                 ParGridFunction &rho_gf, ParGridFunction &v_gf,
                 ParGridFunction &e_gf, ParGridFunction &B_gf, const IntegrationRule **irs = nullptr)
{
   // Print the error.
   if (pd->has_exact_solution)
   {
      real_t error_max, error_l1, error_l2;
      
      FunctionCoefficient rho_exact_coeff(pd->exact_rho);
      rho_exact_coeff.SetTime(t);
      error_max = rho_gf.ComputeMaxError(rho_exact_coeff, irs);
      error_l1  = rho_gf.ComputeL1Error(rho_exact_coeff, irs);
      error_l2  = rho_gf.ComputeL2Error(rho_exact_coeff, irs);
      
      if (Mpi::Root())
      {
         cout << "-------------------------------------------------------" << endl;
         cout << "L_inf  error of rho: " << error_max << endl
              << "L_1    error of rho: " << error_l1 << endl
              << "L_2    error of rho: " << error_l2 << endl;
      }
      
      VectorFunctionCoefficient v_exact_coeff(dim, pd->exact_v);
      v_exact_coeff.SetTime(t);
      error_max = v_gf.ComputeMaxError(v_exact_coeff, irs),
      error_l1  = v_gf.ComputeL1Error(v_exact_coeff, irs),
      error_l2  = v_gf.ComputeL2Error(v_exact_coeff, irs);
      if (Mpi::Root())
      {
         cout << "L_inf  error of v: " << error_max << endl
              << "L_1    error of v: " << error_l1 << endl
              << "L_2    error of v: " << error_l2 << endl;
      }
      
      FunctionCoefficient e_exact_coeff(pd->exact_e);
      e_exact_coeff.SetTime(t);
      error_max = e_gf.ComputeMaxError(e_exact_coeff, irs);
      error_l1  = e_gf.ComputeL1Error(e_exact_coeff, irs);
      error_l2  = e_gf.ComputeL2Error(e_exact_coeff, irs);
      
      if (Mpi::Root())
      {
         cout << "L_inf  error of e: " << error_max << endl
              << "L_1    error of e: " << error_l1 << endl
              << "L_2    error of e: " << error_l2 << endl;
      }
      
      VectorFunctionCoefficient B_exact_coeff(dim, pd->exact_B);
      B_exact_coeff.SetTime(t);
      error_max = B_gf.ComputeMaxError(B_exact_coeff, irs);
      error_l1  = B_gf.ComputeL1Error(B_exact_coeff, irs);
      error_l2  = B_gf.ComputeL2Error(B_exact_coeff, irs);
      
      if (Mpi::Root())
      {
         cout << "L_inf  error of B: " << error_max << endl
              << "L_1    error of B: " << error_l1 << endl
              << "L_2    error of B: " << error_l2 << endl;
         cout << "-------------------------------------------------------" << endl;
      }
   }
}

static void Visualization(bool vis, bool paraview,
   ParMesh *pmesh, ParGridFunction &rho_gf, 
  ParGridFunction &v_gf, ParGridFunction &e_gf,
  ParGridFunction &B_gf, ParGridFunction &A_gf, 
  int ti, real_t t,
  const char *outputdir,
  const char *paraview_basename);

int main(int argc, char *argv[])
{
   // Initialize MPI.
   Mpi::Init();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // Parse command-line options.
   problem = TAYLOR_GREEN;
   dim = 3;
   const char *mesh_file = "default";
   int rs_levels = 2;
   int order_v = 2;
   int order_rho = 2;
   int order_e = 2;
   int order_A = 2;
   int order_q = -1;
   int ode_solver_type = 4;
   double t_final = 0.6;
   double cfl = 0.5;
   double cg_tol = 1e-8;
   int cg_max_iter = 300;
   int max_tsteps = -1;
   bool impose_visc = false;
   bool visualization = false;
   bool paraview = false;
   const char *outputdir = "output";
   const char *paraview_basename = "MHD";
   
   bool ale = true;
   bool preserve_mean_field = false;
   bool comoving_rezone = false;
   bool magnetic_audit_enabled = false;
   bool shear_periodic_x = false;
   real_t shear_boost_x = 0.0;
   const char *box_ic_file = "box.ic";
   real_t shear_boost = 0.0;
   real_t shear_amplitude = 1.0;
   MeshSmoothType mesh_smooth_type = MeshSmoothType::LIMITED_HARMONIC;
   real_t smooth_eps = 1e-4;
   bool debug = false;
   bool fix_step_remesh = false;
   int fix_step_remesh_interval = 20;
   real_t min_detJ = 0.4;
   real_t max_detJ = 5.0;
   real_t max_ratio = 4.0;
   bool remap_at_last_step = false;
   real_t max_displacement = 1e20; 
   // For periodic mesh remeshing, remesh when max displacement in y,z direction > max_displacement
   
   RemapType remap_type_v = RemapType::DGconvection;
   RemapType remap_type_e = RemapType::DGconvection;
   RemapType remap_type_A = dim==3 ? RemapType::HelicityPreserving : RemapType::DGconvection;
   RemapType remap_type_rho = RemapType::L2Projection;
   BoundPreservingType bp_type = BoundPreservingType::POSITIVE;
   
   const char *plot_time_file = "plot_time/plot_time.dat";
   Array<real_t> plot_time;
   bool plot_at_remap = false;
   
   // time step control 
   real_t gamma1 = 0.85;
   real_t gamma2 = 1.02;
   
   // problem dependent paramters;
   real_t Bmag = 1.0;

   OptionsParser args(argc, argv);
   args.AddOption(&dim, "-dim", "--dimension", "Dimension of the problem.");
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption((int*)&problem, "-p", "--problem", "Problem setup to use.");
   args.AddOption(&order_v, "-ok", "--order-kinematic",
                  "Order (degree) of the kinematic finite element space.");
   args.AddOption(&order_e, "-ot", "--order-thermo",
                  "Order (degree) of the thermodynamic finite element space.");
   args.AddOption(&order_q, "-oq", "--order-intrule",
                  "Order  of the integration rule.");
   args.AddOption(&order_A, "-oa", "--order-magnetic-potential",
                   "Order (degree) of the magnetic potential finite element space.");
   args.AddOption(&order_rho, "-or", "--order-density",
                  "Order (degree) of the density finite element space.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Forward Euler,\n\t"
                  "            2 - RK2 SSP, 3 - RK3 SSP, 4 - RK4, 6 - RK6,\n\t");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&cfl, "-cfl", "--cfl", "CFL-condition number.");
   args.AddOption(&cg_tol, "-cgt", "--cg-tol",
                  "Relative CG tolerance (velocity linear solve).");
   args.AddOption(&cg_max_iter, "-cgm", "--cg-max-steps",
                  "Maximum number of CG iterations (velocity linear solve).");
   args.AddOption(&max_tsteps, "-ms", "--max-steps",
                  "Maximum number of steps (negative means no restriction).");
   args.AddOption(&impose_visc, "-iv", "--impose-viscosity", "-niv",
                  "--no-impose-viscosity",
                  "Use active viscosity terms even for smooth problems.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                  "Enable or disable ParaView visualization.");
   args.AddOption(&outputdir, "-od", "--outputdir",
                  "Directory to save the output files.");
   args.AddOption(&paraview_basename, "-pvb", "--paraview-basename",
                  "Base name for the ParaView output files.");
   args.AddOption(&preserve_mean_field, "-pmf", "--preserve-mean-field",
                  "-no-pmf", "--no-preserve-mean-field",
                  "Retain transported uniform periodic background flux through remap.");
   args.AddOption(&comoving_rezone, "-crz", "--comoving-rezone",
                  "-no-crz", "--no-comoving-rezone",
                  "Retain bulk translation in periodic initial-mesh rezoning.");
   args.AddOption(&magnetic_audit_enabled, "-ma", "--magnetic-audit",
                  "-no-ma", "--no-magnetic-audit",
                  "Record native magnetic energy and mean flux around remaps.");
   args.AddOption(&shear_periodic_x, "-spx", "--shear-periodic-x",
                  "-no-spx", "--no-shear-periodic-x",
                  "Make the shear regression periodic in x as well as y/z.");
   args.AddOption(&shear_boost_x, "-sbx", "--shear-boost-x",
                  "Uniform x boost for the fully periodic shear regression.");
   args.AddOption(&box_ic_file, "-bic", "--box-ic",
                  "Fourier initial-condition file for the periodic box (problem 15).");
   args.AddOption(&shear_amplitude, "-samp", "--shear-amplitude", "Shear amplitude; zero gives a pure translation control.");
   args.AddOption(&shear_boost, "-sboost", "--shear-boost", "Uniform y boost for the shear audit.");
   args.AddOption(&ale, "-ale", "--ale", "-no-ale", "--no-ale",
                  "Enable ALE (Arbitrary Lagrangian-Eulerian) mesh motion.");
   args.AddOption((int*)&mesh_smooth_type, "-mst", "--mesh-smooth-type",
                  "Type of mesh smoothing to use for ALE");
   args.AddOption(&smooth_eps, "-mse", "--mesh-smooth-epsilon",
                  "Epsilon for the limited harmonic mesh smoother.");
   args.AddOption(&fix_step_remesh, "-fsr", "--fix-step-remesh",
                  "-no-fsr", "--no-fix-step-remesh",
                  "Enable remeshing at a fixed time step interval.");
   args.AddOption(&fix_step_remesh_interval, "-fsri", "--fix-step-remesh-interval",
                  "Interval for remeshing at a fixed time step (if enabled)."); 
   args.AddOption(&min_detJ, "-mdj", "--min-detJ",
                  "Minimum relative determinant of the Jacobian for remeshing.");
   args.AddOption(&max_detJ, "-Mdj", "--max-detJ",
                  "Maximum relative determinant of the Jacobian for remeshing.");
   args.AddOption(&max_ratio, "-mr", "--max-ratio",
                  "Maximum ratio of the maximum to minimum singular value of the Jacobian for remeshing.");
   args.AddOption(&remap_at_last_step, "-rals", "--remap-at-last-step", "-nrals", "--no-remap-at-last-step", 
                  "Enable remap at last step.");
   args.AddOption(&max_displacement, "-mdis", "--max-displacement",
                  "Maximum displacement in periodic directions for remeshing.");
   args.AddOption((int*)&remap_type_v, "-rmv", "--remap-type-v",
                  "Type of remap for the velocity field.\n\t"
                  "0 - Interpolate, 1 - DG convection.");
   args.AddOption((int*)&remap_type_e, "-rme", "--remap-type-e",
                  "Type of remap for the specific internal energy field.\n\t"
                  "0 - Interpolate, 1 - DG convection.");   
   args.AddOption((int*)&remap_type_A, "-rma", "--remap-type-A",
                  "Type of remap for the magnetic vector potential field.\n\t"
                  "0 - Interpolate, 1 - DG convection2,  - Helicity preserving.");
   args.AddOption((int*)&remap_type_rho, "-rmr", "--remap-type-rho",
                  "Type of remap for the density field.\n\t"
                  "0 - Interpolate.");               
   args.AddOption((int*)&bp_type, "-bpt", "--bound-preserving-type",
                  "Type of bound preserving limiter.\n\t"
                  "0 - None, 1 - Positive, 2 - Local bound, 3 - Global bound.");
                  
   args.AddOption(&plot_time_file, "-ptf", "--plot-time-file",
                  "File to save the plot time data.");
   args.AddOption(&plot_at_remap, "-par", "--plot-at-remap",
                  "-nopar", "--not-plot-at-remap", 
                  "Plot the solution at remeshing time steps.");
                  
   args.AddOption(&gamma1, "-g1", "--gamma1",
                  "Time step control parameter gamma1 (decrease factor).");
   args.AddOption(&gamma2, "-g2", "--gamma2",
                  "Time step control parameter gamma2 (increase factor).");
               
   args.AddOption(&Bmag, "-Bmag", "--B-magnitude", 
                  "Initial coefficient for B magnitude in MHD blast testcase. ");
                  
   args.AddOption(&debug, "-debug", "--debug-mode", "-nodebug", "--no-debug-mode",
                  "Enable debug mode (print additional information).");
   args.Parse();
   if (!args.Good())
   {
      if (Mpi::Root()) { args.PrintUsage(cout); }
      return 1;
   }
   if (Mpi::Root()) { args.PrintOptions(cout); }
   
   if(debug)
   {
      if(Mpi::Root())
      {
         printf("Debug mode on. Root PID: %d \n", getpid());
         printf("Press Enter to continue...");
         std::cin.ignore();
      }
      MPI_Barrier(MPI_COMM_WORLD);
   }
   
   SetPeriodicShearParameters(shear_amplitude, shear_boost, shear_boost_x,
                              shear_periodic_x);
   if (problem == PERIODIC_BOX) { SetPeriodicBoxIC(box_ic_file); }
   ProblemData *pd = GetProblemData(problem, dim, Bmag);
   GetPlotTime(plot_time_file, plot_time);

   // Generate the mesh.
   Mesh *mesh;
   if (strncmp(mesh_file, "default", 7) != 0)
   {
      mesh = new Mesh(mesh_file, true, true);
   }
   else
   {
      MFEM_ABORT("Default mesh is not supported!\n"
                  "Please provide a mesh file with the option -m or --mesh.");
   }
   dim = mesh->Dimension();
   for (int lev = 0; lev < rs_levels; lev++) { mesh->UniformRefinement(); }
   if (Mpi::Root())
   {
      cout << "Number of zones in the serial mesh: " << mesh->GetNE() << endl;
   }
   
   // periodic mesh setup
   if(pd->periodic)
   {
      std::vector<Vector> translations;
      if(dim == 2)
      {
         if(pd->px_length > 0) translations.push_back(Vector({pd->px_length, 0.0}));
         if(pd->py_length > 0) translations.push_back(Vector({0.0, pd->py_length}));
      }
      else if(dim == 3)
      {
         if(pd->px_length > 0) translations.push_back(Vector({pd->px_length, 0.0, 0.0}));
         if(pd->py_length > 0) translations.push_back(Vector({0.0, pd->py_length, 0.0}));
         if(pd->pz_length > 0) translations.push_back(Vector({0.0, 0.0, pd->pz_length}));
      }
      Mesh * periodic_mesh = new Mesh(Mesh::MakePeriodic(*mesh, mesh->CreatePeriodicVertexMapping(translations)));
      delete mesh;
      mesh = periodic_mesh;
   }
   
   // Set the mesh curvature.
   mesh->SetCurvature(order_v, pd->periodic, -1, Ordering::byNODES);
   
   // Sanity checks
   if (dim == 1) MFEM_ABORT("1D is not supported!");
   if(dim!=3 && remap_type_A == RemapType::HelicityPreserving)
   {
      if(Mpi::Root())
      {
         cout << "Warning: Helicity preserving remap is only for 3D problems. Switch to DG convection." << endl;
      }
      remap_type_A = RemapType::DGconvection;
   }
   int A_dim = dim==2 ? 1 : 3;
   if(pd->periodic)
   {
      if(ale && mesh_smooth_type != MeshSmoothType::PERIODIC &&
         mesh_smooth_type != MeshSmoothType::INITIAL)
      {
         MFEM_ABORT("Periodic ALE requires the initial (-mst 1) or periodic (-mst 3) mesh smoother.");
      }
      MFEM_VERIFY(!ale || pd->px_length <= 0.0 ||
                  mesh_smooth_type == MeshSmoothType::INITIAL,
                  "The periodic smoother assumes x walls; use -mst 1 when x is periodic.");
   }

   // Parallel partitioning of the mesh.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   pmesh->PrintInfo();
   
   // Define the parallel finite element spaces. We use:
   // - H1 (Gauss-Lobatto, continuous) for position.
   // - L2 (Gauss-Lobatto, discontinuous) for position in the periodic case.
   // - H1 (Gauss-Lobatto, continuous) for velocity.
   // - L2 (Bernstein, discontinuous) for specific internal energy.
   // - L2 (Bernstein, discontinuous, integral-preserving map) for density. 
   // - H1 (2D case) / ND (3D case) for the magnetic vector potential. 
   // - RT for magnetic flux density. 
   L2_FECollection fec_rho(order_rho, dim, BasisType::Positive, FiniteElement::MapType::INTEGRAL);
   L2_FECollection fec_e(order_e, dim, BasisType::Positive);
   H1_FECollection fec_v(order_v, dim);
   FiniteElementCollection *fec_x = nullptr;
   if(pd->periodic)
   {
      fec_x = new L2_FECollection(order_v, dim, BasisType::GaussLobatto);
   }
   else
   {
      fec_x = new H1_FECollection(order_v, dim);
   }
   H1_FECollection phi_fec(order_A, dim); // when dim = 3, phi is needed for the remap (really?)
   FiniteElementCollection *fec_A;
   if(A_dim==1)
   {
      fec_A = new H1_FECollection(order_A, dim);
   }
   else if (A_dim==3)
   {
      fec_A = new ND_FECollection(order_A, dim);
   }
   RT_FECollection fec_B(order_A-1, dim);
   L2_FECollection fec_divB(order_A-1, dim); // for divB cleaning if needed
   
   ParFiniteElementSpace e_FESpace(pmesh, &fec_e);
   ParFiniteElementSpace rho_FESpace(pmesh, &fec_rho);
   ParFiniteElementSpace v_FESpace(pmesh, &fec_v, pmesh->Dimension());
   ParFiniteElementSpace x_FESpace(pmesh, fec_x, pmesh->Dimension());
   ParFiniteElementSpace A_FESpace(pmesh, fec_A);
   ParFiniteElementSpace B_FESpace(pmesh, &fec_B);
   ParFiniteElementSpace divB_FESpace(pmesh, &fec_divB);

   // Boundary conditions
   Array<int> ess_tdofs, ess_vdofs;
   {
      Array<int> ess_bdr, dofs_marker, dofs_list;
      for (int d = 0; d < pmesh->Dimension(); d++)
      {
         ess_bdr = pd->ess_bdrs_v[d];
         v_FESpace.GetEssentialTrueDofs(ess_bdr, dofs_list, d);
         ess_tdofs.Append(dofs_list);
         v_FESpace.GetEssentialVDofs(ess_bdr, dofs_marker, d);
         FiniteElementSpace::MarkerToList(dofs_marker, dofs_list);
         ess_vdofs.Append(dofs_list);
      }
   }

   // Define the explicit ODE solver used for time integration.
   ODESolver *ode_solver = NULL;
   switch (ode_solver_type)
   {
      case 1: ode_solver = new ForwardEulerSolver; break;
      case 2: ode_solver = new RK2Solver(0.5); break;
      case 3: ode_solver = new RK3SSPSolver; break;
      case 4: ode_solver = new RK4Solver; break;
      case 6: ode_solver = new RK6Solver; break;
      default:
         if (myid == 0)
         {
            cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         }
         delete pmesh;
         MPI_Finalize();
         return 3;
   }

   const HYPRE_BigInt glob_size_rho = rho_FESpace.GlobalTrueVSize();
   const HYPRE_BigInt glob_size_e = e_FESpace.GlobalTrueVSize();
   const HYPRE_BigInt glob_size_v = v_FESpace.GlobalTrueVSize();
   const HYPRE_BigInt glob_size_A = A_FESpace.GlobalTrueVSize();
   const HYPRE_BigInt glob_size_B = B_FESpace.GlobalTrueVSize();
   if (Mpi::Root())
   {
      cout << "Number of velocity dofs: "
           << glob_size_v << endl;
      cout << "Number of density dofs: "
           << glob_size_rho << endl;
      cout << "Number of specific internal energy dofs: "
           << glob_size_e << endl;
      cout << "Number of magnetic vector potential dofs: "
           << glob_size_A << endl;
      cout << "Number of magnetic flux density dofs: "
           << glob_size_B << endl;
   }

   // The monolithic BlockVector stores unknown fields as:
   // - 0 -> position
   // - 1 -> velocity
   // - 2 -> specific internal energy
   const int Vsize_e = e_FESpace.GetVSize();
   const int Vsize_x = x_FESpace.GetVSize();
   const int Vsize_v = v_FESpace.GetVSize();
   Array<int> offset(4);
   offset[0] = 0;
   offset[1] = offset[0] + Vsize_x;
   offset[2] = offset[1] + Vsize_v;
   offset[3] = offset[2] + Vsize_e;
   BlockVector S(offset);

   // Define GridFunction objects for the position, velocity and specific
   // internal energy. There is no function for the density, as we can always
   // compute the density values given the current mesh position, using the
   // property of pointwise mass conservation.
   ParGridFunction x_gf, v_gf, e_gf;
   x_gf.MakeRef(&x_FESpace, S, offset[0]);
   v_gf.MakeRef(&v_FESpace, S, offset[1]);
   e_gf.MakeRef(&e_FESpace, S, offset[2]);

   // Initialize x_gf using the starting mesh coordinates.
   pmesh->SetNodalGridFunction(&x_gf);
   
   // Initialize the velocity.
   VectorFunctionCoefficient v_coeff(pmesh->Dimension(), pd->v0);
   v_gf.ProjectCoefficient(v_coeff);

   // Initialize density and specific internal energy values. 
   ParGridFunction rho_gf(&rho_FESpace);
   if(problem == MHD_BLAST)
   {
      DeltaCoefficient e_coeff(0.0, 0.0, 0.0, 1.0);
      e_gf.ProjectCoefficient(e_coeff);
   }
   else
   {
      FunctionCoefficient e_coeff(pd->e0);
      L2_FECollection l2_fec(order_e, pmesh->Dimension());
      ParFiniteElementSpace l2_fes(pmesh, &l2_fec);
      ParGridFunction l2_e(&l2_fes);
      l2_e.ProjectCoefficient(e_coeff);
      e_gf.ProjectGridFunction(l2_e);
   }
   
   // Initialize the magnetic vector potential and the magnetic flux density.
   VectorFunctionCoefficient A_coeff(A_dim, pd->A0);
   VectorFunctionCoefficient B_coeff(dim, pd->B0);
   ParGridFunction A_gf(&A_FESpace);
   A_gf.ProjectCoefficient(A_coeff);
   ParGridFunction B_gf(&B_FESpace);
   ComputeCurl(A_gf, B_gf);
   if(pd->periodic)
   {
      ParGridFunction B0_gf(&B_FESpace);
      B0_gf = 0.0;
      DivFreeProject(B0_gf, &divB_FESpace, *pd->B0_periodic);
      B_gf += B0_gf;
   }
   
   VectorFunctionCoefficient H_bdry_coeff(dim, pd->H_bdry_func);

   // Piecewise constant ideal gas coefficient over the Lagrangian mesh. The
   // gamma values are projected on function that's constant on the moving mesh.
   L2_FECollection mat_fec(0, pmesh->Dimension());
   ParFiniteElementSpace mat_fes(pmesh, &mat_fec);
   ParGridFunction mat_gf(&mat_fes);
   FunctionCoefficient mat_coeff(pd->gamma_func);
   mat_gf.ProjectCoefficient(mat_coeff);
      
   if (impose_visc) { pd->visc = true; }

   hydrodynamics::LagrangianHydroOperator hydro(S.Size(),
                                                pmesh,
                                                x_FESpace,
                                                v_FESpace, 
                                                e_FESpace, 
                                                rho_FESpace, 
                                                ess_tdofs,
                                                rho_gf,
                                                mat_gf, cfl,
                                                pd->visc, pd->vorticity,
                                                cg_tol, cg_max_iter,
                                                order_q,
                                                A_gf, B_gf, divB_FESpace,
                                                pd->mu, H_bdry_coeff, pd->ess_bdr_H, mesh_smooth_type, smooth_eps, pd, fix_step_remesh, fix_step_remesh_interval);
                                                
   hydro.SetVelocityBoundaryValue(v_gf);
                           
   hydro.SetRemeshParameters(min_detJ, max_detJ, max_ratio, max_displacement);
   
   hydro.SetPreserveMeanField(preserve_mean_field);
   hydro.SetComovingRezone(comoving_rezone);
   hydro.SetRemapType_v(remap_type_v);
   hydro.SetRemapType_e(remap_type_e);
   hydro.SetRemapType_A(remap_type_A);
   hydro.SetRemapType_rho(remap_type_rho, bp_type);

   socketstream vis_rho, vis_v, vis_e, vis_B, vis_A;
   char vishost[] = "localhost";
   int  visport   = 19916;

   const double energy_init = hydro.InternalEnergy(e_gf) +
                              hydro.KineticEnergy(v_gf);

   if (visualization || paraview)
   {
      Visualization(visualization, paraview, pmesh, rho_gf, v_gf, e_gf, B_gf, A_gf, 0, 0.0,
         outputdir, paraview_basename);     
   }

   // Perform time-integration (looping over the time iterations, ti, with a
   // time-step dt). The object oper is of type LagrangianHydroOperator that
   // defines the Mult() method that used by the time integrators.
   ode_solver->Init(hydro);
   hydro.ResetTimeStepEstimate();
   double t = 0.0, dt = hydro.GetTimeStepEstimate(S), t_old;
   bool last_step = false;
   int steps = 0;
   BlockVector S_old(S);
   long mem=0, mmax=0, msum=0;
   const double internal_energy = hydro.InternalEnergy(e_gf);
   const double kinetic_energy = hydro.KineticEnergy(v_gf);
      
   // output divergence error of B
   char B_div_err_file[filename_length];
   sprintf(B_div_err_file, "%s/B_divergence_error.dat", outputdir);
   std::ofstream B_div_err_out;
   if (Mpi::Root()) { B_div_err_out.open(B_div_err_file); }
   real_t B_err = GFDivError(B_gf);
   B_div_err_out << std::scientific << std::setprecision(16)
                     << t << " " << B_err << endl;
                     
   // output minimum value of rho
   char rho_min_file[filename_length];
   sprintf(rho_min_file, "%s/rho_min.dat", outputdir);
   std::ofstream rho_min_out;
   if (Mpi::Root()) { rho_min_out.open(rho_min_file); }
   real_t rho_min, rho_max; 
   GFMinMax(rho_gf, rho_min, rho_max);
   rho_min_out << std::scientific << std::setprecision(16)
                << t << " " << rho_min << endl;
                
   // output helicity 
   char helicity_file[filename_length];
   sprintf(helicity_file, "%s/helicity.dat", outputdir);
   std::ofstream helicity_out;
   if (Mpi::Root()) { helicity_out.open(helicity_file); }
   real_t helicity = 0.0;
   if(dim == 3) helicity = GFInnerProduct(B_gf, A_gf);
   helicity_out << std::scientific << std::setprecision(16)
                 << t << " " << helicity << endl;
                 
   // output time step
   char time_step_file[filename_length];
   sprintf(time_step_file, "%s/time_step.dat", outputdir);
   std::ofstream time_step_out;
   if (Mpi::Root()) { time_step_out.open(time_step_file); }
   
   // output ALE
   char ale_file[filename_length];
   sprintf(ale_file, "%s/ale.dat", outputdir);
   std::ofstream ale_out;
   if (Mpi::Root()) { ale_out.open(ale_file); }
   
   // Integrated diagnostics execute collectively; only root opens the file.
   std::ofstream magnetic_audit;
   if (magnetic_audit_enabled && Mpi::Root())
   {
      magnetic_audit.open(std::string(outputdir) + "/magnetic_audit.csv");
      magnetic_audit << "time,phase,magnetic_energy,total_energy,mean_Bx,mean_By,mean_Bz\n";
   }
   auto audit_magnetic = [&](real_t audit_time, const char *phase)
   {
      if (!magnetic_audit_enabled) { return; }
      real_t eb = GFInnerProduct(B_gf, B_gf)/(2.0*pd->mu);
      real_t et = eb + hydro.InternalEnergy(e_gf) + hydro.KineticEnergy(v_gf);
      Vector local_mean(3), global_mean(3), field(dim);
      local_mean = 0.0;
      real_t local_volume = 0.0, global_volume;
      for (int e = 0; e < pmesh->GetNE(); ++e)
      {
         auto *T = pmesh->GetElementTransformation(e);
         const auto &quad = hydro.GetIntegrationRule();
         for (int q = 0; q < quad.GetNPoints(); ++q)
         {
            const auto &ip = quad.IntPoint(q);
            T->SetIntPoint(&ip);
            B_gf.GetVectorValue(*T, ip, field);
            real_t weight = ip.weight*T->Weight();
            local_volume += weight;
            for (int d = 0; d < dim; ++d) { local_mean(d) += weight*field(d); }
         }
      }
      MPI_Allreduce(local_mean.GetData(), global_mean.GetData(), 3, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());
      MPI_Allreduce(&local_volume, &global_volume, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());
      global_mean /= global_volume;
      if (Mpi::Root())
      {
         magnetic_audit << std::scientific << std::setprecision(16)
                        << audit_time << "," << phase << "," << eb << "," << et
                        << "," << global_mean(0) << "," << global_mean(1) << "," << global_mean(2) << std::endl;
      }
   };
   audit_magnetic(t, "initial");
   PrintError(pd, t, rho_gf, v_gf, e_gf, B_gf);

   for (int ti = 1; !last_step; ti++)
   {
      if (t + dt >= t_final)
      {
         dt = t_final - t;
         last_step = true;
      }
      if (steps == max_tsteps) { last_step = true; }
      S_old = S;
      t_old = t;
      hydro.ResetTimeStepEstimate();

      ode_solver->Step(S, t, dt);
      steps++;

      // Adaptive time step control.
      const double dt_est = hydro.GetTimeStepEstimate(S);
      if (dt_est < dt)
      {
         if(Mpi::Root()) {
            cout << "dt estimate: " << dt_est << " < dt: " << dt << endl;
         }
         
         // Repeat (solve again) with a decreased time step - decrease of the
         // time estimate suggests appearance of oscillations.
         dt *= gamma1;
         if (dt < std::numeric_limits<double>::epsilon())
         { MFEM_ABORT("The time step crashed!"); }
         t = t_old;
         S = S_old;
         if (Mpi::Root()) { 
            cout << "Repeating step " << ti << " with decreased dt = " << dt << endl;
         }
         if (steps < max_tsteps) { last_step = false; }
         ti--; continue;
      }
      else
      {
         time_step_out << std::scientific << std::setprecision(16)
                          << t << " " << dt << endl;
         if (dt_est > 1.25 * dt) { dt *= gamma2; }
      }

      // Ensure the sub-vectors x_gf, v_gf, and e_gf know the location of the
      // data in S. This operation simply updates the Memory validity flags of
      // the sub-vectors to match those of S.
      x_gf.SyncAliasMemory(S);
      v_gf.SyncAliasMemory(S);
      e_gf.SyncAliasMemory(S);

      // Make sure that the mesh corresponds to the new solution state. This is
      // needed, because some time integrators use different S-type vectors
      // and the oper object might have redirected the mesh positions to those.
      pmesh->NewNodes(x_gf, false);
      
      if (Mpi::Root())
      {
         cout << std::fixed;
         cout << "step "  << ti
               << ",\tt = " << std::setprecision(16) << t
               << ",\tdt = " << std::setprecision(16) << dt
               ;
         cout << std::fixed;
         cout << endl;
      }
      PrintError(pd, t, rho_gf, v_gf, e_gf, B_gf);
      
      // ALE remap
      bool need_remesh = false;
      if(ale){
         need_remesh = hydro.NeedRemesh();
         if(last_step) need_remesh = remap_at_last_step;
         if(need_remesh)
         {
            if(((last_step && remap_at_last_step) || plot_at_remap) && (visualization || paraview))
            {
               Visualization(visualization, paraview, pmesh, rho_gf, v_gf, e_gf, B_gf, A_gf, ti, t,
                             outputdir, paraview_basename);
               if(last_step) ti++;
            }
            audit_magnetic(t, "pre_remap");
            hydro.RemeshAndRemap(S);
            audit_magnetic(t, "post_remap");
            if(plot_at_remap && (visualization || paraview))
            {
               Visualization(visualization, paraview, pmesh, rho_gf, v_gf, e_gf, B_gf, A_gf, ++ti, t+100,
                             outputdir, paraview_basename);
            }
            PrintError(pd, t, rho_gf, v_gf, e_gf, B_gf);
         }
      } 
      ale_out << std::scientific << std::setprecision(16)
                << t << " " << need_remesh << endl;
      
      audit_magnetic(t, "accepted_step");
      // output divergence error of B
      B_err = GFDivError(B_gf);
      B_div_err_out << std::scientific << std::setprecision(16)
                     << t << " " << B_err << endl;
                     
      GFMinMax(rho_gf, rho_min, rho_max, &hydro.GetIntegrationRule());
      rho_min_out << std::scientific << std::setprecision(16)
                   << t << " " << rho_min << endl;
                   
      if(dim==3) helicity = GFInnerProduct(B_gf, A_gf);
      helicity_out << std::scientific << std::setprecision(16)
                    << t << " " << helicity << endl;

      if (last_step || CheckPlot(plot_time, t))
      {
         // Make sure all ranks have sent their 'v' solution before initiating
         // another set of GLVis connections (one from each rank):
         MPI_Barrier(pmesh->GetComm());

         if (visualization || paraview)
         {
            Visualization(visualization, paraview, pmesh, rho_gf, v_gf, e_gf, B_gf, A_gf, ti, t,
                         outputdir, paraview_basename);
         }
      }
            
   }

   switch (ode_solver_type)
   {
      case 2: steps *= 2; break;
      case 3: steps *= 3; break;
      case 4: steps *= 4; break;
      case 6: steps *= 6; break;
      case 7: steps *= 2;
   }

   const double energy_final = hydro.InternalEnergy(e_gf) +
                               hydro.KineticEnergy(v_gf);

   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i=0; i < Geometry::NumGeom; ++i)
   {
      if(i==pmesh->GetElementBaseGeometry(0))
      {
         irs[i] = &hydro.GetIntegrationRule();
      }
      else
      {
         irs[i] = nullptr;
      }
   }
   PrintError(pd, t, rho_gf, v_gf, e_gf, B_gf, irs);
   
   B_div_err_out.close();
   rho_min_out.close();
   helicity_out.close();
   time_step_out.close();
   ale_out.close();

   // Free the used memory.
   delete fec_A;
   delete ode_solver;
   delete pmesh;
   delete fec_x;

   return 0;
}

static void Visualization(bool vis, bool paraview, 
                           ParMesh *pmesh, ParGridFunction &rho_gf, 
                          ParGridFunction &v_gf, ParGridFunction &e_gf,
                          ParGridFunction &B_gf, ParGridFunction &A_gf, 
                          int ti, real_t t,
                          const char *outputdir,
                          const char *paraview_basename)
{
   
   static int count = 0;
   count++;
   
   MPI_Barrier(pmesh->GetComm());
   
   static socketstream vis_rho, vis_v, vis_e, vis_B, vis_A;
   char vishost[] = "localhost";
   int  visport   = 19916;
      
   static ParaViewDataCollection paraview_dc(paraview_basename, pmesh);
   if(paraview && count == 1)
   {
      paraview_dc.SetPrefixPath(outputdir);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.RegisterField("Density",  &rho_gf);
      paraview_dc.RegisterField("Velocity", &v_gf);
      paraview_dc.RegisterField("Specific Internal Energy", &e_gf);
      paraview_dc.RegisterField("Magnetic Vector Potential", &A_gf);
      paraview_dc.RegisterField("Magnetic Flux Density", &B_gf);
      paraview_dc.SetCycle(0);
      paraview_dc.SetTime(0.0);
      paraview_dc.Save();
   }
   
   bool gfprint = true;
   if(gfprint && count == 1)
   {
      
      char parentdirname[filename_length];
      sprintf(parentdirname, "%s/gfdata", outputdir, 0);
      if (Mpi::Root()) { cout << "Saving GLVis data in: " << parentdirname << endl; }
      mkdir(parentdirname, 0777);
      
      char dirname[filename_length];
      sprintf(dirname, "%s/Cycle_%08d", parentdirname, 0);
      mkdir(dirname, 0777);
      
      char mesh_filename[filename_length];
      sprintf(mesh_filename, "%s/mesh", dirname);
      {
         std::ofstream mesh_ofs(mesh_filename);
         mesh_ofs.precision(8);
         pmesh->PrintAsSerial(mesh_ofs);
      }
      
      char rho_filename[filename_length];
      sprintf(rho_filename, "%s/rho", dirname);
      rho_gf.SaveAsSerial(rho_filename);
      
      char v_filename[filename_length];
      sprintf(v_filename, "%s/v", dirname);
      v_gf.SaveAsSerial(v_filename);
      
      char e_filename[filename_length];
      sprintf(e_filename, "%s/e", dirname);
      e_gf.SaveAsSerial(e_filename);
      
      char A_filename[filename_length];
      sprintf(A_filename, "%s/A", dirname);
      A_gf.SaveAsSerial(A_filename);
      
      char B_filename[filename_length];
      sprintf(B_filename, "%s/B", dirname);
      B_gf.SaveAsSerial(B_filename);
   }
   
   if(paraview && count > 1)
   {
      paraview_dc.SetCycle(ti);
      paraview_dc.SetTime(t);
      paraview_dc.Save();
   }
   
   if(gfprint && count > 1)
   {
      
      char parentdirname[filename_length];
      sprintf(parentdirname, "%s/gfdata", outputdir, 0);
      
      char dirname[filename_length];
      sprintf(dirname, "%s/Cycle_%08d", parentdirname, ti);
      mkdir(dirname, 0777);
      
      char mesh_filename[filename_length];
      sprintf(mesh_filename, "%s/mesh", dirname);
      {
         std::ofstream mesh_ofs(mesh_filename);
         mesh_ofs.precision(8);
         pmesh->PrintAsSerial(mesh_ofs);
      }
      
      char rho_filename[filename_length];
      sprintf(rho_filename, "%s/rho", dirname);
      rho_gf.SaveAsSerial(rho_filename);
      
      char v_filename[filename_length];
      sprintf(v_filename, "%s/v", dirname);
      v_gf.SaveAsSerial(v_filename);
      
      char e_filename[filename_length];
      sprintf(e_filename, "%s/e", dirname);
      e_gf.SaveAsSerial(e_filename);
      
      char A_filename[filename_length];
      sprintf(A_filename, "%s/A", dirname);
      A_gf.SaveAsSerial(A_filename);
      
      char B_filename[filename_length];
      sprintf(B_filename, "%s/B", dirname);
      B_gf.SaveAsSerial(B_filename);
   }
   
   if(vis)
   {
      vis_rho.precision(8);
      vis_v.precision(8);
      vis_e.precision(8);
      vis_B.precision(8);
      vis_A.precision(8);
      
      int Wx = 0, Wy = 0; // window position
      int Ww = 350, Wh = 350; // window size
      int offx = Ww+10; // window offsets
      hydrodynamics::VisualizeField(vis_rho, vishost, visport, rho_gf,
                                       "Density", Wx, Wy, Ww, Wh);
      Wx += offx;
      hydrodynamics::VisualizeField(vis_v, vishost, visport,
                                    v_gf, "Velocity", Wx, Wy, Ww, Wh);
      Wx += offx;
      hydrodynamics::VisualizeField(vis_e, vishost, visport, e_gf,
                                    "Specific Internal Energy",
                                    Wx, Wy, Ww,Wh);
      Wx += offx;
      hydrodynamics::VisualizeField(vis_B, vishost, visport, B_gf,
                                    "Magnetic Flux Density", Wx, Wy, Ww, Wh);
      Wx += offx;
      hydrodynamics::VisualizeField(vis_A, vishost, visport, A_gf,
                                    "Magnetic vector potential", Wx, Wy, Ww, Wh);
      Wx += offx;                    
   }
   
   
   
}
