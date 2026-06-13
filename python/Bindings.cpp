#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "ForwardModel.hpp"
#include "ParameterSensitivity.hpp"
#include "CompressibleForwardModel.hpp"
#include "IdealGasEOS.hpp"
#include "CompressibleBCs.hpp"

namespace py = pybind11;

// Helper: extract cell centers from Mesh as numpy array (n_cells, 3)
static py::array_t<double> meshCellCenters(const Mesh& mesh) {
    int nc = mesh.nCells();
    py::array_t<double> arr({nc, 3});
    auto r = arr.mutable_unchecked<2>();
    for (int i = 0; i < nc; ++i) {
        const Vec3& c = mesh.cell(i).center;
        r(i, 0) = c.x;  r(i, 1) = c.y;  r(i, 2) = c.z;
    }
    return arr;
}

// Helper: extract cell volumes from Mesh as numpy array (n_cells,)
static py::array_t<double> meshCellVolumes(const Mesh& mesh) {
    int nc = mesh.nCells();
    py::array_t<double> arr(nc);
    auto r = arr.mutable_unchecked<1>();
    for (int i = 0; i < nc; ++i)
        r(i) = mesh.cell(i).volume;
    return arr;
}

// Helper: extract node coordinates from Mesh as numpy array (n_nodes, 3)
static py::array_t<double> meshNodeCoords(const Mesh& mesh) {
    int nn = mesh.nNodes();
    py::array_t<double> arr({nn, 3});
    auto r = arr.mutable_unchecked<2>();
    for (int i = 0; i < nn; ++i) {
        const Vec3& n = mesh.node(i);
        r(i, 0) = n.x;  r(i, 1) = n.y;  r(i, 2) = n.z;
    }
    return arr;
}

// Helper: extract face centers from Mesh as numpy array (n_faces, 3)
// PHASE 7 — patch introspection helpers for wall-function diagnostics.
// We don't bind the full Patch / Face structs (too much surface area); we
// just expose enough per-wall-face data to compute y+ and Cf in Python.

static std::vector<std::string> meshPatchNames(const Mesh& mesh) {
    std::vector<std::string> out;
    out.reserve(mesh.nPatches());
    for (int pi = 0; pi < mesh.nPatches(); ++pi)
        out.push_back(mesh.patch(pi).name);
    return out;
}

static std::vector<std::string> meshPatchTypes(const Mesh& mesh) {
    std::vector<std::string> out;
    out.reserve(mesh.nPatches());
    for (int pi = 0; pi < mesh.nPatches(); ++pi)
        out.push_back(mesh.patch(pi).type);
    return out;
}

static py::dict meshWallPatchData(const Mesh& mesh, const std::string& name) {
    int patchIdx = -1;
    for (int pi = 0; pi < mesh.nPatches(); ++pi) {
        if (mesh.patch(pi).name == name) { patchIdx = pi; break; }
    }
    if (patchIdx < 0)
        throw std::runtime_error("Mesh::wall_patch_data: unknown patch '" + name + "'");
    const Patch& pat = mesh.patch(patchIdx);
    int nf = static_cast<int>(pat.faces.size());

    py::array_t<int>    owner_arr(nf);
    py::array_t<double> delta_arr(nf);
    py::array_t<double> center_arr({nf, 3});
    py::array_t<double> normal_arr({nf, 3});
    py::array_t<double> area_arr(nf);
    auto own = owner_arr.mutable_unchecked<1>();
    auto del = delta_arr.mutable_unchecked<1>();
    auto cen = center_arr.mutable_unchecked<2>();
    auto nor = normal_arr.mutable_unchecked<2>();
    auto are = area_arr.mutable_unchecked<1>();

    for (int k = 0; k < nf; ++k) {
        const Face& face = mesh.face(pat.faces[k]);
        own(k)    = face.owner;
        del(k)    = face.delta;
        cen(k, 0) = face.center.x;
        cen(k, 1) = face.center.y;
        cen(k, 2) = face.center.z;
        nor(k, 0) = face.normal.x;
        nor(k, 1) = face.normal.y;
        nor(k, 2) = face.normal.z;
        are(k)    = face.area;
    }

    py::dict d;
    d["name"]    = pat.name;
    d["type"]    = pat.type;
    d["n_faces"] = nf;
    d["owner"]   = owner_arr;
    d["delta"]   = delta_arr;
    d["center"]  = center_arr;
    d["normal"]  = normal_arr;
    d["area"]    = area_arr;
    return d;
}

static py::array_t<double> meshFaceCenters(const Mesh& mesh) {
    int nf = mesh.nFaces();
    py::array_t<double> arr({nf, 3});
    auto r = arr.mutable_unchecked<2>();
    for (int i = 0; i < nf; ++i) {
        const Vec3& c = mesh.face(i).center;
        r(i, 0) = c.x;  r(i, 1) = c.y;  r(i, 2) = c.z;
    }
    return arr;
}

// Helper: extract scalar field as numpy array (n_cells,)
static py::array_t<double> scalarToNumpy(const ScalarField& f) {
    int n = f.size();
    py::array_t<double> arr(n);
    auto r = arr.mutable_unchecked<1>();
    for (int i = 0; i < n; ++i)
        r(i) = f[i];
    return arr;
}

// Helper: extract vector field as numpy array (n_cells, 3)
static py::array_t<double> vectorToNumpy(const VectorField& f) {
    int n = f.size();
    py::array_t<double> arr({n, 3});
    auto r = arr.mutable_unchecked<2>();
    for (int i = 0; i < n; ++i) {
        r(i, 0) = f[i].x;  r(i, 1) = f[i].y;  r(i, 2) = f[i].z;
    }
    return arr;
}

// Helper: extract all fields from ForwardModel's last solution as a dict of numpy arrays
static py::dict extractFields(const ForwardModel& fm) {
    if (!fm.hasLastFields())
        throw std::runtime_error("No fields available — call evaluate() first");
    const FlowFields& ff = fm.lastFields();
    py::dict d;
    d["U"]     = vectorToNumpy(ff.U);
    d["p"]     = scalarToNumpy(ff.p);
    d["k"]     = scalarToNumpy(ff.k);
    d["omega"] = scalarToNumpy(ff.omega);
    d["nuT"]   = scalarToNumpy(ff.nuT);
    d["F1"]    = scalarToNumpy(ff.F1);
    d["F2"]    = scalarToNumpy(ff.F2);
    d["Pk"]    = scalarToNumpy(ff.Pk);
    return d;
}

// Helper: extract fields from CompressibleForwardModel as a dict.
// Includes density and temperature on top of the incompressible field set so
// the existing visualization/observation tooling can reuse the keys.
static py::dict extractCompressibleFields(const CompressibleForwardModel& fm) {
    if (!fm.hasLastFields())
        throw std::runtime_error("No fields available — call evaluate() first");
    const CompressibleFlowFields& ff = fm.lastCompressibleFields();
    py::dict d;
    d["U"]     = vectorToNumpy(ff.U);
    d["p"]     = scalarToNumpy(ff.p);
    d["T"]     = scalarToNumpy(ff.T);
    d["rho"]   = scalarToNumpy(ff.rho);
    d["k"]     = scalarToNumpy(ff.k);
    d["omega"] = scalarToNumpy(ff.omega);
    d["nuT"]   = scalarToNumpy(ff.nuT);
    d["F1"]    = scalarToNumpy(ff.F1);
    d["F2"]    = scalarToNumpy(ff.F2);
    d["Pk"]    = scalarToNumpy(ff.Pk);
    return d;
}

PYBIND11_MODULE(rans_sst_py, m) {
    m.doc() = "RANS-SST Bayesian Calibration – C++ Forward Model";
    // EvaluationStatus
    py::enum_<EvaluationStatus>(m, "EvaluationStatus")
        .value("Converged", EvaluationStatus::Converged)
        .value("Unconverged", EvaluationStatus::Unconverged)
        .value("DivergenceDetected", EvaluationStatus::Diverged)
        .value("InvalidParameters", EvaluationStatus::InvalidParameters)
        .value("Unknown", EvaluationStatus::Unknown);

    // EvaluationResult
    py::class_<EvaluationResult>(m, "EvaluationResult")
        .def_readonly("status", &EvaluationResult::status)
        .def_readonly("log_lik", &EvaluationResult::loglik)
        .def_readonly("predictions", &EvaluationResult::predictions)
        .def_readonly("simple_iters", &EvaluationResult::simpleIters);

    // SSTCoefficients
    py::class_<SSTCoefficients>(m, "SSTCoefficients")
        .def(py::init<>())
        .def_readwrite("sigma_k1", &SSTCoefficients::sigma_k1)
        .def_readwrite("sigma_w1", &SSTCoefficients::sigma_w1)
        .def_readwrite("beta1", &SSTCoefficients::beta1)
        .def_readwrite("alpha1", &SSTCoefficients::alpha1)
        .def_readwrite("sigma_k2", &SSTCoefficients::sigma_k2)
        .def_readwrite("sigma_w2", &SSTCoefficients::sigma_w2)
        .def_readwrite("beta2", &SSTCoefficients::beta2)
        .def_readwrite("alpha2", &SSTCoefficients::alpha2)
        .def_readwrite("betaStar", &SSTCoefficients::betaStar)
        .def_readwrite("a1", &SSTCoefficients::a1)
        .def_readwrite("kappa", &SSTCoefficients::kappa);

    // InferenceParameterSet
    py::class_<InferenceParameterSet>(m, "InferenceParameterSet")
        .def(py::init<>())
        .def_readonly("name", &InferenceParameterSet::name)
        .def("n_active", &InferenceParameterSet::nActive)
        .def("active_names", &InferenceParameterSet::activeNames)
        .def("pack", &InferenceParameterSet::pack)
        .def("unpack", &InferenceParameterSet::unpack)
        .def("in_bounds", &InferenceParameterSet::inBounds)
        .def("lower_bounds", &InferenceParameterSet::lowerBounds)
        .def("upper_bounds", &InferenceParameterSet::upperBounds)
        .def_static("a1_betaStar", &InferenceParameterSet::a1_betaStar)
        .def_static("inlet_turb", &InferenceParameterSet::inletTurb)
        .def_static("near_wall4", &InferenceParameterSet::nearWall4)
        .def_static("all11", &InferenceParameterSet::all11)
        .def_static("from_indices", &InferenceParameterSet::fromIndices,
                    py::arg("name"), py::arg("indices"));

    // Mesh
    py::class_<Mesh>(m, "Mesh")
        .def_static("make_channel_2d", py::overload_cast<int, int, double, double>(&Mesh::makeChannel2D),
             py::arg("nx"), py::arg("ny"), py::arg("Lx"), py::arg("Ly"))
        .def_static("make_channel_2d", py::overload_cast<int, int, double, double, double, double>(&Mesh::makeChannel2D),
             py::arg("nx"), py::arg("ny"), py::arg("Lx"), py::arg("Ly"),
             py::arg("Re"), py::arg("yPlusTarget") = 1.0)
        .def_static("make_backward_facing_step_2d",
             py::overload_cast<int,int,int,int,double,double,double,double>(
                 &Mesh::makeBackwardFacingStep2D),
             py::arg("nx_up"), py::arg("nx_down"), py::arg("ny_up"), py::arg("ny_down"),
             py::arg("Lu"), py::arg("Ld"), py::arg("h_s"), py::arg("H"))
        .def_static("make_backward_facing_step_2d",
             py::overload_cast<int,int,int,int,double,double,double,double,double,double>(
                 &Mesh::makeBackwardFacingStep2D),
             py::arg("nx_up"), py::arg("nx_down"), py::arg("ny_up"), py::arg("ny_down"),
             py::arg("Lu"), py::arg("Ld"), py::arg("h_s"), py::arg("H"),
             py::arg("Re"), py::arg("yPlusTarget") = 1.0)
        .def_static("load_from_file", &Mesh::loadFromFile)
        .def("compute_wall_distance", &Mesh::computeWallDistance)
        .def("n_cells", &Mesh::nCells)
        .def("n_faces", &Mesh::nFaces)
        .def("n_nodes", &Mesh::nNodes)
        .def("n_patches", &Mesh::nPatches)
        .def("n_internal_faces", &Mesh::nInternalFaces)
        // PHASE 7 patch introspection
        .def("patch_names",  &meshPatchNames)
        .def("patch_types",  &meshPatchTypes)
        .def("wall_patch_data", &meshWallPatchData, py::arg("name"))
        .def("cell_centers", &meshCellCenters)
        .def("cell_volumes", &meshCellVolumes)
        .def("node_coords", &meshNodeCoords)
        .def("face_centers", &meshFaceCenters);

    // FlowBoundaryConditions
    py::class_<FlowBoundaryConditions>(m, "FlowBoundaryConditions")
        .def(py::init<>())
        .def_static("channel_defaults", &FlowBoundaryConditions::channelDefaults)
        .def_static("flat_plate_defaults", &FlowBoundaryConditions::flatPlateDefaults)
        .def_static("bfs_defaults", &FlowBoundaryConditions::bfsDefaults);

    // SolverSettings
    py::class_<SolverSettings>(m, "SolverSettings")
        .def(py::init<>())
        .def_readwrite("max_iterations",    &SolverSettings::maxIterations)
        .def_readwrite("convergence_tol",   &SolverSettings::convergenceTol)
        .def_readwrite("divergence_limit",  &SolverSettings::divergenceLimit)
        .def_readwrite("alpha_u",           &SolverSettings::alphaU)
        .def_readwrite("alpha_p",           &SolverSettings::alphaP)
        .def_readwrite("alpha_k",           &SolverSettings::alphaK)
        .def_readwrite("alpha_omega",       &SolverSettings::alphaOmega)
        .def_readwrite("inner_iterations",  &SolverSettings::innerIterations)
        .def_readwrite("inner_tolerance",   &SolverSettings::innerTolerance)
        .def_readwrite("turb_start_iter",   &SolverSettings::turbStartIter)
        .def_readwrite("turb_update_interval", &SolverSettings::turbUpdateInterval)
        .def_readwrite("k_min",             &SolverSettings::kMin)
        .def_readwrite("omega_min",         &SolverSettings::omegaMin)
        .def_readwrite("verbose",           &SolverSettings::verbose)
        .def_readwrite("report_interval",   &SolverSettings::reportInterval)
        .def_readwrite("alpha_t",           &SolverSettings::alphaT)
        // PHASE 7 — wall functions / coarse-mesh mode
        .def_readwrite("use_wall_functions", &SolverSettings::useWallFunctions)
        .def_readwrite("von_karman",         &SolverSettings::vonKarman)
        .def_readwrite("wall_fn_E",          &SolverSettings::wallFnE)
        // PHASE 3 — closure-structure toggle: 0=Full, 1=NoLimiter, 2=KOmega
        .def_readwrite("sst_variant",        &SolverSettings::sstVariant);

    // ObservationOperator
    py::class_<ObservationOperator>(m, "ObservationOperator")
        .def(py::init<>())
        .def("add_drag", &ObservationOperator::addDrag,
             py::arg("wall_patch"), py::arg("cd_obs"), py::arg("sigma"),
             py::arg("ref_area"), py::arg("ref_vel"), py::arg("sigma_model") = 0.0)
        .def("add_skin_friction", &ObservationOperator::addSkinFriction,
             py::arg("wall_patch"), py::arg("location"), py::arg("cf_obs"),
             py::arg("sigma"), py::arg("ref_vel"), py::arg("sigma_model") = 0.0)
        .def("add_velocity_profile", &ObservationOperator::addVelocityProfile,
             py::arg("location"), py::arg("component"), py::arg("u_obs"),
             py::arg("sigma"), py::arg("sigma_model") = 0.0)
        .def("add_reattachment_length", &ObservationOperator::addReattachmentLength,
             py::arg("wall_patch"), py::arg("xr_obs"), py::arg("sigma"),
             py::arg("sigma_model") = 0.0)
        .def("n_obs", &ObservationOperator::nObs);

    // Vec3 (for location arguments)
    py::class_<Vec3>(m, "Vec3")
        .def(py::init<double, double, double>())
        .def_readwrite("x", &Vec3::x)
        .def_readwrite("y", &Vec3::y)
        .def_readwrite("z", &Vec3::z);

    // ForwardModel
    py::class_<ForwardModel>(m, "ForwardModel")
        .def(py::init<const Mesh&, const InferenceParameterSet&,
                       const ObservationOperator&, const FlowBoundaryConditions&,
                       double, const SolverSettings&, const Vec3&,
                       double, double, double>(),
             py::arg("mesh"), py::arg("param_set"), py::arg("obs_op"),
             py::arg("bcs"), py::arg("nu"),
             py::arg("settings") = SolverSettings{},
             py::arg("u_init") = Vec3{1,0,0},
             py::arg("p_init") = 0.0,
             py::arg("k_init") = 1e-4,
             py::arg("omega_init") = 1.0)
        .def("evaluate", &ForwardModel::evaluate)
        .def("penalized_log_likelihood", &ForwardModel::penalizedLogLikelihood)
        .def("param_set", &ForwardModel::paramSet, py::return_value_policy::reference)
        .def("has_last_fields", &ForwardModel::hasLastFields)
        .def("last_fields", &extractFields);

    // TangentGradientResult — output of the RUNG-1 semi-analytic gradient eta_jacobian_tangent
    // (full ∂η/∂θ via the matrix-free tangent solve + per-coefficient Krylov diagnostics).
    py::class_<TangentGradientResult>(m, "TangentGradientResult")
        .def_readonly("d_obs_d_theta",    &TangentGradientResult::dObsDTheta)   // nObs × 11
        .def_readonly("log_lik_gradient", &TangentGradientResult::logLikGradient) // 11 (warm-FD)
        .def_readonly("krylov_iters",     &TangentGradientResult::krylovIters)
        .def_readonly("krylov_rel_res",   &TangentGradientResult::krylovRelRes)
        .def_readonly("krylov_converged", &TangentGradientResult::krylovConverged)
        .def_readonly("n_residual_evals", &TangentGradientResult::nResidualEvals)
        .def_readonly("n_colors",         &TangentGradientResult::nColors);

    // ParameterSensitivity — ADJOINT GROUNDWORK (∂R/∂θ + ∂g/∂θ at a fixed converged
    // state) plus the RUNG-1 semi-analytic true-model gradient eta_jacobian_tangent.  θ is
    // the FULL 11-vector (InferenceParameters indexing); the Python FD machinery validates
    // d_residual_d_theta against a fixed-state FD of residual(θ) and d_obs_d_theta against a
    // fixed-state FD of observe(θ); eta_jacobian_tangent is validated against full-FD ∂η/∂θ.
    py::class_<ParameterSensitivity>(m, "ParameterSensitivity")
        // keep_alive<1,2>: ParameterSensitivity stores `const Mesh&`, so the Python Mesh
        // (arg 2) must outlive the instance (arg 1).  Without this, `ParameterSensitivity(
        // *build_case())` would free the mesh temporary right after construction and the
        // stored reference would dangle (see the pybind11-lifetime note in auto-memory).
        .def(py::init<const Mesh&, const ObservationOperator&,
                      const FlowBoundaryConditions&, double, const SolverSettings&,
                      const Vec3&, double, double, double>(),
             py::arg("mesh"), py::arg("obs_op"), py::arg("bcs"), py::arg("nu"),
             py::arg("settings") = SolverSettings{},
             py::arg("u_init") = Vec3{1, 0, 0},
             py::arg("p_init") = 0.0, py::arg("k_init") = 1e-4,
             py::arg("omega_init") = 1.0,
             py::keep_alive<1, 2>())
        .def("solve_state", &ParameterSensitivity::solveState, py::arg("theta"))
        .def("has_state", &ParameterSensitivity::hasState)
        .def("n_state", &ParameterSensitivity::nState)
        .def("n_cells", &ParameterSensitivity::nCells)
        .def("n_obs", &ParameterSensitivity::nObs)
        .def("residual", &ParameterSensitivity::residual, py::arg("theta"))
        .def("d_residual_d_theta", &ParameterSensitivity::dResidualDTheta, py::arg("theta"))
        .def("observe", &ParameterSensitivity::observe, py::arg("theta"))
        .def("log_lik", &ParameterSensitivity::logLik, py::arg("theta"))
        .def("d_obs_d_theta", &ParameterSensitivity::dObsDTheta, py::arg("theta"))
        // RUNG 1 — semi-analytic true-model ∂η/∂θ (matrix-free frozen-pressure tangent;
        // NON-HELD).  Requires solve_state(theta) first; pass the SAME theta here.
        .def("eta_jacobian_tangent", &ParameterSensitivity::etaJacobianTangent,
             py::arg("theta"), py::arg("krylov_tol") = 1e-8,
             py::arg("max_iter") = 3000, py::arg("fd_step") = 1e-6)
        // RUNG 1 (PRESSURE-COUPLED) — full semi-analytic ∂η/∂θ (block-preconditioned FGMRES on the
        // 5-block saddle; robustly converged, matches full FD to ~1-2%).  Slower than warm-FD (the
        // production default) but FD-noise-free.  Requires solve_state(theta) first.  See
        // DECISION_RECORD §4b.
        .def("eta_jacobian_tangent_coupled", &ParameterSensitivity::etaJacobianTangentCoupled,
             py::arg("theta"), py::arg("krylov_tol") = 1e-7,
             py::arg("max_iter") = 2000, py::arg("fd_step") = 1e-6)
        // RUNG 1 (WARM-FD) — robust full ∂η/∂θ by warm-started central FD (matches cold full
        // FD; the recommended default true-model gradient).  Requires solve_state(theta) first.
        .def("eta_jacobian_warm_fd", &ParameterSensitivity::etaJacobianWarmFD,
             py::arg("theta"), py::arg("h_rel") = 5e-4, py::arg("h_floor") = 1e-7,
             py::arg("warm_max_iter") = 0, py::arg("warm_tol") = 0.0);

    // ---- Compressible bindings (PHASE 1) ---------------------------------
    //
    // Mirrors the incompressible API: same EvaluationStatus / EvaluationResult
    // (already bound above), same SSTCoefficients / InferenceParameterSet, but
    // with a temperature/density-aware solver and EOS.

    py::class_<IdealGasEOS>(m, "IdealGasEOS")
        .def(py::init<>())
        .def_readwrite("gamma",   &IdealGasEOS::gamma)
        .def_readwrite("R",       &IdealGasEOS::R)
        .def_readwrite("Pr",      &IdealGasEOS::Pr)
        .def_readwrite("Pr_T",    &IdealGasEOS::Pr_T)
        .def_readwrite("mu_ref",  &IdealGasEOS::mu_ref)
        .def_readwrite("T_ref",   &IdealGasEOS::T_ref)
        .def_readwrite("S_suth",  &IdealGasEOS::S_suth)
        .def("Cp",           &IdealGasEOS::Cp)
        .def("Cv",           &IdealGasEOS::Cv)
        .def("viscosity",    &IdealGasEOS::viscosity, py::arg("T"))
        .def("conductivity", &IdealGasEOS::conductivity, py::arg("mu"), py::arg("muT"))
        .def("density",      &IdealGasEOS::density,     py::arg("p"), py::arg("T"))
        .def("temperature",  &IdealGasEOS::temperature, py::arg("p"), py::arg("rho"))
        .def("pressure",     &IdealGasEOS::pressure,    py::arg("rho"), py::arg("T"))
        .def("sound_speed",  &IdealGasEOS::soundSpeed,  py::arg("T"))
        .def("mach_number",  &IdealGasEOS::machNumber,  py::arg("Umag"), py::arg("T"))
        .def("total_enthalpy", &IdealGasEOS::totalEnthalpy,
             py::arg("T"), py::arg("Umag2"));

    py::class_<CompressibleBoundaryConditions>(m, "CompressibleBoundaryConditions")
        .def(py::init<>())
        .def_static("channel_defaults",
             &CompressibleBoundaryConditions::channelDefaults,
             py::arg("mesh"), py::arg("Uin"), py::arg("T_in"), py::arg("p_out"),
             py::arg("kIn"), py::arg("omIn"));

    py::class_<CompressibleForwardModel>(m, "CompressibleForwardModel")
        .def(py::init<const Mesh&, const InferenceParameterSet&,
                       const ObservationOperator&,
                       const CompressibleBoundaryConditions&,
                       const IdealGasEOS&,
                       const SolverSettings&,
                       const Vec3&,
                       double, double, double, double>(),
             py::arg("mesh"), py::arg("param_set"), py::arg("obs_op"),
             py::arg("bcs"), py::arg("eos"),
             py::arg("settings")    = SolverSettings{},
             py::arg("u_init")      = Vec3{1.0, 0.0, 0.0},
             py::arg("p_init")      = 101325.0,
             py::arg("T_init")      = 300.0,
             py::arg("k_init")      = 1e-4,
             py::arg("omega_init")  = 100.0)
        .def("evaluate", &CompressibleForwardModel::evaluate)
        .def("penalized_log_likelihood",
             &CompressibleForwardModel::penalizedLogLikelihood)
        .def("param_set", &CompressibleForwardModel::paramSet,
             py::return_value_policy::reference)
        .def("has_last_fields", &CompressibleForwardModel::hasLastFields)
        .def("last_fields", &extractCompressibleFields);
}
