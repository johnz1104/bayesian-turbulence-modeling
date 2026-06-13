#pragma once
#include "BoundaryCondition.hpp"
#include "CompressibleFlowFields.hpp"
#include "IdealGasEOS.hpp"
#include <cmath>
#include <algorithm>

// Temperature boundary condition types.
enum class TBCType { Dirichlet, Neumann, AdiabaticWall };

struct PatchTBC {
    TBCType     type  = TBCType::Neumann;
    double      value = 300.0;  // fixed temperature [K] for Dirichlet
    std::string patchName;
};

// Full compressible BCs: velocity, pressure, temperature, k, omega.
struct CompressibleBoundaryConditions {
    std::vector<PatchBC>  velocityBC;
    std::vector<PatchBC>  pressureBC;
    std::vector<PatchTBC> temperatureBC;
    std::vector<PatchBC>  kBC;
    std::vector<PatchBC>  omegaBC;

    // Channel/duct flow with subsonic inlet.
    // Inlet: fixed U, T_in, p_outlet=0 (gauge), k, omega.
    static CompressibleBoundaryConditions channelDefaults(
        const Mesh& mesh, double Uin, double T_in, double p_out,
        double kIn, double omIn)
    {
        CompressibleBoundaryConditions bc;
        int np = mesh.nPatches();
        bc.velocityBC.resize(np);
        bc.pressureBC.resize(np);
        bc.temperatureBC.resize(np);
        bc.kBC.resize(np);
        bc.omegaBC.resize(np);

        for (int p = 0; p < np; ++p) {
            const Patch& pat = mesh.patch(p);
            bc.velocityBC[p].patchName    = pat.name;
            bc.pressureBC[p].patchName    = pat.name;
            bc.temperatureBC[p].patchName = pat.name;
            bc.kBC[p].patchName           = pat.name;
            bc.omegaBC[p].patchName       = pat.name;

            if (pat.type == "wall") {
                bc.velocityBC[p].type    = BCType::WallNoSlip;
                bc.pressureBC[p].type    = BCType::Neumann;
                bc.temperatureBC[p].type = TBCType::AdiabaticWall;
                bc.kBC[p].type           = BCType::WallKOmega;
                bc.omegaBC[p].type       = BCType::WallKOmega;
            } else if (pat.type == "inlet") {
                bc.velocityBC[p]    = {BCType::InletVelocity, 0.0, Vec3(Uin,0,0), pat.name};
                bc.pressureBC[p]    = {BCType::Neumann, 0.0, {}, pat.name};
                bc.temperatureBC[p] = {TBCType::Dirichlet, T_in, pat.name};
                bc.kBC[p]           = {BCType::Dirichlet, kIn, {}, pat.name};
                bc.omegaBC[p]       = {BCType::Dirichlet, omIn, {}, pat.name};
            } else if (pat.type == "outlet") {
                bc.velocityBC[p]    = {BCType::Neumann, 0.0, {}, pat.name};
                bc.pressureBC[p]    = {BCType::OutletPressure, p_out, {}, pat.name};
                bc.temperatureBC[p] = {TBCType::Neumann, 0.0, pat.name};
                bc.kBC[p]           = {BCType::Neumann, 0.0, {}, pat.name};
                bc.omegaBC[p]       = {BCType::Neumann, 0.0, {}, pat.name};
            } else {
                // symmetry / cyclic: zero-gradient for all
                bc.temperatureBC[p].type = TBCType::Neumann;
            }
        }
        return bc;
    }
};

// Apply temperature BCs to boundary faces.
inline void applyTemperatureBC(ScalarField& T,
                                const Mesh& mesh,
                                const CompressibleBoundaryConditions& bcs)
{
    for (int p = 0; p < mesh.nPatches(); ++p) {
        const Patch& pat     = mesh.patch(p);
        const PatchTBC& bc   = bcs.temperatureBC[p];
        for (FaceID fi : pat.faces) {
            switch (bc.type) {
                case TBCType::Dirichlet:
                    T.bface(fi) = bc.value; break;
                case TBCType::AdiabaticWall:
                case TBCType::Neumann:
                default:
                    T.bface(fi) = T[mesh.face(fi).owner]; break;
            }
        }
    }
}

// Apply all compressible BCs (velocity, pressure, T, k, omega).
inline void applyAllCompressibleBCs(
    VectorField& U, ScalarField& p, ScalarField& T,
    ScalarField& k, ScalarField& omega,
    const Mesh& mesh, const CompressibleBoundaryConditions& bcs, double nu)
{
    // Velocity
    for (int p_ = 0; p_ < mesh.nPatches(); ++p_) {
        const Patch& pat = mesh.patch(p_);
        const PatchBC& bc = bcs.velocityBC[p_];
        for (FaceID fi : pat.faces) {
            switch (bc.type) {
                case BCType::WallNoSlip:    U.bface(fi) = Vec3(0,0,0); break;
                case BCType::InletVelocity:
                case BCType::Dirichlet:     U.bface(fi) = bc.vecValue; break;
                default:                   U.bface(fi) = U[mesh.face(fi).owner]; break;
            }
        }
    }
    // Pressure
    for (int p_ = 0; p_ < mesh.nPatches(); ++p_) {
        const Patch& pat = mesh.patch(p_);
        const PatchBC& bc = bcs.pressureBC[p_];
        for (FaceID fi : pat.faces) {
            switch (bc.type) {
                case BCType::OutletPressure:
                case BCType::Dirichlet:  p.bface(fi) = bc.value; break;
                default:                 p.bface(fi) = p[mesh.face(fi).owner]; break;
            }
        }
    }
    applyTemperatureBC(T, mesh, bcs);
    applyKBC(k, mesh, {bcs.velocityBC, bcs.pressureBC, bcs.kBC, bcs.omegaBC});
    applyOmegaBC(omega, mesh, {bcs.velocityBC, bcs.pressureBC, bcs.kBC, bcs.omegaBC}, nu);
}
