#include "Mesh.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <iostream>
#include <filesystem>
#include <queue>
#include <numeric>
#include <cstring>

// loadFromFile supported mesh formats: 
//      - .foam / directory path    => OpenFoam polyMesh 
//      - .msh                      => custom bindary format
// For OpenFOAM, pass case directory or the path to the the polyMesh directory

Mesh Mesh::loadFromFile(const std::string& path) {
    namespace fs = std::filesystem;

    // Directory-based dispatch 
    // checks if path is a directory (treat as OpenFOAM case) or a direct polyMesh directory
    if (fs::is_directory(path)) {
        fs::path p(path);
        if (fs::exists(p / "points") && fs::exists(p / "faces")) {
            return loadOpenFOAM(path);
        }
        fs::path polyMesh = p / "constant" / "polyMesh";
        if (fs::exists(polyMesh / "points")) {
            return loadOpenFOAM(polyMesh.string());
        }
        throw std::runtime_error("Cannot find polyMesh in: " + path);
    }

    // File-based dispatch
    auto ext = fs::path(path).extension().string();
    if (ext == ".foam" || ext == "") {
        return loadOpenFOAM(path);
    }
    if (ext == ".msh") {
        return loadBinary(path);
    }
    throw std::runtime_error("Unknown mesh format: " + ext);
}

// OpenFOAM polyMesh reader (OpenFOAM mesh usually defined by 5 files)
// points   -- node coordinates
// faces    -- face node connectives
// owner    -- owner cell for each face
// neighbor -- neighbor cell for internal faces
// boundary -- patch definitions

// Parsing Helper: skips OpenfOAM file header until we find the start of data
static void skipFoamHeader(std::istream& is){
    std::string line;
    while (std::getline(is, line)) {
        auto pos = line.find("//");
        if (pos != std::string::npos) line.erase(pos);
        line.erase(0, line.find_first_not_of(" \t\r"));
        if (line == "(") return;
    }
    throw std::runtime_error("Could not find data block");
}

// Parsing helper: reads the element count from an OpenFOAM data file
static int readFoamCount(std::istream& is){
    std::string line;
    while (std::getline(is, line)) {
        auto pos = line.find("//");
        if (pos != std::string::npos) line.erase(pos);
        line.erase(0, line.find_first_not_of(" \t\r"));
        line.erase(line.find_last_not_of(" \t\r") + 1);
        if (line.empty() || line[0] == '/' || line[0] == 'F' ||
            line[0] == 'O' || line[0] == 'o' || line[0] == 'v' ||
            line[0] == 'C' || line[0] == 'a' || line[0] == '{' ||
            line[0] == '}')
            continue;
        try { return std::stoi(line); } catch (...) { continue; }
    }
    throw std::runtime_error("Could not read count");
}

Mesh Mesh::loadOpenFOAM(const std::string& polyMeshDir){
    namespace fs = std::filesystem;
    Mesh m;

    // extracts OpenFOAM mesh point data (defines geometric node coordinates)
    {
        std::ifstream ifs(fs::path(polyMeshDir) / "points");
        if (!ifs) throw std::runtime_error("Cannot open points file");
        int nPts = readFoamCount(ifs);
        skipFoamHeader(ifs);
        m.nodes_.resize(nPts);
        for (int i = 0; i < nPts; ++i) {
            char c;
            ifs >> c; // '('
            ifs >> m.nodes_[i].x >> m.nodes_[i].y >> m.nodes_[i].z;
            ifs >> c; // ')'
        }
    } 
    
    // extracts OpenFOAM mesh face data (defines face-node connectivity topology)
    std::vector<std::vector<int>> faceNodes;
    {
        std::ifstream ifs(fs::path(polyMeshDir) / "faces");
        if (!ifs) throw std::runtime_error("Cannot open faces file");
        int nF = readFoamCount(ifs);
        skipFoamHeader(ifs);
        faceNodes.resize(nF);
        for (int i = 0; i < nF; ++i) {
            int nv;
            ifs >> nv;
            char c;
            ifs >> c; // '('
            faceNodes[i].resize(nv);
            for (int j = 0; j < nv; ++j) ifs >> faceNodes[i][j];
            ifs >> c; // ')'
        }
        m.faces_.resize(nF);
    }

    // extracts OpenFOAM mesh owner data (defines face-cell connectivitytopology)
    {
        std::ifstream ifs(fs::path(polyMeshDir) / "owner");
        if (!ifs) throw std::runtime_error("Cannot open owner file in " + polyMeshDir);
        int nOwn = readFoamCount(ifs);
        skipFoamHeader(ifs);
        for (int i = 0; i < nOwn; ++i) {
            int o;
            ifs >> o;
            m.faces_[i].owner = o;
        }
    }

    // extracts OpenFOAM mesh neighbor data (defines adjacent cell assignments internal faces)
    {
        fs::path nbrPath = fs::path(polyMeshDir) / "neighbour";
        std::ifstream ifs(nbrPath);
        if (!ifs) throw std::runtime_error("Cannot open neighbour file in " + polyMeshDir);
        int nNbr = readFoamCount(ifs);
        m.nInternal_ = nNbr;
        skipFoamHeader(ifs);
        for (int i = 0; i < nNbr; ++i) {
            int n;
            ifs >> n;
            m.faces_[i].neighbor = n;
        }
    }

    // determine total cell count from face owner and neighbor indices
    int nCells = 0;
    for (auto& f : m.faces_) {
        nCells = std::max(nCells, f.owner + 1);
        if (f.neighbor >= 0) nCells = std::max(nCells, f.neighbor + 1);
    }
    m.cells_.resize(nCells);

    // build cell-face adjacency (assigns each face to its owner and neighbor cells)
    for (int fi = 0; fi < static_cast<int>(m.faces_.size()); ++fi) {
        m.cells_[m.faces_[fi].owner].faces.push_back(fi);
        if (m.faces_[fi].neighbor >= 0)
            m.cells_[m.faces_[fi].neighbor].faces.push_back(fi);
    }

    // reads OpenFOAM boundary patch definitions and construct patch objects 
    // assigns boundary faces to patches and label faces with their patch ID
    {
        std::ifstream ifs(fs::path(polyMeshDir) / "boundary");
        if (ifs) {
            int nPatches = readFoamCount(ifs);
            skipFoamHeader(ifs);
            m.patches_.resize(nPatches);
            for (int p = 0; p < nPatches; ++p) {
                std::string line;
                // read patch name
                while (std::getline(ifs, line)) {
                    auto pos = line.find("//");
                    if (pos != std::string::npos) line.erase(pos);
                    line.erase(0, line.find_first_not_of(" \t\r"));
                    line.erase(line.find_last_not_of(" \t\r") + 1);
                    if (!line.empty() && line != "{" && line != "}") {
                        m.patches_[p].name = line;
                        break;
                    }
                }
                // parse patch body { type ...; nFaces ...; startFace ...; }
                int pnFaces = 0, startFace = 0;
                while (std::getline(ifs, line)) {
                    auto pos = line.find("//");
                    if (pos != std::string::npos) line.erase(pos);
                    line.erase(0, line.find_first_not_of(" \t\r"));
                    if (line.find('}') != std::string::npos) break;
                    std::istringstream ss(line);
                    std::string key;
                    ss >> key;
                    if (key == "type") {
                        std::string val;
                        ss >> val;
                        if (!val.empty() && val.back() == ';') val.pop_back();
                        m.patches_[p].type = val;
                    } else if (key == "nFaces") {
                        ss >> pnFaces;
                    } else if (key == "startFace") {
                        ss >> startFace;
                    }
                }
                m.patches_[p].faces.resize(pnFaces);
                for (int i = 0; i < pnFaces; ++i) {
                    int fIdx = startFace + i;
                    m.patches_[p].faces[i] = fIdx;
                    m.faces_[fIdx].patchID = p;
                }
            }
        }
    }
    // Mesh structure now: (-> means maps to)
    // Topology: face -> owner, face-> neighbor, cell -> faces
    // Boundary structure: patch -> faces, face -> patche, patch -> type


    // build geometry from topology
    // computes face centroid, area, and normal vector
    for (int fi = 0; fi < static_cast<int>(m.faces_.size()); ++fi) {
        const auto& fn = faceNodes[fi];
        int nv = static_cast<int>(fn.size());

        Vec3 ctr{};                                 // compute face centroid
        for (int j = 0; j < nv; ++j) {
            ctr = ctr + m.nodes_[fn[j]];
        }
        ctr = ctr / static_cast<double>(nv);
        m.faces_[fi].center = ctr;

        Vec3 areaNormal{};                          // compute face area normal vector
        for (int j = 0; j < nv; ++j) {
            const Vec3& a = m.nodes_[fn[j]];
            const Vec3& b = m.nodes_[fn[(j + 1) % nv]];
            areaNormal = areaNormal + (a - ctr).cross(b - ctr) * 0.5;
        }
        m.faces_[fi].area   = areaNormal.norm();    // face area (scalar)    
        m.faces_[fi].normal = areaNormal.unit();    // face unit normal
    }

    // build owner/neighbor connectivity arrays
    m.ownerList_.resize(m.nInternal_);              // m.nInternal_ = number of internal faces.
    m.neighborList_.resize(m.nInternal_);
    for (int fi = 0; fi < m.nInternal_; ++fi) {
        m.ownerList_[fi]    = m.faces_[fi].owner;
        m.neighborList_[fi] = m.faces_[fi].neighbor;
    }

    // compute cell centers, volumes, interpolation weights, delta vectors
    m.computeCellCentersAndVolumes();
    m.computeFaceGeometry();
    m.ComputeInterpolationWeights();

    // prints summary
    std::cout << "OpenFOAM mesh loaded: " << m.nCells() << " cells, "
              << m.nFaces() << " faces, " << m.nInternal_ << " internal faces, "
              << m.nPatches() << " patches\n";
    return m;
}

// Geometry Computations

void Mesh::computeCellCentersAndVolumes(){
    // determines cell centers by averaging the centers of all cell faces
    for (int ci = 0; ci < nCells(); ++ci) {
        Vec3 ctr{};
        for (FaceID fi : cells_[ci].faces) {
            ctr = ctr + faces_[fi].center;
        }
        cells_[ci].center = ctr / static_cast<double>(cells_[ci].faces.size());
    }

    // using divergence theorem to compute polyhedral cell volumes 
    // by summing signed contributions of each face’s area vector 
    // dotted with the vector from cell center to face center.
    // V = sum_f (Sf . (cf - cC))
    // area vector dotted with vector from cell center to face center
    // summing over faces gives volume 
    for (int ci = 0; ci < nCells(); ++ci) {
        double vol = 0.0;
        for (FaceID fi : cells_[ci].faces) {
            const Face& f = faces_[fi];
            Vec3 Sf = f.normal * f.area;
            double sign = (f.owner == ci) ? 1.0 : -1.0;
            vol += sign * Sf.dot(f.center - cells_[ci].center);
        }
        cells_[ci].volume = std::abs(vol) / 3.0;
        if (cells_[ci].volume < 1e-30) {
            cells_[ci].volume = 1e-20; // fallback for degenerate or nearly flat cells
        }
    }
}

// computes face-cell distance vectors and magnitudes
void Mesh::computeFaceGeometry() {
    // delta = distance between owner and neighbor centers
    // d     = vector from owner to neighbor centers
    for (int fi = 0; fi < nFaces(); ++fi) {
        Face& f = faces_[fi];
        if (!f.isBoundary()) {
            f.d     = cells_[f.neighbor].center - cells_[f.owner].center;
            f.delta = f.d.norm();
        } else {
            f.d     = f.center - cells_[f.owner].center;
            f.delta = f.d.norm();
        }
        if (f.delta < 1e-30) f.delta = 1e-20;
    }
}

// computes linear interpolation weights for internal faces from cell-face center distances
void Mesh::ComputeInterpolationWeights() {
    // weight = |fC - N| / (|fC - P| + |fC - N|)
    // phi_f = w * phi_P + (1-w) * phi_N
    for (int fi = 0; fi < nInternal_; ++fi) {
        Face& f = faces_[fi];
        double dP = (f.center - cells_[f.owner].center).norm();
        double dN = (f.center - cells_[f.neighbor].center).norm();
        double sum = dP + dN;
        f.weight = (sum > 1e-30) ? dN / sum : 0.5;
    }
}

// converts topological mesh into a discretization-ready computational grid
void Mesh::computeGeometry() {
    computeCellCentersAndVolumes();
    computeFaceGeometry();
    ComputeInterpolationWeights();

    ownerList_.resize(nInternal_);
    neighborList_.resize(nInternal_);
    for (int fi = 0; fi < nInternal_; ++fi) {
        ownerList_[fi]    = faces_[fi].owner;
        neighborList_[fi] = faces_[fi].neighbor;
    }
}

// computeWallDistance — brute-force from wall patches
void Mesh::computeWallDistance(){
    wallDist_.assign(nCells(), std::numeric_limits<double>::max());
    
    // collects wall face centers
    std::vector<Vec3> wallPts;
    for (const auto& p : patches_) {
        if (p.type == "wall") {
            for (FaceID fi : p.faces) {
                wallPts.push_back(faces_[fi].center);
            }
        }
    }
    // handles no wall case
    if (wallPts.empty()) {
        std::fill(wallDist_.begin(), wallDist_.end(), 1e10);
        return;
    }
    // brute force distance search
    // computes minimum Euclidean distance from each cell center to any wall face center
    for (int ci = 0; ci < nCells(); ++ci) {
        double minD = std::numeric_limits<double>::max();
        const Vec3& cc = cells_[ci].center;
        for (const Vec3& wp : wallPts) {
            double d = (cc - wp).norm();
            if (d < minD) minD = d;
        }
        wallDist_[ci] = minD;
    }
}

// lookup mechanism to retrieve a boundary patch index by its name
PatchID Mesh::patchByName(const std::string& name) const {
    for (int i = 0; i < static_cast<int>(patches_.size()); ++i) {
        if (patches_[i].name == name) return i;
    }
    throw std::runtime_error("Patch not found: " + name);
}

// retype a boundary patch; the caller re-runs computeWallDistance so a patch
// retyped away from "wall" stops contributing wall distances (SST blending)
void Mesh::setPatchType(const std::string& name, const std::string& type) {
    patches_[patchByName(name)].type = type;
}

// Streamwise-periodic curved-bottom channel (see the header note). All cell and
// face geometry is computed HERE, exactly, from the quad corners: the generic
// helpers cannot be used because (a) the wrap face's geometric center sits at
// x = Lx while its neighbor cell sits at x ~ 0, so face-center-averaged cell
// centers and center-to-center distances would span the whole domain, and (b)
// the wrap face's d/delta/weight must use the PERIODIC IMAGE of the neighbor.
Mesh Mesh::makeCurvedChannelPeriodic2D(const std::vector<double>& xNodes,
                                       const std::vector<double>& yBottom,
                                       double yTop, int ny,
                                       double Re, double yPlusTarget) {
    const int nx = static_cast<int>(xNodes.size()) - 1;
    if (nx < 3 || static_cast<int>(yBottom.size()) != nx + 1)
        throw std::runtime_error("makeCurvedChannelPeriodic2D: need nx+1 x nodes "
                                 "and matching yBottom samples");
    if (std::abs(yBottom.front() - yBottom.back()) > 1e-12)
        throw std::runtime_error("makeCurvedChannelPeriodic2D: yBottom must be "
                                 "periodic (first == last sample)");
    const double Lx = xNodes.back() - xNodes.front();
    const double dz = 1.0;

    // wall-normal distribution eta_j in [0,1], symmetric tanh clustering toward
    // both walls; the stretch is solved on the MEAN channel height so every
    // column shares the same eta (smooth terrain-following grid lines)
    double meanH = 0.0;
    for (int i = 0; i <= nx; ++i) meanH += (yTop - yBottom[i]);
    meanH /= (nx + 1);
    double stretch = 2.0;
    if (Re > 0) {
        double Cf    = 0.058 * std::pow(Re, -0.2);
        double uTau  = std::sqrt(Cf / 2.0);
        double nu    = meanH / Re;
        double y1t   = yPlusTarget * nu / uTau;
        auto fc = [&](double s) {
            return 0.5 * meanH * (1.0 + std::tanh(s * (2.0 / ny - 1.0))
                                        / std::tanh(s));
        };
        if (y1t < meanH / ny) {
            double sLo = 0.1, sHi = 20.0;
            if (y1t < fc(sHi)) {
                stretch = sHi;
            } else {
                for (int it = 0; it < 100; ++it) {
                    double sm = 0.5 * (sLo + sHi);
                    if (fc(sm) > y1t) sLo = sm; else sHi = sm;
                    if (sHi - sLo < 1e-10) break;
                }
                stretch = 0.5 * (sLo + sHi);
            }
        }
    }
    std::vector<double> eta(ny + 1);
    for (int j = 0; j <= ny; ++j) {
        double e = static_cast<double>(j) / ny;
        eta[j] = 0.5 * (1.0 + std::tanh(stretch * (2.0 * e - 1.0))
                              / std::tanh(stretch));
    }

    // node coordinates (front plane; back plane duplicated at z = dz)
    auto ynode = [&](int i, int j) {
        return yBottom[i] + (yTop - yBottom[i]) * eta[j];
    };
    Mesh m;
    const int ptsPerPlane = (nx + 1) * (ny + 1);
    m.nodes_.resize(2 * ptsPerPlane);
    auto nid = [&](int i, int j, int k) {
        return k * ptsPerPlane + j * (nx + 1) + i;
    };
    for (int j = 0; j <= ny; ++j)
        for (int i = 0; i <= nx; ++i) {
            m.nodes_[nid(i, j, 0)] = Vec3(xNodes[i], ynode(i, j), 0.0);
            m.nodes_[nid(i, j, 1)] = Vec3(xNodes[i], ynode(i, j), dz);
        }

    // cells: exact quad centroid and area (shoelace), volume = area * dz
    const int nC = nx * ny;
    m.cells_.resize(nC);
    auto cid = [&](int i, int j) { return j * nx + i; };
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            const double px[4] = {xNodes[i], xNodes[i + 1], xNodes[i + 1], xNodes[i]};
            const double py[4] = {ynode(i, j), ynode(i + 1, j),
                                  ynode(i + 1, j + 1), ynode(i, j + 1)};
            double A2 = 0.0, cxA = 0.0, cyA = 0.0;
            for (int kv = 0; kv < 4; ++kv) {
                int kn = (kv + 1) % 4;
                double cross = px[kv] * py[kn] - px[kn] * py[kv];
                A2  += cross;
                cxA += (px[kv] + px[kn]) * cross;
                cyA += (py[kv] + py[kn]) * cross;
            }
            double A = 0.5 * A2;                 // positive for CCW corners
            Cell& c = m.cells_[cid(i, j)];
            c.center = Vec3(cxA / (6.0 * A), cyA / (6.0 * A), 0.5 * dz);
            c.volume = std::abs(A) * dz;
        }

    // face counts: vertical internal (nx-1 columns + nx wrap? no: nx-1 interior
    // columns plus ONE wrap column) + horizontal internal + boundaries
    const int nVint  = (nx - 1) * ny;    // between columns i and i+1
    const int nWrap  = ny;               // between column nx-1 and column 0
    const int nHint  = nx * (ny - 1);    // between rows j and j+1
    m.nInternal_ = nVint + nWrap + nHint;
    const int nBnd = 2 * nx;             // bottom + top
    m.faces_.resize(m.nInternal_ + nBnd);

    int fi = 0;
    auto edgeFace = [&](Face& f, double x0, double y0, double x1, double y1,
                        bool normalPlusX) {
        // 2D edge (x0,y0)-(x1,y1) extruded by dz; unit normal in-plane
        double ex = x1 - x0, ey = y1 - y0;
        double len = std::sqrt(ex * ex + ey * ey);
        f.center = Vec3(0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * dz);
        f.area   = len * dz;
        // the two in-plane normals are (ey,-ex)/len and (-ey,ex)/len
        if (normalPlusX) f.normal = Vec3(ey / len, -ex / len, 0.0);
        else             f.normal = Vec3(-ey / len, ex / len, 0.0);
    };
    auto setDW = [&](Face& f, const Vec3& cO, const Vec3& cN) {
        f.d      = cN - cO;
        f.delta  = std::max(f.d.norm(), 1e-20);
        double dP = (f.center - cO).norm();
        double dN = (f.center - cN).norm();
        double sum = dP + dN;
        f.weight = (sum > 1e-30) ? dN / sum : 0.5;
    };

    // internal vertical faces (columns are straight: edge along +y at x_{i+1},
    // normal exactly +x, orthogonal to the column direction)
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx - 1; ++i) {
            Face& f = m.faces_[fi];
            f.owner = cid(i, j); f.neighbor = cid(i + 1, j);
            edgeFace(f, xNodes[i + 1], ynode(i + 1, j),
                        xNodes[i + 1], ynode(i + 1, j + 1), true);
            setDW(f, m.cells_[f.owner].center, m.cells_[f.neighbor].center);
            m.cells_[f.owner].faces.push_back(fi);
            m.cells_[f.neighbor].faces.push_back(fi); ++fi;
        }
    // wrap faces: owner = last column, neighbor = first column; the geometric
    // face sits at x = Lx and every distance uses the neighbor's PERIODIC IMAGE
    for (int j = 0; j < ny; ++j) {
        Face& f = m.faces_[fi];
        f.owner = cid(nx - 1, j); f.neighbor = cid(0, j);
        edgeFace(f, xNodes[nx], ynode(nx, j),
                    xNodes[nx], ynode(nx, j + 1), true);
        Vec3 cNimage = m.cells_[f.neighbor].center + Vec3(Lx, 0.0, 0.0);
        setDW(f, m.cells_[f.owner].center, cNimage);
        m.cells_[f.owner].faces.push_back(fi);
        m.cells_[f.neighbor].faces.push_back(fi); ++fi;
    }
    // internal horizontal faces (terrain-following: tilted near the slope)
    for (int j = 0; j < ny - 1; ++j)
        for (int i = 0; i < nx; ++i) {
            Face& f = m.faces_[fi];
            f.owner = cid(i, j); f.neighbor = cid(i, j + 1);
            edgeFace(f, xNodes[i], ynode(i, j + 1),
                        xNodes[i + 1], ynode(i + 1, j + 1), false);
            setDW(f, m.cells_[f.owner].center, m.cells_[f.neighbor].center);
            m.cells_[f.owner].faces.push_back(fi);
            m.cells_[f.neighbor].faces.push_back(fi); ++fi;
        }

    // bottom wall (curved), outward normal points down-slope-outward
    {
        Patch p; p.name = "bottom_wall"; p.type = "wall";
        for (int i = 0; i < nx; ++i) {
            Face& f = m.faces_[fi];
            f.owner = cid(i, 0); f.neighbor = -1;
            f.patchID = static_cast<int>(m.patches_.size());
            edgeFace(f, xNodes[i], ynode(i, 0),
                        xNodes[i + 1], ynode(i + 1, 0), true);
            // normalPlusX=true gives (ey,-ex): ey ~ slope, -ex < 0 so it points
            // DOWN out of the domain as required for the bottom boundary
            f.d = f.center - m.cells_[f.owner].center;
            f.delta = std::max(f.d.norm(), 1e-20);
            m.cells_[f.owner].faces.push_back(fi);
            p.faces.push_back(fi); ++fi;
        }
        m.patches_.push_back(std::move(p));
    }
    // top wall (flat), outward normal +y
    {
        Patch p; p.name = "top_wall"; p.type = "wall";
        for (int i = 0; i < nx; ++i) {
            Face& f = m.faces_[fi];
            f.owner = cid(i, ny - 1); f.neighbor = -1;
            f.patchID = static_cast<int>(m.patches_.size());
            edgeFace(f, xNodes[i], ynode(i, ny),
                        xNodes[i + 1], ynode(i + 1, ny), false);
            f.d = f.center - m.cells_[f.owner].center;
            f.delta = std::max(f.d.norm(), 1e-20);
            m.cells_[f.owner].faces.push_back(fi);
            p.faces.push_back(fi); ++fi;
        }
        m.patches_.push_back(std::move(p));
    }

    // owner/neighbor lists for the linear solver
    m.ownerList_.resize(m.nInternal_);
    m.neighborList_.resize(m.nInternal_);
    for (int f = 0; f < m.nInternal_; ++f) {
        m.ownerList_[f]    = m.faces_[f].owner;
        m.neighborList_[f] = m.faces_[f].neighbor;
    }

    std::cout << "CurvedChannelPeriodic2D: " << nx << "x" << ny << " ("
              << nC << " cells, " << m.faces_.size() << " faces, "
              << nWrap << " wrap faces)\n";
    return m;
}

// save/load binary mesh code
// binary serialization layer
static const uint32_t MESH_MAGIC = 0x4D534831; // "MSH1"

void Mesh::saveBinary(const std::string& path) const {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) throw std::runtime_error("Cannot open for writing: " + path);

    auto writeInt = [&](int v)    { ofs.write(reinterpret_cast<const char*>(&v), sizeof(int)); };
    auto writeDbl = [&](double v) { ofs.write(reinterpret_cast<const char*>(&v), sizeof(double)); };
    auto writeVec = [&](const Vec3& v) { writeDbl(v.x); writeDbl(v.y); writeDbl(v.z); };
    auto writeStr = [&](const std::string& s) {
        int len = static_cast<int>(s.size());
        writeInt(len);
        ofs.write(s.data(), len);
    };

    uint32_t magic = MESH_MAGIC;
    ofs.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    writeInt(nCells());
    writeInt(nFaces());
    writeInt(nNodes());
    writeInt(nPatches());
    writeInt(nInternal_);

    for (const auto& n : nodes_) writeVec(n);

    for (const auto& f : faces_) {
        writeInt(f.owner); writeInt(f.neighbor); writeInt(f.patchID);
        writeVec(f.center); writeVec(f.normal);
        writeDbl(f.area); writeDbl(f.delta);
        writeVec(f.d); writeDbl(f.weight);
    }

    for (const auto& c : cells_) {
        writeVec(c.center); writeDbl(c.volume);
        int nf = static_cast<int>(c.faces.size());
        writeInt(nf);
        for (FaceID fid : c.faces) writeInt(fid);
    }

    for (const auto& p : patches_) {
        writeStr(p.name); writeStr(p.type);
        int nf = static_cast<int>(p.faces.size());
        writeInt(nf);
        for (FaceID fid : p.faces) writeInt(fid);
    }

    std::cout << "Mesh saved: " << path << "\n";
}

Mesh Mesh::loadBinary(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) throw std::runtime_error("Cannot open binary mesh: " + path);

    auto readInt = [&]() -> int    { int v;    ifs.read(reinterpret_cast<char*>(&v), sizeof(int));    return v; };
    auto readDbl = [&]() -> double { double v; ifs.read(reinterpret_cast<char*>(&v), sizeof(double)); return v; };
    auto readVec = [&]() -> Vec3   { return {readDbl(), readDbl(), readDbl()}; };
    auto readStr = [&]() -> std::string {
        int len = readInt();
        std::string s(len, '\0');
        ifs.read(s.data(), len);
        return s;
    };

    uint32_t magic;
    ifs.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != MESH_MAGIC)
        throw std::runtime_error("Invalid binary mesh magic number");

    Mesh m;
    int nc = readInt(), nf = readInt(), nn = readInt(), np = readInt();
    m.nInternal_ = readInt();

    m.nodes_.resize(nn);
    for (int i = 0; i < nn; ++i) m.nodes_[i] = readVec();

    m.faces_.resize(nf);
    for (int i = 0; i < nf; ++i) {
        Face& f = m.faces_[i];
        f.owner = readInt(); f.neighbor = readInt(); f.patchID = readInt();
        f.center = readVec(); f.normal = readVec();
        f.area = readDbl(); f.delta = readDbl();
        f.d = readVec(); f.weight = readDbl();
    }

    m.cells_.resize(nc);
    for (int i = 0; i < nc; ++i) {
        Cell& c = m.cells_[i];
        c.center = readVec(); c.volume = readDbl();
        int cfn = readInt();
        c.faces.resize(cfn);
        for (int j = 0; j < cfn; ++j) c.faces[j] = readInt();
    }

    m.patches_.resize(np);
    for (int i = 0; i < np; ++i) {
        m.patches_[i].name = readStr();
        m.patches_[i].type = readStr();
        int pfn = readInt();
        m.patches_[i].faces.resize(pfn);
        for (int j = 0; j < pfn; ++j) m.patches_[i].faces[j] = readInt();
    }

    m.ownerList_.resize(m.nInternal_);
    m.neighborList_.resize(m.nInternal_);
    for (int i = 0; i < m.nInternal_; ++i) {
        m.ownerList_[i]    = m.faces_[i].owner;
        m.neighborList_[i] = m.faces_[i].neighbor;
    }

    std::cout << "Binary mesh loaded: " << nc << " cells, " << nf << " faces\n";
    return m;
}


// add validation mesh
// patches: "inlet" (x=0), "outlet" (x=Lx), "top" (y=Ly, wall), "bottom" (y=0, wall)
// y-direction - tanh stretching toward both walls for BL resolution.
Mesh Mesh::makeChannel2D(int nx, int ny, double Lx, double Ly) {
    Mesh m;
    double dz = 1.0; // unit depth for per-unit-depth quantities

    // y-coordinates with tanh stretching toward both walls
    double stretch = 2.0;
    std::vector<double> yc(ny + 1);
    for (int j = 0; j <= ny; ++j) {
        double eta = static_cast<double>(j) / ny;
        yc[j] = 0.5 * Ly * (1.0 + std::tanh(stretch * (2.0 * eta - 1.0))
                                   / std::tanh(stretch));
    }
    // uniform x-coordinates
    std::vector<double> xc(nx + 1);
    for (int i = 0; i <= nx; ++i) {
        xc[i] = Lx * static_cast<double>(i) / nx;
    }
    // nodes: (nx+1)*(ny+1) * 2 planes (front z=0, back z=dz)
    int ptsPerPlane = (nx + 1) * (ny + 1);
    m.nodes_.resize(2 * ptsPerPlane);
    auto nid = [&](int i, int j, int k) { return k * ptsPerPlane + j * (nx + 1) + i; };
    for (int j = 0; j <= ny; ++j)
        for (int i = 0; i <= nx; ++i) {
            m.nodes_[nid(i, j, 0)] = Vec3(xc[i], yc[j], 0.0);
            m.nodes_[nid(i, j, 1)] = Vec3(xc[i], yc[j], dz);
        }

    // cells
    int nC = nx * ny;
    m.cells_.resize(nC);
    auto cid = [&](int i, int j) { return j * nx + i; };

    // faces
    int nInternalH = nx * (ny - 1);      // horizontal internal (between j, j+1)
    int nInternalV = (nx - 1) * ny;      // vertical internal   (between i, i+1)
    m.nInternal_ = nInternalH + nInternalV;
    int nBnd = 2 * nx + 2 * ny;          // bottom + top + inlet + outlet
    m.faces_.resize(m.nInternal_ + nBnd);

    int fi = 0;

    // internal horizontal faces
    for (int j = 0; j < ny - 1; ++j) {
        for (int i = 0; i < nx; ++i) {
            Face& f    = m.faces_[fi];
            f.owner    = cid(i, j);
            f.neighbor = cid(i, j + 1);
            double x0 = xc[i], x1 = xc[i + 1];
            f.center = Vec3(0.5 * (x0 + x1), yc[j + 1], 0.5 * dz);
            f.area   = (x1 - x0) * dz;
            f.normal = Vec3(0, 1, 0);
            m.cells_[f.owner].faces.push_back(fi);
            m.cells_[f.neighbor].faces.push_back(fi);
            ++fi;
        }
    }

    // internal vertical faces
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx - 1; ++i) {
            Face& f    = m.faces_[fi];
            f.owner    = cid(i, j);
            f.neighbor = cid(i + 1, j);
            double y0 = yc[j], y1 = yc[j + 1];
            f.center = Vec3(xc[i + 1], 0.5 * (y0 + y1), 0.5 * dz);
            f.area   = (y1 - y0) * dz;
            f.normal = Vec3(1, 0, 0);
            m.cells_[f.owner].faces.push_back(fi);
            m.cells_[f.neighbor].faces.push_back(fi);
            ++fi;
        }
    }

    // boundary faces
    // bottom wall (y = 0)
    Patch bottom; bottom.name = "bottom"; bottom.type = "wall";
    for (int i = 0; i < nx; ++i) {
        Face& f = m.faces_[fi];
        f.owner = cid(i, 0); f.neighbor = -1;
        f.patchID = static_cast<int>(m.patches_.size());
        f.center = Vec3(0.5 * (xc[i] + xc[i + 1]), 0.0, 0.5 * dz);
        f.area = (xc[i + 1] - xc[i]) * dz;
        f.normal = Vec3(0, -1, 0);
        m.cells_[f.owner].faces.push_back(fi);
        bottom.faces.push_back(fi); ++fi;
    }
    m.patches_.push_back(std::move(bottom));

    // top wall (y = Ly)
    Patch top; top.name = "top"; top.type = "wall";
    for (int i = 0; i < nx; ++i) {
        Face& f = m.faces_[fi];
        f.owner = cid(i, ny - 1); f.neighbor = -1;
        f.patchID = static_cast<int>(m.patches_.size());
        f.center = Vec3(0.5 * (xc[i] + xc[i + 1]), Ly, 0.5 * dz);
        f.area = (xc[i + 1] - xc[i]) * dz;
        f.normal = Vec3(0, 1, 0);
        m.cells_[f.owner].faces.push_back(fi);
        top.faces.push_back(fi); ++fi;
    }
    m.patches_.push_back(std::move(top));

    // inlet (x = 0)
    Patch inlet; inlet.name = "inlet"; inlet.type = "inlet";
    for (int j = 0; j < ny; ++j) {
        Face& f = m.faces_[fi];
        f.owner = cid(0, j); f.neighbor = -1;
        f.patchID = static_cast<int>(m.patches_.size());
        f.center = Vec3(0.0, 0.5 * (yc[j] + yc[j + 1]), 0.5 * dz);
        f.area = (yc[j + 1] - yc[j]) * dz;
        f.normal = Vec3(-1, 0, 0);
        m.cells_[f.owner].faces.push_back(fi);
        inlet.faces.push_back(fi); ++fi;
    }
    m.patches_.push_back(std::move(inlet));

    // outlet (x = Lx)
    Patch outlet; outlet.name = "outlet"; outlet.type = "outlet";
    for (int j = 0; j < ny; ++j) {
        Face& f = m.faces_[fi];
        f.owner = cid(nx - 1, j); f.neighbor = -1;
        f.patchID = static_cast<int>(m.patches_.size());
        f.center = Vec3(Lx, 0.5 * (yc[j] + yc[j + 1]), 0.5 * dz);
        f.area = (yc[j + 1] - yc[j]) * dz;
        f.normal = Vec3(1, 0, 0);
        m.cells_[f.owner].faces.push_back(fi);
        outlet.faces.push_back(fi); ++fi;
    }
    m.patches_.push_back(std::move(outlet));

    // owner/neighbor lists
    m.ownerList_.resize(m.nInternal_);
    m.neighborList_.resize(m.nInternal_);
    for (int f = 0; f < m.nInternal_; ++f) {
        m.ownerList_[f]    = m.faces_[f].owner;
        m.neighborList_[f] = m.faces_[f].neighbor;
    }

    for (int j = 0; j < ny; ++j) {                                                                                                                                                                                  
        for (int i = 0; i < nx; ++i) {                                                                                                                                                                              
            Cell& c = m.cells_[cid(i, j)];                                                                                                                                                                          
            c.center = Vec3(0.5*(xc[i]+xc[i+1]), 0.5*(yc[j]+yc[j+1]), 0.5*dz);                                                                                                                                      
            c.volume = (xc[i+1]-xc[i]) * (yc[j+1]-yc[j]) * dz;                                                                                                                                               
        }                                                                                                                                                                                                    
    } 
    m.computeFaceGeometry();
    m.ComputeInterpolationWeights();

    std::cout << "Channel2D mesh: " << nx << "x" << ny
              << " (" << m.nCells() << " cells, " << m.nFaces() << " faces)\n";
    return m;
}

// One-sided wall-clustered plate mesh: every refinement cell goes to the
// BOTTOM wall (the top is a far-field boundary, not a viscous wall), so the
// near-wall growth ratio at matched ny is roughly half the two-sided
// channel's. The turbulent omega field's 1/y^2 sublayer tail needs that
// resolution: on the two-sided channel clustering the transported omega
// collapses through the buffer layer and the SST settles into an over-mixed
// state (measured at the shock-interaction baseline bring-up: skin friction
// 35 to 60 percent high, converging as the near-wall growth drops).
Mesh Mesh::makePlate2D(int nx, int ny, double Lx, double Ly, double Re,
                       double yPlusTarget) {
    // friction estimate as makeChannel2D: Cf ~ 0.058 Re^-0.2, U_ref = 1
    double Cf = 0.058 * std::pow(Re, -0.2);
    double u_tau = std::sqrt(Cf / 2.0);
    double nu = (Ly / 2.0) / Re;
    double y1_target = yPlusTarget * nu / u_tau;
    double y1_uniform = Ly / ny;

    // GEOMETRIC growth from the first cell: the tanh mapping concentrates
    // its refinement below the first few y+ and then roughly doubles each
    // cell, leaving the buffer layer (y+ 5 to 30) with a handful of cells at
    // any stretch; a geometric ratio spends the same ny at a near-constant
    // growth (about 1.03 to 1.10 for these targets), which is what the
    // transported near-wall omega needs. Solve r from
    //   y1 (r^ny - 1) / (r - 1) = Ly
    double ratio = 1.0;
    if (y1_target < y1_uniform) {
        auto totalHeight = [&](double r) {
            return y1_target * (std::pow(r, ny) - 1.0) / (r - 1.0);
        };
        double rLo = 1.0 + 1e-9, rHi = 1.5;
        if (totalHeight(rHi) < Ly) {
            std::cout << "  WARNING: Ly unreachable at growth 1.5 with ny="
                      << ny << "; increase ny\n";
            ratio = rHi;
        } else {
            for (int iter = 0; iter < 200; ++iter) {
                double rMid = 0.5 * (rLo + rHi);
                if (totalHeight(rMid) < Ly)
                    rLo = rMid;
                else
                    rHi = rMid;
                if (rHi - rLo < 1e-12) break;
            }
            ratio = 0.5 * (rLo + rHi);
        }
    }
    std::cout << "  Plate mesh: Re=" << Re << " y+_target=" << yPlusTarget
              << " y1_target=" << y1_target << " growth=" << ratio << "\n";

    Mesh m;
    double dz = 1.0;

    std::vector<double> yc(ny + 1);
    if (ratio > 1.0 + 1e-9) {
        yc[0] = 0.0;
        double h = y1_target;
        for (int j = 1; j <= ny; ++j) {
            yc[j] = yc[j - 1] + h;
            h *= ratio;
        }
        // scale out the bisection remainder so the top lands exactly on Ly
        double scale = Ly / yc[ny];
        for (int j = 1; j <= ny; ++j) yc[j] *= scale;
    } else {
        for (int j = 0; j <= ny; ++j)
            yc[j] = Ly * static_cast<double>(j) / ny;
    }

    std::vector<double> xc(nx + 1);
    for (int i = 0; i <= nx; ++i)
        xc[i] = Lx * static_cast<double>(i) / nx;

    int ptsPerPlane = (nx + 1) * (ny + 1);
    m.nodes_.resize(2 * ptsPerPlane);
    auto nid = [&](int i, int j, int k) { return k * ptsPerPlane + j * (nx + 1) + i; };
    for (int j = 0; j <= ny; ++j)
        for (int i = 0; i <= nx; ++i) {
            m.nodes_[nid(i, j, 0)] = Vec3(xc[i], yc[j], 0.0);
            m.nodes_[nid(i, j, 1)] = Vec3(xc[i], yc[j], dz);
        }

    int nC = nx * ny;
    m.cells_.resize(nC);
    auto cid = [&](int i, int j) { return j * nx + i; };

    int nInternalH = nx * (ny - 1);
    int nInternalV = (nx - 1) * ny;
    m.nInternal_ = nInternalH + nInternalV;
    int nBnd = 2 * nx + 2 * ny;
    m.faces_.resize(m.nInternal_ + nBnd);

    int fi = 0;
    for (int j = 0; j < ny - 1; ++j)
        for (int i = 0; i < nx; ++i) {
            Face& f = m.faces_[fi];
            f.owner = cid(i, j); f.neighbor = cid(i, j + 1);
            f.center = Vec3(0.5 * (xc[i] + xc[i + 1]), yc[j + 1], 0.5 * dz);
            f.area = (xc[i + 1] - xc[i]) * dz;
            f.normal = Vec3(0, 1, 0);
            m.cells_[f.owner].faces.push_back(fi);
            m.cells_[f.neighbor].faces.push_back(fi);
            ++fi;
        }
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx - 1; ++i) {
            Face& f = m.faces_[fi];
            f.owner = cid(i, j); f.neighbor = cid(i + 1, j);
            f.center = Vec3(xc[i + 1], 0.5 * (yc[j] + yc[j + 1]), 0.5 * dz);
            f.area = (yc[j + 1] - yc[j]) * dz;
            f.normal = Vec3(1, 0, 0);
            m.cells_[f.owner].faces.push_back(fi);
            m.cells_[f.neighbor].faces.push_back(fi);
            ++fi;
        }

    Patch bottom; bottom.name = "bottom"; bottom.type = "wall";
    for (int i = 0; i < nx; ++i) {
        Face& f = m.faces_[fi];
        f.owner = cid(i, 0); f.neighbor = -1;
        f.patchID = static_cast<int>(m.patches_.size());
        f.center = Vec3(0.5 * (xc[i] + xc[i + 1]), 0.0, 0.5 * dz);
        f.area = (xc[i + 1] - xc[i]) * dz;
        f.normal = Vec3(0, -1, 0);
        m.cells_[f.owner].faces.push_back(fi);
        bottom.faces.push_back(fi); ++fi;
    }
    m.patches_.push_back(std::move(bottom));

    // far-field top: NOT a wall (patch type reflects the physics, so any
    // consumer of the mesh's own wall distance sees only the plate)
    Patch top; top.name = "top"; top.type = "patch";
    for (int i = 0; i < nx; ++i) {
        Face& f = m.faces_[fi];
        f.owner = cid(i, ny - 1); f.neighbor = -1;
        f.patchID = static_cast<int>(m.patches_.size());
        f.center = Vec3(0.5 * (xc[i] + xc[i + 1]), Ly, 0.5 * dz);
        f.area = (xc[i + 1] - xc[i]) * dz;
        f.normal = Vec3(0, 1, 0);
        m.cells_[f.owner].faces.push_back(fi);
        top.faces.push_back(fi); ++fi;
    }
    m.patches_.push_back(std::move(top));

    Patch inlet; inlet.name = "inlet"; inlet.type = "inlet";
    for (int j = 0; j < ny; ++j) {
        Face& f = m.faces_[fi];
        f.owner = cid(0, j); f.neighbor = -1;
        f.patchID = static_cast<int>(m.patches_.size());
        f.center = Vec3(0.0, 0.5 * (yc[j] + yc[j + 1]), 0.5 * dz);
        f.area = (yc[j + 1] - yc[j]) * dz;
        f.normal = Vec3(-1, 0, 0);
        m.cells_[f.owner].faces.push_back(fi);
        inlet.faces.push_back(fi); ++fi;
    }
    m.patches_.push_back(std::move(inlet));

    Patch outlet; outlet.name = "outlet"; outlet.type = "outlet";
    for (int j = 0; j < ny; ++j) {
        Face& f = m.faces_[fi];
        f.owner = cid(nx - 1, j); f.neighbor = -1;
        f.patchID = static_cast<int>(m.patches_.size());
        f.center = Vec3(Lx, 0.5 * (yc[j] + yc[j + 1]), 0.5 * dz);
        f.area = (yc[j + 1] - yc[j]) * dz;
        f.normal = Vec3(1, 0, 0);
        m.cells_[f.owner].faces.push_back(fi);
        outlet.faces.push_back(fi); ++fi;
    }
    m.patches_.push_back(std::move(outlet));

    m.ownerList_.resize(m.nInternal_);
    m.neighborList_.resize(m.nInternal_);
    for (int f = 0; f < m.nInternal_; ++f) {
        m.ownerList_[f] = m.faces_[f].owner;
        m.neighborList_[f] = m.faces_[f].neighbor;
    }
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            Cell& c = m.cells_[cid(i, j)];
            c.center = Vec3(0.5*(xc[i]+xc[i+1]), 0.5*(yc[j]+yc[j+1]), 0.5*dz);
            c.volume = (xc[i+1]-xc[i]) * (yc[j+1]-yc[j]) * dz;
        }
    m.computeFaceGeometry();
    m.ComputeInterpolationWeights();

    std::cout << "Plate2D mesh: " << nx << "x" << ny
              << " (" << m.nCells() << " cells, " << m.nFaces() << " faces)\n";
    return m;
}

// Re-adaptive wall clustering overload
// Computes tanh stretch parameter so the first cell targets yPlusTarget.
Mesh Mesh::makeChannel2D(int nx, int ny, double Lx, double Ly, double Re, double yPlusTarget) {
    // Estimate friction velocity from flat-plate correlation
    // Cf ~ 0.058 * Re^(-0.2)
    // u_tau = sqrt(Cf/2) * U_ref   (U_ref = 1)
    double Cf = 0.058 * std::pow(Re, -0.2);
    double u_tau = std::sqrt(Cf / 2.0);

    // nu = U_ref * L_ref / Re, with U_ref = 1, L_ref = Ly/2
    double nu = (Ly / 2.0) / Re;

    // Target first cell height from y+ definition: y1 = y+ * nu / u_tau
    double y1_target = yPlusTarget * nu / u_tau;

    // Uniform spacing for comparison
    double y1_uniform = Ly / ny;

    double stretch = 0.0; // default: uniform mesh

    if (y1_target < y1_uniform) {
        // Bisect for stretch: larger stretch => smaller first cell
        // First cell height from tanh formula:
        //   y1(s) = 0.5 * Ly * (1 + tanh(s * (2.0/ny - 1)) / tanh(s))
        double sLo = 0.1, sHi = 20.0;

        // Check if target is reachable at max stretch
        auto firstCellHeight = [&](double s) {
            return 0.5 * Ly * (1.0 + std::tanh(s * (2.0 / ny - 1.0))
                                     / std::tanh(s));
        };

        double y1_max_stretch = firstCellHeight(sHi);
        if (y1_target < y1_max_stretch) {
            std::cout << "  WARNING: y1_target=" << y1_target
                      << " unreachable with ny=" << ny
                      << "; using max stretch=" << sHi
                      << " (y1=" << y1_max_stretch << ")\n";
            stretch = sHi;
        } else {
            // Bisection: find stretch where firstCellHeight(s) == y1_target
            for (int iter = 0; iter < 100; ++iter) {
                double sMid = 0.5 * (sLo + sHi);
                double y1_mid = firstCellHeight(sMid);
                if (y1_mid > y1_target)
                    sLo = sMid; // need more stretching
                else
                    sHi = sMid;
                if (sHi - sLo < 1e-10) break;
            }
            stretch = 0.5 * (sLo + sHi);
        }
    }
    // else: y1_target >= y1_uniform => uniform mesh is fine, stretch = 0

    std::cout << "  Adaptive mesh: Re=" << Re << " y+_target=" << yPlusTarget
              << " y1_target=" << y1_target << " stretch=" << stretch << "\n";

    // Build the mesh using the same logic as the original, but with computed stretch
    Mesh m;
    double dz = 1.0;

    // y-coordinates with tanh stretching
    std::vector<double> yc(ny + 1);
    if (stretch > 1e-12) {
        for (int j = 0; j <= ny; ++j) {
            double eta = static_cast<double>(j) / ny;
            yc[j] = 0.5 * Ly * (1.0 + std::tanh(stretch * (2.0 * eta - 1.0))
                                       / std::tanh(stretch));
        }
    } else {
        for (int j = 0; j <= ny; ++j) {
            yc[j] = Ly * static_cast<double>(j) / ny;
        }
    }

    // uniform x-coordinates
    std::vector<double> xc(nx + 1);
    for (int i = 0; i <= nx; ++i) {
        xc[i] = Lx * static_cast<double>(i) / nx;
    }

    // nodes
    int ptsPerPlane = (nx + 1) * (ny + 1);
    m.nodes_.resize(2 * ptsPerPlane);
    auto nid = [&](int i, int j, int k) { return k * ptsPerPlane + j * (nx + 1) + i; };
    for (int j = 0; j <= ny; ++j)
        for (int i = 0; i <= nx; ++i) {
            m.nodes_[nid(i, j, 0)] = Vec3(xc[i], yc[j], 0.0);
            m.nodes_[nid(i, j, 1)] = Vec3(xc[i], yc[j], dz);
        }

    // cells
    int nC = nx * ny;
    m.cells_.resize(nC);
    auto cid = [&](int i, int j) { return j * nx + i; };

    // faces
    int nInternalH = nx * (ny - 1);
    int nInternalV = (nx - 1) * ny;
    m.nInternal_ = nInternalH + nInternalV;
    int nBnd = 2 * nx + 2 * ny;
    m.faces_.resize(m.nInternal_ + nBnd);

    int fi = 0;

    // internal horizontal faces
    for (int j = 0; j < ny - 1; ++j) {
        for (int i = 0; i < nx; ++i) {
            Face& f    = m.faces_[fi];
            f.owner    = cid(i, j);
            f.neighbor = cid(i, j + 1);
            double x0 = xc[i], x1 = xc[i + 1];
            f.center = Vec3(0.5 * (x0 + x1), yc[j + 1], 0.5 * dz);
            f.area   = (x1 - x0) * dz;
            f.normal = Vec3(0, 1, 0);
            m.cells_[f.owner].faces.push_back(fi);
            m.cells_[f.neighbor].faces.push_back(fi);
            ++fi;
        }
    }

    // internal vertical faces
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx - 1; ++i) {
            Face& f    = m.faces_[fi];
            f.owner    = cid(i, j);
            f.neighbor = cid(i + 1, j);
            double y0 = yc[j], y1 = yc[j + 1];
            f.center = Vec3(xc[i + 1], 0.5 * (y0 + y1), 0.5 * dz);
            f.area   = (y1 - y0) * dz;
            f.normal = Vec3(1, 0, 0);
            m.cells_[f.owner].faces.push_back(fi);
            m.cells_[f.neighbor].faces.push_back(fi);
            ++fi;
        }
    }

    // boundary faces — bottom wall
    Patch bottom; bottom.name = "bottom"; bottom.type = "wall";
    for (int i = 0; i < nx; ++i) {
        Face& f = m.faces_[fi];
        f.owner = cid(i, 0); f.neighbor = -1;
        f.patchID = static_cast<int>(m.patches_.size());
        f.center = Vec3(0.5 * (xc[i] + xc[i + 1]), 0.0, 0.5 * dz);
        f.area = (xc[i + 1] - xc[i]) * dz;
        f.normal = Vec3(0, -1, 0);
        m.cells_[f.owner].faces.push_back(fi);
        bottom.faces.push_back(fi); ++fi;
    }
    m.patches_.push_back(std::move(bottom));

    // top wall
    Patch top; top.name = "top"; top.type = "wall";
    for (int i = 0; i < nx; ++i) {
        Face& f = m.faces_[fi];
        f.owner = cid(i, ny - 1); f.neighbor = -1;
        f.patchID = static_cast<int>(m.patches_.size());
        f.center = Vec3(0.5 * (xc[i] + xc[i + 1]), Ly, 0.5 * dz);
        f.area = (xc[i + 1] - xc[i]) * dz;
        f.normal = Vec3(0, 1, 0);
        m.cells_[f.owner].faces.push_back(fi);
        top.faces.push_back(fi); ++fi;
    }
    m.patches_.push_back(std::move(top));

    // inlet
    Patch inlet; inlet.name = "inlet"; inlet.type = "inlet";
    for (int j = 0; j < ny; ++j) {
        Face& f = m.faces_[fi];
        f.owner = cid(0, j); f.neighbor = -1;
        f.patchID = static_cast<int>(m.patches_.size());
        f.center = Vec3(0.0, 0.5 * (yc[j] + yc[j + 1]), 0.5 * dz);
        f.area = (yc[j + 1] - yc[j]) * dz;
        f.normal = Vec3(-1, 0, 0);
        m.cells_[f.owner].faces.push_back(fi);
        inlet.faces.push_back(fi); ++fi;
    }
    m.patches_.push_back(std::move(inlet));

    // outlet
    Patch outlet; outlet.name = "outlet"; outlet.type = "outlet";
    for (int j = 0; j < ny; ++j) {
        Face& f = m.faces_[fi];
        f.owner = cid(nx - 1, j); f.neighbor = -1;
        f.patchID = static_cast<int>(m.patches_.size());
        f.center = Vec3(Lx, 0.5 * (yc[j] + yc[j + 1]), 0.5 * dz);
        f.area = (yc[j + 1] - yc[j]) * dz;
        f.normal = Vec3(1, 0, 0);
        m.cells_[f.owner].faces.push_back(fi);
        outlet.faces.push_back(fi); ++fi;
    }
    m.patches_.push_back(std::move(outlet));

    // owner/neighbor lists
    m.ownerList_.resize(m.nInternal_);
    m.neighborList_.resize(m.nInternal_);
    for (int f = 0; f < m.nInternal_; ++f) {
        m.ownerList_[f]    = m.faces_[f].owner;
        m.neighborList_[f] = m.faces_[f].neighbor;
    }

    // cell centers and volumes
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            Cell& c = m.cells_[cid(i, j)];
            c.center = Vec3(0.5*(xc[i]+xc[i+1]), 0.5*(yc[j]+yc[j+1]), 0.5*dz);
            c.volume = (xc[i+1]-xc[i]) * (yc[j+1]-yc[j]) * dz;
        }
    }
    m.computeFaceGeometry();
    m.ComputeInterpolationWeights();

    std::cout << "Channel2D mesh: " << nx << "x" << ny
              << " (" << m.nCells() << " cells, " << m.nFaces() << " faces)\n";
    return m;
}

// Backward-facing step (non-adaptive): default tanh stretch = 2.0
Mesh Mesh::makeBackwardFacingStep2D(int nx_up, int nx_down, int ny_up, int ny_down,
                                     double Lu, double Ld, double h_s, double H) {
    return makeBackwardFacingStep2D(nx_up, nx_down, ny_up, ny_down, Lu, Ld, h_s, H, 0.0, 1.0);
}

// Backward-facing step with Re-adaptive y+ clustering.
// Upper block: y in [h_s, H] with symmetric tanh (near both step lip and top wall).
// Lower block: y in [0, h_s] with one-sided tanh toward y=0 (bottom wall).
// Six named patches: top_wall, bottom_wall_up, step_face, bottom_wall_down, inlet, outlet.
Mesh Mesh::makeBackwardFacingStep2D(int nx_up, int nx_down, int ny_up, int ny_down,
                                     double Lu, double Ld, double h_s, double H,
                                     double Re, double yPlusTarget) {
    const int nx_tot = nx_up + nx_down;
    const double dz  = 1.0;
    const double H_up = H - h_s;

    // --- x-coordinates ---
    std::vector<double> xc(nx_tot + 1);
    for (int i = 0; i <= nx_up; ++i)
        xc[i] = -Lu + Lu * i / nx_up;
    for (int i = 1; i <= nx_down; ++i)
        xc[nx_up + i] = Ld * i / nx_down;

    // --- y-coordinates: upper block [h_s, H], symmetric tanh toward both walls ---
    double stretch_up = 2.0;
    if (Re > 0) {
        double Cf    = 0.058 * std::pow(Re, -0.2);
        double u_tau = std::sqrt(Cf / 2.0);
        double nu    = H_up / Re;
        double y1t   = yPlusTarget * nu / u_tau;
        double y1u   = H_up / ny_up;
        if (y1t < y1u) {
            auto fc = [&](double s) {
                return 0.5 * H_up * (1.0 + std::tanh(s*(2.0/ny_up - 1.0)) / std::tanh(s));
            };
            double sLo = 0.1, sHi = 20.0;
            if (y1t < fc(sHi)) {
                stretch_up = sHi;
            } else {
                for (int it = 0; it < 100; ++it) {
                    double sm = 0.5*(sLo+sHi);
                    if (fc(sm) > y1t) sLo = sm; else sHi = sm;
                    if (sHi-sLo < 1e-10) break;
                }
                stretch_up = 0.5*(sLo+sHi);
            }
        }
    }
    std::vector<double> yc_up(ny_up + 1);
    for (int j = 0; j <= ny_up; ++j) {
        double eta  = static_cast<double>(j) / ny_up;
        yc_up[j] = h_s + 0.5*H_up*(1.0 + std::tanh(stretch_up*(2.0*eta-1.0))/std::tanh(stretch_up));
    }

    // --- y-coordinates: lower block [0, h_s], one-sided tanh toward y=0 ---
    // Uses the same symmetric-tanh formula as the channel, applied to the lower half:
    //   yc_low[j] = 0.5*h_s * (1 + tanh(s*(2*j/ny_down - 1)) / tanh(s))
    // which clusters cells near both j=0 and j=ny_down; we accept clustering at
    // the h_s interface too (it is the shear-layer side and benefits from resolution).
    double stretch_low = 2.0;
    if (Re > 0) {
        double Cf    = 0.058 * std::pow(Re, -0.2);
        double u_tau = std::sqrt(Cf / 2.0);
        double nu    = h_s / Re;
        double y1t   = yPlusTarget * nu / u_tau;
        double y1u   = h_s / ny_down;
        if (y1t < y1u) {
            // fc(s) = first cell height (distance from y=0 to yc_low[1])
            auto fc = [&](double s) {
                return 0.5 * h_s * (1.0 + std::tanh(s*(2.0/ny_down - 1.0)) / std::tanh(s));
            };
            double sLo = 0.1, sHi = 20.0;
            if (y1t < fc(sHi)) {
                stretch_low = sHi;
            } else {
                for (int it = 0; it < 100; ++it) {
                    double sm = 0.5*(sLo+sHi);
                    if (fc(sm) > y1t) sLo = sm; else sHi = sm;
                    if (sHi-sLo < 1e-10) break;
                }
                stretch_low = 0.5*(sLo+sHi);
            }
        }
    }
    std::vector<double> yc_low(ny_down + 1);
    for (int j = 0; j <= ny_down; ++j) {
        double eta = static_cast<double>(j) / ny_down;
        yc_low[j] = 0.5*h_s*(1.0 + std::tanh(stretch_low*(2.0*eta-1.0))/std::tanh(stretch_low));
    }

    // Cell ID helpers
    auto cid_up  = [&](int i, int j) { return j * nx_tot + i; };
    auto cid_low = [&](int i, int j) { return nx_tot * ny_up + j * nx_down + i; };

    // Face counts
    int nHup    = nx_tot * (ny_up - 1);
    int nHlow   = nx_down * (ny_down - 1);
    int nCouple = nx_down;
    int nVup    = (nx_tot - 1) * ny_up;
    int nVlow   = (nx_down - 1) * ny_down;
    int nInternal = nHup + nHlow + nCouple + nVup + nVlow;
    int nBnd = nx_tot + nx_up + ny_down + nx_down + ny_up + (ny_up + ny_down);

    Mesh m;
    int nC = nx_tot * ny_up + nx_down * ny_down;
    m.cells_.resize(nC);
    m.faces_.resize(nInternal + nBnd);
    m.nInternal_ = nInternal;

    int fi = 0;

    // Internal: horizontal upper block
    for (int j = 0; j < ny_up - 1; ++j) {
        for (int i = 0; i < nx_tot; ++i) {
            Face& f = m.faces_[fi];
            f.owner = cid_up(i,j); f.neighbor = cid_up(i,j+1);
            f.center = Vec3(0.5*(xc[i]+xc[i+1]), yc_up[j+1], 0.5*dz);
            f.area = (xc[i+1]-xc[i])*dz; f.normal = Vec3(0,1,0);
            m.cells_[f.owner].faces.push_back(fi);
            m.cells_[f.neighbor].faces.push_back(fi); ++fi;
        }
    }
    // Internal: horizontal lower block
    for (int j = 0; j < ny_down - 1; ++j) {
        for (int i = 0; i < nx_down; ++i) {
            Face& f = m.faces_[fi];
            f.owner = cid_low(i,j); f.neighbor = cid_low(i,j+1);
            f.center = Vec3(0.5*(xc[nx_up+i]+xc[nx_up+i+1]), yc_low[j+1], 0.5*dz);
            f.area = (xc[nx_up+i+1]-xc[nx_up+i])*dz; f.normal = Vec3(0,1,0);
            m.cells_[f.owner].faces.push_back(fi);
            m.cells_[f.neighbor].faces.push_back(fi); ++fi;
        }
    }
    // Internal: coupling faces at y=h_s (owner=upper j=0, normal downward toward lower)
    for (int i = 0; i < nx_down; ++i) {
        Face& f = m.faces_[fi];
        f.owner = cid_up(nx_up+i, 0); f.neighbor = cid_low(i, ny_down-1);
        f.center = Vec3(0.5*(xc[nx_up+i]+xc[nx_up+i+1]), h_s, 0.5*dz);
        f.area = (xc[nx_up+i+1]-xc[nx_up+i])*dz; f.normal = Vec3(0,-1,0);
        m.cells_[f.owner].faces.push_back(fi);
        m.cells_[f.neighbor].faces.push_back(fi); ++fi;
    }
    // Internal: vertical upper block
    for (int j = 0; j < ny_up; ++j) {
        for (int i = 0; i < nx_tot-1; ++i) {
            Face& f = m.faces_[fi];
            f.owner = cid_up(i,j); f.neighbor = cid_up(i+1,j);
            f.center = Vec3(xc[i+1], 0.5*(yc_up[j]+yc_up[j+1]), 0.5*dz);
            f.area = (yc_up[j+1]-yc_up[j])*dz; f.normal = Vec3(1,0,0);
            m.cells_[f.owner].faces.push_back(fi);
            m.cells_[f.neighbor].faces.push_back(fi); ++fi;
        }
    }
    // Internal: vertical lower block
    for (int j = 0; j < ny_down; ++j) {
        for (int i = 0; i < nx_down-1; ++i) {
            Face& f = m.faces_[fi];
            f.owner = cid_low(i,j); f.neighbor = cid_low(i+1,j);
            f.center = Vec3(xc[nx_up+i+1], 0.5*(yc_low[j]+yc_low[j+1]), 0.5*dz);
            f.area = (yc_low[j+1]-yc_low[j])*dz; f.normal = Vec3(1,0,0);
            m.cells_[f.owner].faces.push_back(fi);
            m.cells_[f.neighbor].faces.push_back(fi); ++fi;
        }
    }

    // Boundary: top_wall
    {
        Patch p; p.name="top_wall"; p.type="wall";
        for (int i=0; i<nx_tot; ++i) {
            Face& f=m.faces_[fi]; f.owner=cid_up(i,ny_up-1); f.neighbor=-1;
            f.patchID=static_cast<int>(m.patches_.size());
            f.center=Vec3(0.5*(xc[i]+xc[i+1]),H,0.5*dz); f.area=(xc[i+1]-xc[i])*dz; f.normal=Vec3(0,1,0);
            m.cells_[f.owner].faces.push_back(fi); p.faces.push_back(fi); ++fi;
        }
        m.patches_.push_back(std::move(p));
    }
    // Boundary: bottom_wall_up (upstream floor at y=h_s)
    {
        Patch p; p.name="bottom_wall_up"; p.type="wall";
        for (int i=0; i<nx_up; ++i) {
            Face& f=m.faces_[fi]; f.owner=cid_up(i,0); f.neighbor=-1;
            f.patchID=static_cast<int>(m.patches_.size());
            f.center=Vec3(0.5*(xc[i]+xc[i+1]),h_s,0.5*dz); f.area=(xc[i+1]-xc[i])*dz; f.normal=Vec3(0,-1,0);
            m.cells_[f.owner].faces.push_back(fi); p.faces.push_back(fi); ++fi;
        }
        m.patches_.push_back(std::move(p));
    }
    // Boundary: step_face (x=0, y in [0,h_s])
    {
        Patch p; p.name="step_face"; p.type="wall";
        for (int j=0; j<ny_down; ++j) {
            Face& f=m.faces_[fi]; f.owner=cid_low(0,j); f.neighbor=-1;
            f.patchID=static_cast<int>(m.patches_.size());
            f.center=Vec3(0.0,0.5*(yc_low[j]+yc_low[j+1]),0.5*dz); f.area=(yc_low[j+1]-yc_low[j])*dz; f.normal=Vec3(-1,0,0);
            m.cells_[f.owner].faces.push_back(fi); p.faces.push_back(fi); ++fi;
        }
        m.patches_.push_back(std::move(p));
    }
    // Boundary: bottom_wall_down (y=0)
    {
        Patch p; p.name="bottom_wall_down"; p.type="wall";
        for (int i=0; i<nx_down; ++i) {
            Face& f=m.faces_[fi]; f.owner=cid_low(i,0); f.neighbor=-1;
            f.patchID=static_cast<int>(m.patches_.size());
            f.center=Vec3(0.5*(xc[nx_up+i]+xc[nx_up+i+1]),0.0,0.5*dz); f.area=(xc[nx_up+i+1]-xc[nx_up+i])*dz; f.normal=Vec3(0,-1,0);
            m.cells_[f.owner].faces.push_back(fi); p.faces.push_back(fi); ++fi;
        }
        m.patches_.push_back(std::move(p));
    }
    // Boundary: inlet
    {
        Patch p; p.name="inlet"; p.type="inlet";
        for (int j=0; j<ny_up; ++j) {
            Face& f=m.faces_[fi]; f.owner=cid_up(0,j); f.neighbor=-1;
            f.patchID=static_cast<int>(m.patches_.size());
            f.center=Vec3(-Lu,0.5*(yc_up[j]+yc_up[j+1]),0.5*dz); f.area=(yc_up[j+1]-yc_up[j])*dz; f.normal=Vec3(-1,0,0);
            m.cells_[f.owner].faces.push_back(fi); p.faces.push_back(fi); ++fi;
        }
        m.patches_.push_back(std::move(p));
    }
    // Boundary: outlet (upper + lower)
    {
        Patch p; p.name="outlet"; p.type="outlet";
        for (int j=0; j<ny_up; ++j) {
            Face& f=m.faces_[fi]; f.owner=cid_up(nx_tot-1,j); f.neighbor=-1;
            f.patchID=static_cast<int>(m.patches_.size());
            f.center=Vec3(Ld,0.5*(yc_up[j]+yc_up[j+1]),0.5*dz); f.area=(yc_up[j+1]-yc_up[j])*dz; f.normal=Vec3(1,0,0);
            m.cells_[f.owner].faces.push_back(fi); p.faces.push_back(fi); ++fi;
        }
        for (int j=0; j<ny_down; ++j) {
            Face& f=m.faces_[fi]; f.owner=cid_low(nx_down-1,j); f.neighbor=-1;
            f.patchID=static_cast<int>(m.patches_.size());
            f.center=Vec3(Ld,0.5*(yc_low[j]+yc_low[j+1]),0.5*dz); f.area=(yc_low[j+1]-yc_low[j])*dz; f.normal=Vec3(1,0,0);
            m.cells_[f.owner].faces.push_back(fi); p.faces.push_back(fi); ++fi;
        }
        m.patches_.push_back(std::move(p));
    }

    // Owner/neighbor lists
    m.ownerList_.resize(nInternal);
    m.neighborList_.resize(nInternal);
    for (int f = 0; f < nInternal; ++f) {
        m.ownerList_[f]    = m.faces_[f].owner;
        m.neighborList_[f] = m.faces_[f].neighbor;
    }

    // Cell centers and volumes
    for (int j=0; j<ny_up; ++j)
        for (int i=0; i<nx_tot; ++i) {
            Cell& c = m.cells_[cid_up(i,j)];
            c.center = Vec3(0.5*(xc[i]+xc[i+1]), 0.5*(yc_up[j]+yc_up[j+1]), 0.5*dz);
            c.volume = (xc[i+1]-xc[i])*(yc_up[j+1]-yc_up[j])*dz;
        }
    for (int j=0; j<ny_down; ++j)
        for (int i=0; i<nx_down; ++i) {
            Cell& c = m.cells_[cid_low(i,j)];
            c.center = Vec3(0.5*(xc[nx_up+i]+xc[nx_up+i+1]), 0.5*(yc_low[j]+yc_low[j+1]), 0.5*dz);
            c.volume = (xc[nx_up+i+1]-xc[nx_up+i])*(yc_low[j+1]-yc_low[j])*dz;
        }

    m.computeFaceGeometry();
    m.ComputeInterpolationWeights();

    std::cout << "BackwardFacingStep2D: [" << nx_up << "+" << nx_down << "]x["
              << ny_up << "+" << ny_down << "]"
              << " (" << nC << " cells, " << m.nFaces() << " faces)\n";
    return m;
}