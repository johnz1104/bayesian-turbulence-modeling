cmake_minimum_required(VERSION 3.14)
project(rans_sst LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Compile everything position-independent. The btm_* static libraries are linked
# into the shared pybind11 module rans_sst_py, and on Linux (ELF x86-64) linking
# non-PIC static objects into a shared object fails with an -fPIC relocation
# error; macOS does not require this, so it must be set explicitly for portability.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG   "-g -O0 -Wall -Wextra -Wpedantic")

# Testing is opt-in but on by default for top-level builds; downstream
# consumers using add_subdirectory() see BUILD_TESTING=OFF unless they enable it.
option(BUILD_TESTING "Build unit/regression tests (CTest)" ON)
if(BUILD_TESTING)
    enable_testing()
endif()

# ---- btm_core: physics-agnostic infrastructure ----
# Mesh, Field, LinearSolver, BoundaryCondition, SSTModel,
# ObservationOperator (header-only), InferenceParameters (header-only).
add_library(btm_core STATIC
    core/src/Mesh.cpp
    core/src/SSTModel.cpp
)
target_include_directories(btm_core PUBLIC core/include)
target_link_libraries(btm_core PUBLIC pthread)

# ---- btm_incompressible: incompressible SIMPLE solver ----
add_library(btm_incompressible STATIC
    incompressible/src/SIMPLESolver.cpp
    incompressible/src/ForwardModel.cpp
    incompressible/src/ParameterSensitivity.cpp
)
# Expose both core and solver-specific headers to downstream targets.
target_include_directories(btm_incompressible PUBLIC
    core/include
    incompressible/include
)
target_link_libraries(btm_incompressible PUBLIC btm_core)

# ---- Incompressible CLI executable ----
add_executable(rans_sst incompressible/src/Main.cpp)
target_link_libraries(rans_sst PRIVATE btm_incompressible)

# ---- btm_compressible: pressure-based compressible RANS solver ----
# Extends btm_core with ideal-gas EOS, Sutherland viscosity, energy equation,
# and density-weighted SIMPLE for Ma < ~0.8 flows.
add_library(btm_compressible STATIC
    compressible/src/CompressibleSIMPLESolver.cpp
    compressible/src/CompressibleForwardModel.cpp
)
target_include_directories(btm_compressible PUBLIC
    core/include
    incompressible/include    # ObservationOperator, EvaluationTypes
    compressible/include
)
target_link_libraries(btm_compressible PUBLIC btm_incompressible)

# ---- Compressible CLI executable ----
add_executable(rans_sst_compressible compressible/src/CompressibleMain.cpp)
target_link_libraries(rans_sst_compressible PRIVATE btm_compressible)

# ---- btm_dbns: density-based, shock-capturing compressible RANS solver ----
# Additive high-speed path (HLLC + MUSCL + SSP-RK + SST), separate from the
# low-Mach pressure-based btm_compressible.  Reuses btm_core (Mesh, Field,
# SSTModel) and the compressible IdealGasEOS header.
add_library(btm_dbns STATIC
    dbns/src/DBNSSolver.cpp
)
target_include_directories(btm_dbns PUBLIC
    core/include
    compressible/include    # IdealGasEOS
    dbns/include
)
target_link_libraries(btm_dbns PUBLIC btm_core)

# ---- Optional: pybind11 Python bindings ----
option(BUILD_PYTHON_BINDINGS "Build pybind11 Python module" OFF)
if(BUILD_PYTHON_BINDINGS)
    set(PYBIND11_FINDPYTHON ON)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.12.0
    )
    FetchContent_MakeAvailable(pybind11)

    pybind11_add_module(rans_sst_py python/Bindings.cpp)
    # btm_compressible transitively pulls in btm_incompressible and btm_core,
    # so the bindings TU sees both incompressible and compressible headers.
    # btm_dbns adds the density-based shock-capturing solver path.
    target_link_libraries(rans_sst_py PRIVATE btm_compressible btm_dbns)
    target_include_directories(rans_sst_py PRIVATE dbns/include)
endif()

# ---- Tests (CTest) ----
if(BUILD_TESTING AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt)
    add_subdirectory(tests)
endif()
