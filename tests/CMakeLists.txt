# Tests for QBTM.  Registered with CTest via the top-level enable_testing().
#
# Layout:
#   tests/cpp/        — C++ unit/smoke tests linked against btm_* libraries
#   tests/python/     — pytest-driven tests (run separately via pytest)
#   tests/regression/ — saved baseline outputs (no CMake target; consumed by
#                       Python tests that compare current outputs to baselines)
#
# Conventions:
#   - Every C++ test executable returns 0 on success, non-zero on failure.
#   - Tests must be deterministic (use fixed seeds; no wall-clock dependence).
#   - Tests must run quickly (target: < ~30 s each on a laptop).

# ---- mesh_bfs: BFS mesh factory smoke test ----
add_executable(test_mesh_bfs cpp/test_mesh_bfs.cpp)
target_link_libraries(test_mesh_bfs PRIVATE btm_core)
add_test(NAME mesh_bfs COMMAND test_mesh_bfs)
set_tests_properties(mesh_bfs PROPERTIES TIMEOUT 30 LABELS "unit;mesh")

# ---- couette_bc: moving-wall (Couette) boundary-condition test ----
add_executable(test_couette_bc cpp/test_couette_bc.cpp)
target_link_libraries(test_couette_bc PRIVATE btm_core)
add_test(NAME couette_bc COMMAND test_couette_bc)
set_tests_properties(couette_bc PROPERTIES TIMEOUT 30 LABELS "unit;bc")

# ---- bfs_lemoin_bc: free-slip top + profile inflow (Le-Moin BFS) BC test ----
add_executable(test_bfs_lemoin_bc cpp/test_bfs_lemoin_bc.cpp)
target_link_libraries(test_bfs_lemoin_bc PRIVATE btm_core)
add_test(NAME bfs_lemoin_bc COMMAND test_bfs_lemoin_bc)
set_tests_properties(bfs_lemoin_bc PROPERTIES TIMEOUT 30 LABELS "unit;bc")

# ---- transpose_stress: Boussinesq transpose-stress deferred correction ----
add_executable(test_transpose_stress cpp/test_transpose_stress.cpp)
target_link_libraries(test_transpose_stress PRIVATE btm_core)
add_test(NAME transpose_stress COMMAND test_transpose_stress)
set_tests_properties(transpose_stress PROPERTIES TIMEOUT 30 LABELS "unit;solver")

# ---- sst_production_2003: corrected omega production (limiter-discriminating) ----
add_executable(test_sst_production_2003 cpp/test_sst_production_2003.cpp)
target_link_libraries(test_sst_production_2003 PRIVATE btm_core)
add_test(NAME sst_production_2003 COMMAND test_sst_production_2003)
set_tests_properties(sst_production_2003 PROPERTIES TIMEOUT 30 LABELS "unit;sst")

# ---- cp_normalization: pressure-coefficient references (audit fix) ----
add_executable(test_cp_normalization cpp/test_cp_normalization.cpp)
target_link_libraries(test_cp_normalization PRIVATE btm_core)
add_test(NAME cp_normalization COMMAND test_cp_normalization)
set_tests_properties(cp_normalization PROPERTIES TIMEOUT 30 LABELS "unit;observation")

# ---- infra_fixes: Gauss-Seidel sweep, inBounds guard, cache copy, RC probe ----
add_executable(test_infra_fixes cpp/test_infra_fixes.cpp)
target_link_libraries(test_infra_fixes PRIVATE btm_incompressible)
add_test(NAME infra_fixes COMMAND test_infra_fixes)
set_tests_properties(infra_fixes PROPERTIES TIMEOUT 240 LABELS "unit;infra")

# ---- injection_source: Reynolds-stress injection manufactured-field test ----
add_executable(test_injection_source cpp/test_injection_source.cpp)
target_link_libraries(test_injection_source PRIVATE btm_incompressible)
add_test(NAME injection_source COMMAND test_injection_source)
set_tests_properties(injection_source PROPERTIES TIMEOUT 30 LABELS "unit;injection")

# ---- nut_floor_release: startup-only floor + molecular wall-stress observable ----
add_executable(test_nut_floor_release cpp/test_nut_floor_release.cpp)
target_link_libraries(test_nut_floor_release PRIVATE btm_incompressible)
add_test(NAME nut_floor_release COMMAND test_nut_floor_release)
set_tests_properties(nut_floor_release PROPERTIES TIMEOUT 240 LABELS "unit;solver;sst")

# ---- convergence_accounting: carried turbulence norms under update intervals ----
add_executable(test_convergence_accounting cpp/test_convergence_accounting.cpp)
target_link_libraries(test_convergence_accounting PRIVATE btm_incompressible)
add_test(NAME convergence_accounting COMMAND test_convergence_accounting)

add_executable(test_state_validation cpp/test_state_validation.cpp)
target_link_libraries(test_state_validation PRIVATE btm_compressible)
add_test(NAME state_validation COMMAND test_state_validation)
set_tests_properties(convergence_accounting PROPERTIES TIMEOUT 240 LABELS "unit;solver")

# ---- periodic_channel: streamwise-periodic curved-channel mesh + body force ----
add_executable(test_periodic_channel cpp/test_periodic_channel.cpp)
target_link_libraries(test_periodic_channel PRIVATE btm_incompressible)
add_test(NAME periodic_channel COMMAND test_periodic_channel)
set_tests_properties(periodic_channel PROPERTIES TIMEOUT 240 LABELS "unit;mesh;solver")

# ---- compressible_smoke: Ma=0.1 channel smoke convergence ----
add_executable(test_compressible_smoke cpp/test_compressible_smoke.cpp)
target_link_libraries(test_compressible_smoke PRIVATE btm_compressible)
add_test(NAME compressible_smoke COMMAND test_compressible_smoke)

add_executable(test_two_pressure_contract cpp/test_two_pressure_contract.cpp)
target_link_libraries(test_two_pressure_contract PRIVATE btm_compressible)
add_test(NAME two_pressure_contract COMMAND test_two_pressure_contract)

add_executable(test_compressible_observation_contract
               cpp/test_compressible_observation_contract.cpp)
target_link_libraries(test_compressible_observation_contract PRIVATE btm_compressible)
add_test(NAME compressible_observation_contract
         COMMAND test_compressible_observation_contract)
set_tests_properties(compressible_observation_contract PROPERTIES
    TIMEOUT 30
    LABELS "unit;observation;compressible")
set_tests_properties(compressible_smoke PROPERTIES
    TIMEOUT 180
    LABELS "smoke;compressible")

# ---- dbns_numerics: density-based solver foundational numerics (header-only) ----
add_executable(test_dbns_numerics cpp/test_dbns_numerics.cpp)
target_include_directories(test_dbns_numerics PRIVATE ${CMAKE_SOURCE_DIR}/dbns/include)
target_link_libraries(test_dbns_numerics PRIVATE btm_compressible)
add_test(NAME dbns_numerics COMMAND test_dbns_numerics)
set_tests_properties(dbns_numerics PROPERTIES TIMEOUT 30 LABELS "unit;dbns")

# ---- dbns_verification: shock tube vs exact Riemann + MMS order of accuracy ----
add_executable(test_dbns_verification cpp/test_dbns_verification.cpp)
target_link_libraries(test_dbns_verification PRIVATE btm_dbns)
add_test(NAME dbns_verification COMMAND test_dbns_verification)
set_tests_properties(dbns_verification PROPERTIES TIMEOUT 240 LABELS "verification;dbns")

# ---- dbns_verification2: oblique-shock reflection + wall observation operator ----
add_executable(test_dbns_verification2 cpp/test_dbns_verification2.cpp)
target_link_libraries(test_dbns_verification2 PRIVATE btm_dbns)
add_test(NAME dbns_verification2 COMMAND test_dbns_verification2)
set_tests_properties(dbns_verification2 PROPERTIES TIMEOUT 120 LABELS "verification;dbns")

# ---- dbns_implicit: LU-SGS viscous-steady rungs (Couette exact + laminar plate) ----
add_executable(test_dbns_implicit cpp/test_dbns_implicit.cpp)
target_link_libraries(test_dbns_implicit PRIVATE btm_dbns)
add_test(NAME dbns_implicit COMMAND test_dbns_implicit)
set_tests_properties(dbns_implicit PROPERTIES TIMEOUT 480 LABELS "verification;dbns")

# ---- dbns_injection: deferred-correction coupling with the heat-flux reach ----
add_executable(test_dbns_injection cpp/test_dbns_injection.cpp)
target_link_libraries(test_dbns_injection PRIVATE btm_dbns)
add_test(NAME dbns_injection COMMAND test_dbns_injection)
set_tests_properties(dbns_injection PROPERTIES TIMEOUT 480 LABELS "verification;dbns")

# ---- dbns_sst_wall: omega wall anchoring sustains an imposed turbulent layer ----
add_executable(test_dbns_sst_wall cpp/test_dbns_sst_wall.cpp)
target_link_libraries(test_dbns_sst_wall PRIVATE btm_dbns)
add_test(NAME dbns_sst_wall COMMAND test_dbns_sst_wall)
set_tests_properties(dbns_sst_wall PROPERTIES TIMEOUT 600 LABELS "verification;dbns")

# ---- pytest target ----
# Optional convenience target; the canonical way to run Python tests is
# `pytest tests/python` from the repo root after building the .so.
find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    add_test(NAME pytest
        COMMAND ${Python3_EXECUTABLE} -m pytest -q
                ${CMAKE_CURRENT_SOURCE_DIR}/python
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    set_tests_properties(pytest PROPERTIES
        TIMEOUT 300
        LABELS "python"
        ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_SOURCE_DIR}/python")
endif()
