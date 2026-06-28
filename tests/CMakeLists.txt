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

# ---- compressible_smoke: Ma=0.1 channel smoke convergence ----
add_executable(test_compressible_smoke cpp/test_compressible_smoke.cpp)
target_link_libraries(test_compressible_smoke PRIVATE btm_compressible)
add_test(NAME compressible_smoke COMMAND test_compressible_smoke)
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
