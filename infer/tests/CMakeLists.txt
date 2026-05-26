# Tests for hand_pipeline. Depends on the parent CMakeLists.txt having
# defined the `hand_pipeline` target (include / link interface auto-propagated).
#
# Naming convention: each test entry source is `unit_<name>_test.cpp`.
# Non-test helpers (COCO parser, IoU / PCK) keep their generic names.

# REPO_ROOT: from external/hand_model_rknn_pipeline/infer/tests/ → 4 levels up.
get_filename_component(REPO_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
set(NLOHMANN_INCLUDE_DIR
    "${REPO_ROOT}/external/nlohmann_json/single_include")
if(NOT EXISTS "${NLOHMANN_INCLUDE_DIR}/nlohmann/json.hpp")
    message(FATAL_ERROR
        "nlohmann/json.hpp not found at ${NLOHMANN_INCLUDE_DIR}")
endif()

add_executable(unit_bench_e2e_test
    unit_bench_e2e_test.cpp
    eval.cpp
)
target_include_directories(unit_bench_e2e_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}    # for "eval.h"
    ${NLOHMANN_INCLUDE_DIR}
)
target_link_libraries(unit_bench_e2e_test PRIVATE hand_pipeline)

set_target_properties(unit_bench_e2e_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
