cmake_minimum_required(VERSION 3.27)

find_package(Torch REQUIRED)
find_package(Python3 COMPONENTS Development REQUIRED)

# === Library sources ===
add_library(flash_attn SHARED
    mps/flash_attn.mm       # C++ PyTorch operator registration (main entry)
)

# === Include directories ===
target_include_directories(flash_attn PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/mps
    ${TORCH_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
)

# === Link Torch, Python, and any other dependencies ===
target_link_libraries(flash_attn PRIVATE
    ${TORCH_LIBRARIES}
    ${Python3_LIBRARIES}
)

# === Find libtorch_python (for pybind11/type_caster support) ===
find_library(TORCH_PYTHON_LIBRARY
    NAMES torch_python
    PATHS
        ${CMAKE_SOURCE_DIR}/../pytorch/build/lib
        ${CMAKE_SOURCE_DIR}/../pytorch/torch/lib
        /Users/ianreitsma/projects/orchard/pytorch/build/lib
        /Users/ianreitsma/projects/orchard/pytorch/torch/lib
    NO_DEFAULT_PATH
    REQUIRED
)
if (NOT TORCH_PYTHON_LIBRARY)
    message(FATAL_ERROR "Could not find libtorch_python.dylib – needed for the pybind11 type_casters")
endif()
message(STATUS "Linking against: ${TORCH_PYTHON_LIBRARY}")
target_link_libraries(flash_attn PRIVATE ${TORCH_PYTHON_LIBRARY})

# === C++ and ObjC++ standard enforcement ===
target_compile_features(flash_attn PRIVATE cxx_std_20)
set_property(TARGET flash_attn PROPERTY OBJCXX_STANDARD 20)

# === Output settings ===
set_target_properties(flash_attn PROPERTIES
    PREFIX ""
    OUTPUT_NAME "libflash_attn"
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/kernel_lib/flashattn
)

# === Build output status messages (debugging) ===
message(STATUS "PYTHON EXECUTABLE: ${Python3_EXECUTABLE}")
message(STATUS "PYTHON INCLUDE DIRS: ${Python3_INCLUDE_DIRS}")
message(STATUS "TORCH LIB DIRS: ${TORCH_LIBRARY_DIRS}")
