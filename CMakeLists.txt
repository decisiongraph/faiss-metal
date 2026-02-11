cmake_minimum_required(VERSION 3.24)
project(faiss-metal LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 17)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

# --- Dependencies ---
find_package(faiss REQUIRED)
find_library(METAL_FW Metal REQUIRED)
find_library(MPS_FW MetalPerformanceShaders REQUIRED)
find_library(FOUNDATION_FW Foundation REQUIRED)

# --- Compile .metal shaders → .metallib ---
file(GLOB METAL_SOURCES ${CMAKE_SOURCE_DIR}/shaders/*.metal)

# Compile each .metal to .air, then link into one .metallib
set(METAL_AIR_FILES "")
foreach(METAL_SRC ${METAL_SOURCES})
    get_filename_component(METAL_NAME ${METAL_SRC} NAME_WE)
    set(AIR_FILE ${CMAKE_BINARY_DIR}/${METAL_NAME}.air)
    add_custom_command(
        OUTPUT ${AIR_FILE}
        COMMAND xcrun -sdk macosx metal -std=metal3.0 -c ${METAL_SRC} -o ${AIR_FILE}
        DEPENDS ${METAL_SRC}
        COMMENT "Compiling ${METAL_NAME}.metal → .air"
    )
    list(APPEND METAL_AIR_FILES ${AIR_FILE})
endforeach()

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/faiss_metal.metallib
    COMMAND xcrun -sdk macosx metallib ${METAL_AIR_FILES} -o ${CMAKE_BINARY_DIR}/faiss_metal.metallib
    DEPENDS ${METAL_AIR_FILES}
    COMMENT "Linking Metal shaders → faiss_metal.metallib"
)
add_custom_target(metal_shaders DEPENDS ${CMAKE_BINARY_DIR}/faiss_metal.metallib)

# --- Library ---
set(FAISS_METAL_SOURCES
    src/MetalContext.mm
    src/MetalDeviceCapabilities.mm
    src/StandardMetalResources.mm
    src/MetalL2Norm.mm
    src/MetalDistance.mm
    src/MetalSelect.mm
    src/MetalIndexFlat.mm
)

add_library(faiss_metal STATIC ${FAISS_METAL_SOURCES})
add_dependencies(faiss_metal metal_shaders)

target_include_directories(faiss_metal
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(faiss_metal
    PUBLIC
        faiss
    PRIVATE
        ${METAL_FW}
        ${MPS_FW}
        ${FOUNDATION_FW}
        "-framework Accelerate"
)

# Pass metallib path as compile definition
target_compile_definitions(faiss_metal PRIVATE
    FAISS_METAL_METALLIB_PATH="${CMAKE_BINARY_DIR}/faiss_metal.metallib"
)

# --- Tests ---
option(FAISS_METAL_BUILD_TESTS "Build tests" ON)
if(FAISS_METAL_BUILD_TESTS)
    enable_testing()

    add_executable(test_metal_flat tests/test_metal_flat.mm)
    target_link_libraries(test_metal_flat faiss_metal faiss ${METAL_FW} ${MPS_FW} ${FOUNDATION_FW})
    add_test(NAME test_metal_flat COMMAND test_metal_flat)

    add_executable(test_metal_distance tests/test_metal_distance.mm)
    target_link_libraries(test_metal_distance faiss_metal faiss ${METAL_FW} ${MPS_FW} ${FOUNDATION_FW})
    add_test(NAME test_metal_distance COMMAND test_metal_distance)

    add_executable(bench_metal_flat tests/bench_metal_flat.mm)
    target_link_libraries(bench_metal_flat faiss_metal faiss ${METAL_FW} ${MPS_FW} ${FOUNDATION_FW})
endif()
