cmake_minimum_required(VERSION 3.18)
project(QuadrigaLib LANGUAGES C CXX)

# ---------------------------------------------------------------------------
# Version — single source of truth: include/quadriga_lib.hpp
# Parses: #define QUADRIGA_LIB_VERSION_STR "0.10.9"
# ---------------------------------------------------------------------------
file(STRINGS "${CMAKE_SOURCE_DIR}/include/quadriga_lib.hpp" _ver_line
     REGEX "#define QUADRIGA_LIB_VERSION_STR")
string(REGEX REPLACE ".*QUADRIGA_LIB_VERSION_STR \"([0-9]+)\\.([0-9]+)\\.([0-9]+)\".*"
       "\\1" QUADRIGA_VERSION_MAJOR "${_ver_line}")
string(REGEX REPLACE ".*QUADRIGA_LIB_VERSION_STR \"([0-9]+)\\.([0-9]+)\\.([0-9]+)\".*"
       "\\2" QUADRIGA_VERSION_MINOR "${_ver_line}")
string(REGEX REPLACE ".*QUADRIGA_LIB_VERSION_STR \"([0-9]+)\\.([0-9]+)\\.([0-9]+)\".*"
       "\\3" QUADRIGA_VERSION_PATCH "${_ver_line}")
set(QUADRIGA_VERSION
    "${QUADRIGA_VERSION_MAJOR}.${QUADRIGA_VERSION_MINOR}.${QUADRIGA_VERSION_PATCH}")
message(STATUS "quadriga-lib version: ${QUADRIGA_VERSION}")

# Option to enable MATLAB/Octave MEX compilation (default: ON)
# For octave, make sure that 'mkoctfile' is installed and can be called
option(ENABLE_MATLAB "Enable MATLAB MEX API" ON)
option(ENABLE_OCTAVE "Enable Octave MEX API" ON)
option(ENABLE_MEX_DOC "Enable MEX Documentation" ON)
option(ENABLE_PYTHON "Enable Python API" ON)
option(ENABLE_STATIC_LIB "Build the static library" ON)
option(ENABLE_SHARED_LIB "Build the shared library (Linux only)" OFF)
option(ENABLE_AVX2 "Enable AVX2 Acceleration" ON)
option(ENABLE_CUDA "Enable CUDA GPU Acceleration" OFF)
option(ENABLE_TESTS "Build tests" OFF)

set(HDF5_PATH "" CACHE PATH "Location of the HDF5 library (include files and .a/.lib file).
    If left empty, HDF5 location is detected automatically; if not found, HDF5 is built from sources.")
set(_HDF5_PATH ${HDF5_PATH}) # For internal use

set(CATCH2_PATH "" CACHE PATH "Location of the Catch2 library. If left empty, Catch2 is built from sources.")
set(_CATCH2_PATH ${CATCH2_PATH}) # For internal use

# If HDF5_STATIC is ON, HDF5 is build from sources and statically linked
# Static linking the HDF5 library may cause Octave to crash but usually works fine with MATLAB
option(HDF5_STATIC "Link HDF5 statically" OFF)

# Cmake tries to detect Armadillo include files on the system and link to these files.
# If Armadillo is not found, the version provided in the "external" folder is used instead.
option(ARMA_EXT "Use armadillo headers provided with quadriga-lib" OFF)

# Versions for external libraries (adjust as needed)
set(armadillo_version "14.2.2")
set(pugixml_version "1.15")
set(pybind11_version "3.0.0")
set(hdf5_version "1.14.6")
set(catch2_version "3.8.1")

# Set the C++ standard and basic compile flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(MEX_OUTPUT_DIR "${CMAKE_BINARY_DIR}/+quadriga_lib")

if(MSVC)
    add_compile_options(/EHsc /Zc:__cplusplus /nologo /MP)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    set(MEX_CXXFLAGS "/std:c++17 /nologo $<$<CONFIG:Debug>:/MTd> $<$<CONFIG:Release>:/MT>")
else()
    add_compile_options(-O3 -fPIC -Wall -Wextra $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>)
    set(MEX_CXXFLAGS "-std=c++17 -O3 -fPIC -fopenmp -Wall -Wpedantic -Wextra")
endif()

include(ExternalProject)
include(CheckCXXCompilerFlag)

# Check for OpenMP support
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found, multi-threading enabled")
else()
    message(STATUS "OpenMP not found, falling back to single-threaded")
endif()

# External library files
set(ARMADILLO_ZIP "${CMAKE_SOURCE_DIR}/external/armadillo-${armadillo_version}.zip")
set(PUGIXML_ZIP "${CMAKE_SOURCE_DIR}/external/pugixml-${pugixml_version}.zip")
set(PYBIND11_ZIP "${CMAKE_SOURCE_DIR}/external/pybind11-${pybind11_version}.zip")
set(HDF5_ZIP "${CMAKE_SOURCE_DIR}/external/hdf5-${hdf5_version}.zip")
set(CATCH2_ZIP "${CMAKE_SOURCE_DIR}/external/Catch2-${catch2_version}.zip")
set(MATLAB_MEX_ZIP "${CMAKE_SOURCE_DIR}/external/matlab_mex.zip")

# External packages
set(ARMADILLO_SRC_DIR "${CMAKE_BINARY_DIR}/armadillo-${armadillo_version}")
set(PUGIXML_SRC_DIR "${CMAKE_BINARY_DIR}/pugixml-${pugixml_version}")
set(PYBIND11_SRC_DIR "${CMAKE_BINARY_DIR}/pybind11-${pybind11_version}")
set(MATLAB_MEX_DIR "${CMAKE_BINARY_DIR}/matlab_mex")

# Unzip external packages
add_custom_command(
    OUTPUT ${ARMADILLO_SRC_DIR}/.stamp
    COMMAND ${CMAKE_COMMAND} -E tar xf ${ARMADILLO_ZIP}
    COMMAND ${CMAKE_COMMAND} -E touch ${ARMADILLO_SRC_DIR}/.stamp
    COMMENT "Unzipping Armadillo library to ${ARMADILLO_SRC_DIR}"
    VERBATIM
)
add_custom_command(
    OUTPUT ${PUGIXML_SRC_DIR}/.stamp
    COMMAND ${CMAKE_COMMAND} -E tar xf ${PUGIXML_ZIP}
    COMMAND ${CMAKE_COMMAND} -E touch ${PUGIXML_SRC_DIR}/.stamp
    COMMENT "Unzipping PugiXML library to ${PUGIXML_SRC_DIR}"
    VERBATIM
)
add_custom_command(
    OUTPUT ${PYBIND11_SRC_DIR}/.stamp
    COMMAND ${CMAKE_COMMAND} -E tar xf ${PYBIND11_ZIP}
    COMMAND ${CMAKE_COMMAND} -E touch ${PYBIND11_SRC_DIR}/.stamp
    COMMENT "Unzipping Pybind11 library to ${PYBIND11_SRC_DIR}"
    VERBATIM
)
add_custom_command(
    OUTPUT ${MATLAB_MEX_DIR}/.stamp
    COMMAND ${CMAKE_COMMAND} -E tar xf ${MATLAB_MEX_ZIP}
    COMMAND ${CMAKE_COMMAND} -E touch ${MATLAB_MEX_DIR}/.stamp
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Unzipping MATLAB MEX headers and libraries to ${MATLAB_MEX_DIR}"
    VERBATIM
)

# Create a custom target for the unzip step.
add_custom_target(pugixml_lib ALL DEPENDS ${PUGIXML_SRC_DIR}/.stamp)
add_custom_target(pybind11_lib DEPENDS ${PYBIND11_SRC_DIR}/.stamp)

set(PUGIXML_H   "${PUGIXML_SRC_DIR}/src")
set(PYBIND11_H  "${PYBIND11_SRC_DIR}/include")

# Options for armadillo
if (ARMA_EXT)
    add_custom_target(armadillo_lib ALL DEPENDS ${ARMADILLO_SRC_DIR}/.stamp)
    set(ARMADILLO_INCLUDE_DIRS "${ARMADILLO_SRC_DIR}/include")
    set(ARMADILLO_FOUND OFF)
else()
    find_package(Armadillo) # Sets ARMADILLO_INCLUDE_DIRS and ARMADILLO_FOUND
    if (NOT ARMADILLO_FOUND)
        add_custom_target(armadillo_lib ALL DEPENDS ${ARMADILLO_SRC_DIR}/.stamp)
        set(ARMADILLO_INCLUDE_DIRS "${ARMADILLO_SRC_DIR}/include")
    endif()
endif()

# External include directories
message(STATUS "Armadillo Include dir: ${ARMADILLO_INCLUDE_DIRS}")

# --- Components to build ---
set(BUILD_MATLAB ${ENABLE_MATLAB})
set(BUILD_OCTAVE ${ENABLE_OCTAVE})
set(BUILD_MEX_DOC ${ENABLE_MEX_DOC})
set(BUILD_PYTHON ${ENABLE_PYTHON})
set(BUILD_STATIC_LIB ${ENABLE_STATIC_LIB})
set(BUILD_SHARED_LIB ${ENABLE_SHARED_LIB})
set(BUILD_CUDA ${ENABLE_CUDA})
set(BUILD_HDF5 ${HDF5_STATIC})

if(MSVC) # Platform Overwrites for MSVC / Windows
    if(BUILD_OCTAVE)
        message(STATUS "Octave not supported on Windows. Octave API disabled!")
        set(BUILD_OCTAVE OFF)
    endif()
    if(BUILD_SHARED_LIB)
        message(STATUS "Shared library not supported on Windows.")
        set(BUILD_SHARED_LIB OFF)
    endif()
    if(BUILD_CUDA)
        message(STATUS "CUDA not supported on Windows.")
        set(BUILD_CUDA OFF)
    endif()
endif()

if (BUILD_PYTHON OR BUILD_MEX_DOC) # Detect python
    find_package(Python3 COMPONENTS Interpreter Development.Module)
    if (NOT Python3_FOUND)
        set(BUILD_PYTHON OFF)
    endif()
endif()

if (NOT BUILD_PYTHON) # Mex documentation requires Python
    set(BUILD_MEX_DOC OFF)
endif()

if(BUILD_MATLAB) # Detect MATLAB MEX headers
    if(EXISTS ${MATLAB_MEX_ZIP})
        add_custom_target(matlab_mex_unzip ALL DEPENDS ${MATLAB_MEX_DIR}/.stamp)
        set(MATLAB_MEX_INCLUDE_DIR "${MATLAB_MEX_DIR}/include")
        set(MATLAB_MEX_LIB_DIR "${MATLAB_MEX_DIR}/lib")
        set(MATLAB_MEX_VERSION_SCRIPT "${MATLAB_MEX_DIR}/mexFunction.map")
        message(STATUS "Using vendored MATLAB MEX headers from: ${MATLAB_MEX_ZIP}")
    else()
        message(STATUS "MATLAB MEX package not found (${MATLAB_MEX_ZIP}), MATLAB API disabled!")
        set(BUILD_MATLAB OFF)
    endif()
endif()

if(BUILD_MATLAB) # MATLAB requires static lib (HDF5 version clash due to MALTABs own libhdf5.so)
    set(BUILD_STATIC_LIB ON)
    message(STATUS "MATLAB enabled: forcing BUILD_STATIC_LIB=ON (HDF5 isolation)")
endif()

if (BUILD_OCTAVE) # Detect Octave
    find_program(MKOCTFILE_EXECUTABLE mkoctfile)
    if(MKOCTFILE_EXECUTABLE)
        message(STATUS "Found mkoctfile: ${MKOCTFILE_EXECUTABLE}")
    else()
        message(STATUS "Octave not found. Octave API disabled!")
        set(BUILD_OCTAVE OFF)
    endif()
endif()

if(NOT BUILD_STATIC_LIB AND NOT BUILD_SHARED_LIB) # Validation Rules
    message(FATAL_ERROR "At least one of STATIC_LIB or SHARED_LIB must be ON")
endif()

# Auto-detect pre-built HDF5 in external/hdf5-prebuilt
if (MSVC AND NOT _HDF5_PATH AND NOT BUILD_HDF5)
    set(_HDF5_PREBUILT_LIB "${CMAKE_SOURCE_DIR}/external/hdf5-prebuilt/lib/libhdf5.lib")
    if (EXISTS "${_HDF5_PREBUILT_LIB}")
        set(_HDF5_PATH "${CMAKE_SOURCE_DIR}/external/hdf5-prebuilt")
        message(STATUS "Auto-detected pre-built HDF5: ${_HDF5_PATH}")
    endif()
endif()

# Build instructions for HDF5 library
if (NOT _HDF5_PATH AND NOT BUILD_HDF5) # Detect HDF5 automatically
    find_package(HDF5 MODULE COMPONENTS C HL)
    if(HDF5_FOUND)
        get_filename_component(HDF5_LIB_DIR "${HDF5_LIBRARIES}" DIRECTORY)
        message(STATUS "HDF5 Include dir: ${HDF5_INCLUDE_DIRS}")
        message(STATUS "HDF5 Libraries: ${HDF5_LIBRARIES}")
    else()
        set(BUILD_HDF5 ON)
    endif()
else()
    set(HDF5_FOUND OFF)
endif()

# Octave does not support static linking HDF5 library (version clash, undefined behavior)
if (_HDF5_PATH OR BUILD_HDF5)
    if (BUILD_OCTAVE)
        message(STATUS "Octave does not support static linking HDF5 library. Octave disabled!")
        set(BUILD_OCTAVE OFF)
    endif()
endif()

if (_HDF5_PATH) # Use given HDF5 location before building it
    set(BUILD_HDF5 OFF)
    get_filename_component(HDF5_PATH_ABSOLUTE "${_HDF5_PATH}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
    set(HDF5_INCLUDE_DIRS "${_HDF5_PATH}/include")
    if (MSVC)
        set(HDF5_LIB "${HDF5_PATH_ABSOLUTE}/lib/libhdf5.lib")
    else()
        set(HDF5_LIB "${HDF5_PATH_ABSOLUTE}/lib/libhdf5.a")
    endif()
endif()

if (BUILD_HDF5) # Compile HDF5 library
    set(HDF5_EP_ARGS
        URL ${HDF5_ZIP}
        BINARY_DIR "${CMAKE_BINARY_DIR}/hdf5-${hdf5_version}_build"
        INSTALL_DIR "${CMAKE_BINARY_DIR}/hdf5-${hdf5_version}_bin"
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
            -DCMAKE_BUILD_TYPE=Release
            -DBUILD_SHARED_LIBS=OFF
            -DHDF5_ENABLE_Z_LIB_SUPPORT=OFF
            -DHDF5_ENABLE_SZIP_SUPPORT=OFF
            -DHDF5_BUILD_UTILS=OFF
            -DBUILD_TESTING=OFF
            -DHDF5_BUILD_TOOLS=OFF
            -DHDF5_BUILD_EXAMPLES=OFF
            -DHDF5_BUILD_HL_LIB=OFF
            -DHDF5_BUILD_CPP_LIB=OFF
            -DHDF5_BUILD_JAVA=OFF
            -DHDF5_BUILD_FORTRAN=OFF
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    )
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
        list(APPEND HDF5_EP_ARGS DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    endif()

    if (MSVC)
        list(APPEND HDF5_EP_ARGS CMAKE_ARGS -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded)
        set(HDF5_LIB "${CMAKE_BINARY_DIR}/hdf5-${hdf5_version}_bin/lib/libhdf5.lib")
    else()
        set(HDF5_LIB "${CMAKE_BINARY_DIR}/hdf5-${hdf5_version}_bin/lib/libhdf5.a")
    endif()
    set(HDF5_INCLUDE_DIRS "${CMAKE_BINARY_DIR}/hdf5-${hdf5_version}_bin/include")

    ExternalProject_Add(hdf5_dep ${HDF5_EP_ARGS} BUILD_BYPRODUCTS ${HDF5_LIB})
endif()

if(HDF5_FOUND)
    set(HDF5_LINK_LIBS ${HDF5_LIBRARIES})
elseif(HDF5_LIB)
    set(HDF5_LINK_LIBS ${HDF5_LIB})
endif()

# Build instructions for CATCH2 library
if (ENABLE_TESTS)
    if (MSVC AND NOT _CATCH2_PATH AND EXISTS "${CMAKE_SOURCE_DIR}/external/Catch2-prebuilt/lib/Catch2.lib")
        set(_CATCH2_PATH "${CMAKE_SOURCE_DIR}/external/Catch2-prebuilt")
        message(STATUS "Auto-detected pre-built Catch2: ${_CATCH2_PATH}")
    endif()
    if (_CATCH2_PATH)
        set(CATCH2_LIB "${_CATCH2_PATH}/lib/Catch2.lib")
        set(CATCH2_INCLUDE_DIR "${_CATCH2_PATH}/include")
        add_custom_target(catch2_dep)
    else()
        set(CATCH2_EP_ARGS
            URL ${CATCH2_ZIP}
            BINARY_DIR "${CMAKE_BINARY_DIR}/catch2-${catch2_version}_build"
            INSTALL_DIR "${CMAKE_BINARY_DIR}/catch2-${catch2_version}_bin"
            CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> -DCMAKE_BUILD_TYPE=Release
        )
        if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
            list(APPEND CATCH2_EP_ARGS DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
        endif()
        if (MSVC)
            list(APPEND CATCH2_EP_ARGS CMAKE_ARGS -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded)
            set(CATCH2_LIB "${CMAKE_BINARY_DIR}/catch2-${catch2_version}_bin/lib/Catch2.lib")
        else()
            set(CATCH2_LIB "${CMAKE_BINARY_DIR}/catch2-${catch2_version}_bin/lib/libCatch2.a")
        endif()
        set(CATCH2_INCLUDE_DIR "${CMAKE_BINARY_DIR}/catch2-${catch2_version}_bin/include")
        ExternalProject_Add(catch2_dep ${CATCH2_EP_ARGS} BUILD_BYPRODUCTS ${CATCH2_LIB})
    endif()
endif()

# Core includes needed by all library targets
set(QUADRIGA_CORE_INCLUDES
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${ARMADILLO_INCLUDE_DIRS}
    ${PUGIXML_H}
)

# Locate all core source files
file(GLOB SRC_FILES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/*.cpp")
file(GLOB SRC_AVX2 CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/*_avx2.cpp")
file(GLOB SRC_CUDA CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/*_cuda.cu")
list(REMOVE_ITEM SRC_FILES ${SRC_AVX2})

# For files that require AVX2/FMA, add the extra flags.
set(BUILD_WITH_AVX2 1)
if(ENABLE_AVX2)
    if (MSVC)
        check_cxx_compiler_flag("/arch:AVX2" COMPILER_SUPPORTS_AVX2)
    else()
        check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
    endif()
endif()
if(ENABLE_AVX2 AND COMPILER_SUPPORTS_AVX2)
    foreach(src ${SRC_AVX2})
        if (MSVC)
            set_source_files_properties(${src} PROPERTIES COMPILE_FLAGS "/arch:AVX2")
        else()
            set_source_files_properties(${src} PROPERTIES COMPILE_FLAGS "-mavx2 -mfma")
        endif()
    endforeach()
    message(STATUS "AVX2 acceleration enabled")
else()
    set(BUILD_WITH_AVX2 0)
    set(SRC_AVX2 "")
    message(STATUS "AVX2 acceleration disabled")
endif()

# For files that require CUDA, detect nvcc and add CUDA language support.
set(BUILD_WITH_CUDA 0)
if(BUILD_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        if(NOT MSVC)
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options -fPIC")
        endif()
        set(BUILD_WITH_CUDA 1)
        find_package(CUDAToolkit REQUIRED)
        set(CUDA_LINK_LIBS CUDA::cudart_static)
        set(CUDA_MEX_LINK_LIBS -L${CUDAToolkit_LIBRARY_DIR} -lcudart_static -ldl -lrt)
        message(STATUS "CUDA acceleration enabled (nvcc: ${CMAKE_CUDA_COMPILER})")
    else()
        message(STATUS "CUDA compiler not found, CUDA acceleration disabled")
        foreach(src ${SRC_CUDA})
            list(REMOVE_ITEM SRC_FILES ${src})
        endforeach()
        set(SRC_CUDA "")
    endif()
else()
    message(STATUS "CUDA acceleration disabled")
    foreach(src ${SRC_CUDA})
        list(REMOVE_ITEM SRC_FILES ${src})
    endforeach()
    set(SRC_CUDA "")
endif()

# --- Static library ---
if(BUILD_STATIC_LIB)
    message(STATUS "Building static library")
    add_library(quadriga_static STATIC ${SRC_FILES} ${SRC_AVX2} ${SRC_CUDA})
    target_include_directories(quadriga_static PRIVATE
        ${QUADRIGA_CORE_INCLUDES}
        ${HDF5_INCLUDE_DIRS}
    )
    set_target_properties(quadriga_static PROPERTIES OUTPUT_NAME quadriga)
    add_dependencies(quadriga_static pugixml_lib)
    target_compile_definitions(quadriga_static PRIVATE
        BUILD_WITH_AVX2=${BUILD_WITH_AVX2}
        BUILD_WITH_CUDA=${BUILD_WITH_CUDA}
        ARMA_DONT_USE_BLAS
        ARMA_DONT_USE_LAPACK
        _USE_MATH_DEFINES
    )
    if(OpenMP_CXX_FOUND)
        target_link_libraries(quadriga_static PUBLIC OpenMP::OpenMP_CXX)
    endif()
    if(BUILD_WITH_CUDA)
        target_link_libraries(quadriga_static PUBLIC ${CUDA_LINK_LIBS})
        set_target_properties(quadriga_static PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    endif()
    if (BUILD_HDF5)
        add_dependencies(quadriga_static hdf5_dep)
    endif()
    if (NOT ARMADILLO_FOUND)
        add_dependencies(quadriga_static armadillo_lib)
    endif()
endif()

# --- Shared library ---
if (BUILD_SHARED_LIB)
    message(STATUS "Building shared library")
    add_library(quadriga_shared SHARED ${SRC_FILES} ${SRC_AVX2} ${SRC_CUDA})
    target_include_directories(quadriga_shared PRIVATE
        ${QUADRIGA_CORE_INCLUDES}
        ${HDF5_INCLUDE_DIRS}
    )
    set_target_properties(quadriga_shared PROPERTIES
        OUTPUT_NAME quadriga
        VERSION ${QUADRIGA_VERSION}
        SOVERSION ${QUADRIGA_VERSION_MAJOR}
    )
    add_dependencies(quadriga_shared pugixml_lib)
    target_compile_definitions(quadriga_shared PRIVATE
        BUILD_WITH_AVX2=${BUILD_WITH_AVX2}
        BUILD_WITH_CUDA=${BUILD_WITH_CUDA}
        ARMA_DONT_USE_BLAS
        ARMA_DONT_USE_LAPACK
        _USE_MATH_DEFINES
    )
    if(OpenMP_CXX_FOUND)
        target_link_libraries(quadriga_shared PUBLIC OpenMP::OpenMP_CXX)
    endif()
    if(BUILD_WITH_CUDA)
        target_link_libraries(quadriga_shared PUBLIC CUDA::cudart)
        set_target_properties(quadriga_shared PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    endif()
    if (NOT ARMADILLO_FOUND)
        add_dependencies(quadriga_shared armadillo_lib)
    endif()
    # HDF5 handling: link directly into shared lib
    if (BUILD_HDF5)
        add_dependencies(quadriga_shared hdf5_dep)
        target_link_libraries(quadriga_shared PUBLIC ${HDF5_LIB})
    elseif(_HDF5_PATH)
        target_link_libraries(quadriga_shared PUBLIC ${HDF5_LIB})
    elseif(HDF5_FOUND)
        target_link_libraries(quadriga_shared PUBLIC HDF5::HDF5)
    endif()
endif()

# --- Link Target Selection ---
if(BUILD_SHARED_LIB)
    set(QUADRIGA_LINK_TARGET quadriga_shared)
else()
    set(QUADRIGA_LINK_TARGET quadriga_static)
endif()

# --- MATLAB API ---
if(BUILD_MATLAB)
    # Compiles MEX files directly with g++ (Linux) or cl.exe (Windows). No MATLAB installation required
    # Note: MATLAB always links quadriga_static regardless of QUADRIGA_LINK_TARGET to avoid conflicts with MATLAB's own bundled libhdf5.so
    file(MAKE_DIRECTORY ${MEX_OUTPUT_DIR})
    file(GLOB API_MEX_FILES "${CMAKE_SOURCE_DIR}/api_mex/*.cpp")
    set(MEX_MATLAB_TARGETS "")

    foreach(mex_src ${API_MEX_FILES})
        get_filename_component(mex_basename ${mex_src} NAME_WE)

        # Uniform dependency: all MEX files depend on the link target
        set(mex_deps ${mex_src})
        list(APPEND mex_deps quadriga_static)
        if(mex_basename STREQUAL "ieee_chan_indoor")
            list(APPEND mex_deps ${CMAKE_SOURCE_DIR}/src/ieee_channel_models.hpp)
        endif()

        if(MSVC)
            set(mex_output "${MEX_OUTPUT_DIR}/${mex_basename}.mexw64")
            add_custom_command(
                OUTPUT ${mex_output}
                COMMAND ${CMAKE_CXX_COMPILER}
                        /nologo /std:c++17 /EHs /O2 /MT /LD /DMATLAB_MEX_FILE
                        /I${MATLAB_MEX_INCLUDE_DIR} /I${CMAKE_SOURCE_DIR}/include /I${CMAKE_SOURCE_DIR}/src /I${ARMADILLO_INCLUDE_DIRS}
                        ${mex_src} /link $<TARGET_FILE:quadriga_static> ${HDF5_LINK_LIBS} ${CUDA_MEX_LINK_LIBS}
                        shlwapi.lib "${MATLAB_MEX_LIB_DIR}/libmex.lib" "${MATLAB_MEX_LIB_DIR}/libmx.lib"
                        /OUT:${mex_output} /DLL /EXPORT:mexFunction
                DEPENDS ${mex_deps}
                COMMENT "Building MATLAB MEX file ${mex_basename}.mexw64"
                VERBATIM
            )
        else()
            set(mex_output "${MEX_OUTPUT_DIR}/${mex_basename}.mexa64")
            add_custom_command(
                OUTPUT ${mex_output}
                COMMAND ${CMAKE_CXX_COMPILER}
                        -std=c++17 -O3 -shared -fPIC -DMATLAB_MEX_FILE
                        -I${MATLAB_MEX_INCLUDE_DIR} -I${CMAKE_SOURCE_DIR}/include -I${CMAKE_SOURCE_DIR}/src -I${ARMADILLO_INCLUDE_DIRS}
                        ${mex_src} $<TARGET_FILE:quadriga_static> ${HDF5_LINK_LIBS} ${CUDA_MEX_LINK_LIBS} -lgomp
                        -Wl,--version-script,${MATLAB_MEX_VERSION_SCRIPT} -Wl,--no-undefined-version -Wl,--unresolved-symbols=ignore-all
                        -o ${mex_output}
                DEPENDS ${mex_deps}
                COMMENT "Building MATLAB MEX file ${mex_basename}.mexa64"
                VERBATIM
            )
        endif()

        add_custom_target(mex_${mex_basename} ALL DEPENDS ${mex_output})
        list(APPEND mex_deps ${MATLAB_MEX_DIR}/.stamp)
        list(APPEND MEX_MATLAB_TARGETS mex_${mex_basename})
    endforeach()

    add_custom_target(mex_matlab ALL DEPENDS ${MEX_MATLAB_TARGETS})
endif()

# --- Octave API ---
if(BUILD_OCTAVE)
    if(QUADRIGA_LINK_TARGET STREQUAL "quadriga_shared") # Shared lib
        execute_process(COMMAND ${MKOCTFILE_EXECUTABLE} -p INCFLAGS OUTPUT_VARIABLE OCTAVE_INCFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
        execute_process(COMMAND ${MKOCTFILE_EXECUTABLE} -p OCTINCLUDEDIR OUTPUT_VARIABLE OCTAVE_OCTINCLUDEDIR OUTPUT_STRIP_TRAILING_WHITESPACE)
        execute_process(COMMAND ${MKOCTFILE_EXECUTABLE} -p OCTLIBDIR OUTPUT_VARIABLE OCTAVE_LIBDIR OUTPUT_STRIP_TRAILING_WHITESPACE)
    endif()

    file(MAKE_DIRECTORY ${MEX_OUTPUT_DIR})
    file(GLOB API_MEX_FILES "${CMAKE_SOURCE_DIR}/api_mex/*.cpp")
    set(MEX_OCTAVE_TARGETS "")

    foreach(mex_src ${API_MEX_FILES})
        get_filename_component(mex_basename ${mex_src} NAME_WE)
        set(octave_output "${MEX_OUTPUT_DIR}/${mex_basename}.mex")

        set(mex_deps ${mex_src})
        list(APPEND mex_deps ${QUADRIGA_LINK_TARGET})
        if(mex_basename STREQUAL "ieee_chan_indoor")
            list(APPEND mex_deps ${CMAKE_SOURCE_DIR}/src/ieee_channel_models.hpp)
        endif()

        if(QUADRIGA_LINK_TARGET STREQUAL "quadriga_shared") # Shared lib: use direct g++ (CUDA already resolved inside the .so)
            add_custom_command(
                OUTPUT ${octave_output}
                COMMAND ${CMAKE_CXX_COMPILER}
                        -std=c++17 -O3 -shared -fPIC -DMATLAB_MEX_FILE ${OCTAVE_INCFLAGS}
                        -I${OCTAVE_OCTINCLUDEDIR} -I${CMAKE_SOURCE_DIR}/include -I${CMAKE_SOURCE_DIR}/src -I${ARMADILLO_INCLUDE_DIRS}
                        ${mex_src} $<TARGET_FILE:${QUADRIGA_LINK_TARGET}>
                        -L${OCTAVE_LIBDIR} -loctinterp -loctave -lgomp
                        -Wl,-rpath,$ORIGIN -o ${octave_output}
                DEPENDS ${mex_deps}
                COMMENT "Building Octave MEX file ${octave_output}"
                VERBATIM
            )
        else() # Static lib: must use mkoctfile to handle CUDA correctly
            add_custom_command(
                OUTPUT ${octave_output}
                COMMAND env "CXXFLAGS=${MEX_CXXFLAGS} -U_FORTIFY_SOURCE" ${MKOCTFILE_EXECUTABLE} --mex -o ${octave_output}
                        ${mex_src} $<TARGET_FILE:quadriga_static> ${CUDA_MEX_LINK_LIBS}
                        -I${CMAKE_SOURCE_DIR}/include -I${CMAKE_SOURCE_DIR}/src -I${ARMADILLO_INCLUDE_DIRS} -s
                DEPENDS ${mex_deps}
                COMMENT "Building Octave MEX file ${octave_output}"
                VERBATIM
            )
        endif()

        list(APPEND MEX_OCTAVE_TARGETS ${octave_output})
    endforeach()
    add_custom_target(mex_octave ALL DEPENDS ${MEX_OCTAVE_TARGETS})
endif()

# --- MEX Documentation ---
if(BUILD_MEX_DOC AND (BUILD_MATLAB OR BUILD_OCTAVE))
    file(MAKE_DIRECTORY ${MEX_OUTPUT_DIR})
    file(GLOB API_MEX_FILES "${CMAKE_SOURCE_DIR}/api_mex/*.cpp")
    set(MEX_DOC_TARGETS "")

    foreach(mex_src ${API_MEX_FILES})
        get_filename_component(mex_basename ${mex_src} NAME_WE)
        set(m_file_output "${MEX_OUTPUT_DIR}/${mex_basename}.m")
        add_custom_command(
            OUTPUT ${m_file_output}
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/extract_matlab_comments.py ${mex_src} ${m_file_output}
            COMMENT "Building MEX Documentation ${m_file_output}"
            VERBATIM
        )
        list(APPEND MEX_DOC_TARGETS ${m_file_output})
    endforeach()
    add_custom_target(mex_documentation ALL DEPENDS ${MEX_DOC_TARGETS})
endif()

# --- Python API ---
if (BUILD_PYTHON)
    execute_process( # Get the Python extension suffix (like .cpython-311-x86_64-linux-gnu.so)
        COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
        OUTPUT_VARIABLE PYTHON_EXTENSION_SUFFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(PYTHON_SOURCES 
        api_python/python_main.cpp
        api_python/python_arrayant.cpp
        api_python/python_channel.cpp
        api_python/python_RTtools.cpp
        api_python/python_tools.cpp
    )
    add_library(quadriga_py MODULE ${PYTHON_SOURCES})
    target_include_directories(quadriga_py PRIVATE
        ${QUADRIGA_CORE_INCLUDES}
        ${PYBIND11_H}
        ${Python3_INCLUDE_DIRS}
    )
    add_dependencies(quadriga_py pybind11_lib pugixml_lib)
    if(BUILD_HDF5)
        add_dependencies(quadriga_py hdf5_dep)
    endif()
    if(QUADRIGA_LINK_TARGET STREQUAL "quadriga_shared")
        target_link_libraries(quadriga_py PRIVATE ${QUADRIGA_LINK_TARGET} ${Python3_LIBRARIES})
    else()
        target_link_libraries(quadriga_py PRIVATE ${QUADRIGA_LINK_TARGET} ${Python3_LIBRARIES} ${HDF5_LINK_LIBS} ${CUDA_LINK_LIBS})
    endif()
    set_target_properties(quadriga_py PROPERTIES PREFIX "" OUTPUT_NAME "quadriga_lib" SUFFIX "${PYTHON_EXTENSION_SUFFIX}" POSITION_INDEPENDENT_CODE ON)
    if(MSVC)
        target_link_libraries(quadriga_py PRIVATE shlwapi.lib)
    else()
        set_target_properties(quadriga_py PROPERTIES INSTALL_RPATH "$ORIGIN" BUILD_RPATH "$ORIGIN")
        target_compile_options(quadriga_py PRIVATE -Wno-unused-function)
    endif()
    file(GLOB PYTHON_INCLUDED_FILES "api_python/*.hpp" "api_python/*.cpp")
    set_source_files_properties(${PYTHON_SOURCES} PROPERTIES OBJECT_DEPENDS "${PYTHON_INCLUDED_FILES}")
    message(STATUS "Building Python module: quadriga_lib${PYTHON_EXTENSION_SUFFIX}")
endif()

# --- Catch2 Tests ---
if (ENABLE_TESTS)
    set(CATCH2_MAIN_SRC "${CMAKE_CURRENT_SOURCE_DIR}/tests/quadriga_lib_catch2_tests.cpp")

    file(GLOB TEST_SOURCES CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/catch2_tests/*.cpp
    )
    
    add_executable(test_bin ${CATCH2_MAIN_SRC} ${TEST_SOURCES})
    add_dependencies(test_bin catch2_dep)
    target_include_directories(test_bin PRIVATE
        ${QUADRIGA_CORE_INCLUDES}
        ${CATCH2_INCLUDE_DIR}
    )

    target_link_libraries(test_bin PRIVATE ${QUADRIGA_LINK_TARGET} ${CATCH2_LIB})
    if(QUADRIGA_LINK_TARGET STREQUAL "quadriga_static")
        target_link_libraries(test_bin PRIVATE ${HDF5_LINK_LIBS})
    endif()
    
    if(MSVC)
        target_link_libraries(test_bin PRIVATE shlwapi.lib)
        target_compile_options(test_bin PRIVATE /EHsc /Zc:__cplusplus /nologo /MP)
        target_compile_definitions(test_bin PRIVATE _USE_MATH_DEFINES ARMA_DONT_USE_BLAS ARMA_DONT_USE_LAPACK)
    else()
        target_compile_options(test_bin PRIVATE -O3 -fPIC -w)
    endif()
endif()

# --- Install options ---
if(CMAKE_INSTALL_PREFIX STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}")
    # Default: install to source directory (local / development layout)
    message(STATUS "Using source dir as install dir")

    set(INSTALL_LIBDIR lib)
    set(INSTALL_INCLUDEDIR include)
    set(INSTALL_MEX_DIR .)
    set(INSTALL_DOCDIR html_docu)
    set(INSTALL_LICENSEDIR .)
    set(INSTALL_ARMADIR .)
    set(INSTALL_IS_LOCAL TRUE)
elseif(NOT DEFINED SKBUILD_PLATLIB_DIR)
    # System install: use GNUInstallDirs for Debian/FHS-standard paths
    include(GNUInstallDirs)
    message(STATUS "Installing to: ${CMAKE_INSTALL_PREFIX}")

    set(INSTALL_LIBDIR ${CMAKE_INSTALL_LIBDIR})
    set(INSTALL_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR})
    set(INSTALL_MEX_DIR share/quadriga-lib)
    set(INSTALL_DOCDIR share/doc/quadriga-lib/html)
    set(INSTALL_LICENSEDIR share/doc/quadriga-lib)
    set(INSTALL_ARMADIR ${CMAKE_INSTALL_INCLUDEDIR})
    set(INSTALL_IS_LOCAL FALSE)

    install(DIRECTORY include/ DESTINATION ${INSTALL_INCLUDEDIR})
    if (NOT ARMADILLO_FOUND)
        install(DIRECTORY ${ARMADILLO_INCLUDE_DIRS} DESTINATION ${INSTALL_ARMADIR})
    endif()
    install(DIRECTORY html_docu/ DESTINATION ${INSTALL_DOCDIR})
    install(FILES LICENSE DESTINATION ${INSTALL_LICENSEDIR})
endif()

if(BUILD_STATIC_LIB AND NOT DEFINED SKBUILD_PLATLIB_DIR)
    install(TARGETS quadriga_static
            ARCHIVE DESTINATION ${INSTALL_LIBDIR}
            LIBRARY DESTINATION ${INSTALL_LIBDIR})
endif()

if (BUILD_SHARED_LIB AND NOT DEFINED SKBUILD_PLATLIB_DIR)
    install(TARGETS quadriga_shared
            ARCHIVE DESTINATION ${INSTALL_LIBDIR}
            LIBRARY DESTINATION ${INSTALL_LIBDIR})
    # For local installs, copy shared lib into MEX directory so Octave find it
    if(INSTALL_IS_LOCAL)
        install(FILES
            $<TARGET_FILE:quadriga_shared>
            $<TARGET_SONAME_FILE:quadriga_shared>
            DESTINATION ${INSTALL_MEX_DIR}/+quadriga_lib)
    endif()
endif()

if ((BUILD_HDF5 OR _HDF5_PATH) AND NOT DEFINED SKBUILD_PLATLIB_DIR)
    install(FILES ${HDF5_LIB} DESTINATION ${INSTALL_LIBDIR})
endif()

if((BUILD_MATLAB OR BUILD_OCTAVE) AND NOT DEFINED SKBUILD_PLATLIB_DIR)
    install(DIRECTORY ${MEX_OUTPUT_DIR} DESTINATION ${INSTALL_MEX_DIR})
endif()

if (BUILD_PYTHON)
    if(DEFINED SKBUILD_PLATLIB_DIR) # pip / scikit-build-core install
        message(STATUS "SKBUILD_PLATLIB_DIR = '${SKBUILD_PLATLIB_DIR}'")
        install(TARGETS quadriga_py LIBRARY DESTINATION ${SKBUILD_PLATLIB_DIR})
    elseif(INSTALL_IS_LOCAL)
        install(TARGETS quadriga_py
            ARCHIVE DESTINATION ${INSTALL_LIBDIR}
            LIBRARY DESTINATION ${INSTALL_LIBDIR})
    else()
        # Install to Python dist-packages for system installs
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('platlib', 'deb_system'))"
            OUTPUT_VARIABLE PYTHON3_DIST_PACKAGES
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _py_dist_result
        )
        if(_py_dist_result OR NOT PYTHON3_DIST_PACKAGES)
            # Fallback if deb_system scheme is not available
            execute_process(
                COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('platlib'))"
                OUTPUT_VARIABLE PYTHON3_DIST_PACKAGES
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
        endif()
        # Strip the leading prefix to get a relative path for install(TARGETS)
        string(REGEX REPLACE "^${CMAKE_INSTALL_PREFIX}/?" "" PYTHON3_DIST_REL "${PYTHON3_DIST_PACKAGES}")
        message(STATUS "Python install dir: ${PYTHON3_DIST_REL}")
        install(TARGETS quadriga_py
            LIBRARY DESTINATION ${PYTHON3_DIST_REL})
    endif()
endif()
