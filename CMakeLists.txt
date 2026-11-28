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
option(ENABLE_SANITIZERS "Build with AddressSanitizer and UndefinedBehaviorSanitizer for testing (GCC or Clang)" OFF)
option(ENABLE_FUZZING "Build the fuzz targets in tests/fuzzing with libFuzzer and the sanitizers (Clang only)" OFF)

set(CATCH2_PATH "" CACHE PATH "Location of the Catch2 library. If left empty, Catch2 is built from sources.")
set(_CATCH2_PATH ${CATCH2_PATH}) # For internal use

# Cmake tries to detect Armadillo include files on the system and link to these files.
# If Armadillo is not found, the version provided in the "external" folder is used instead.
option(ARMA_EXT "Use armadillo headers provided with quadriga-lib" OFF)

# Versions for external libraries (adjust as needed)
set(armadillo_version "15.6.2")
set(pugixml_version "1.15")
set(pybind11_version "3.1.0")
set(catch2_version "3.16.1")

# Set the C++ standard and basic compile flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(MEX_OUTPUT_DIR "${CMAKE_BINARY_DIR}/+quadriga_lib")

if(MSVC)
    add_compile_options(/EHsc /Zc:__cplusplus /nologo /MP)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    set(MEX_CXXFLAGS "/std:c++17 /nologo $<$<CONFIG:Debug>:/MTd> $<$<CONFIG:Release>:/MT>")
    if(ENABLE_SANITIZERS OR ENABLE_FUZZING)
        message(STATUS "Sanitizers and fuzzing need GCC or Clang. ENABLE_SANITIZERS and ENABLE_FUZZING ignored!")
        set(ENABLE_FUZZING OFF)
    endif()
else()
    set(OPT_FLAGS -O3)
    set(OPT_FLAGS_API -O2) # MEX, Python and test code: the library keeps -O3
    if(ENABLE_FUZZING) # The whole library gets the coverage instrumentation. UBSan stays recoverable, so that the fuzz
        # runner can stop at undefined behaviour (halt_on_error=1) and skip known cases with a suppressions file
        if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            message(FATAL_ERROR "ENABLE_FUZZING needs Clang (libFuzzer), e.g. -D CMAKE_CXX_COMPILER=clang++")
        endif()
        set(OPT_FLAGS -O1 -g -fno-omit-frame-pointer -fsanitize=fuzzer-no-link,address,undefined)
        set(OPT_FLAGS_API "")
        add_link_options(-fsanitize=address,undefined)
    elseif(ENABLE_SANITIZERS) # -O1 -g instead of -O3 keeps the stack traces in the reports readable
        set(OPT_FLAGS -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fsanitize-recover=address)
        set(OPT_FLAGS_API "")
        add_link_options(-fsanitize=address,undefined)
    endif()
    # One section per function and data object, so that the MATLAB MEX links (-Wl,--gc-sections) drop the library
    # code a MEX file does not use. MATLAB disassembles every MEX file when it first loads it, which takes time in
    # proportion to the code size.
    add_compile_options(${OPT_FLAGS} -fPIC -ffunction-sections -fdata-sections -Wall -Wextra -Wfloat-conversion $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>)
    set(MEX_CXXFLAGS "-std=c++17 -O3 -fPIC -fopenmp -Wall -Wpedantic -Wextra -Wfloat-conversion")
endif()

include(ExternalProject)

# The mold linker, when installed, links the MEX files, the Python module and test_bin in a fraction of the time
find_program(MOLD_LINKER mold)
if(MOLD_LINKER AND NOT MSVC)
    add_link_options(-fuse-ld=mold)
    message(STATUS "Using mold linker: ${MOLD_LINKER}")
endif()

# Compiler cache: ccache, when installed, stores the object files, so that unchanged sources compile fast in a new
# checkout or build folder. Paths below the source folder are hashed relative to it, so that checkouts at different
# paths share the cache; the sloppiness setting lets it cache files that use a precompiled header.
option(ENABLE_CCACHE "Use ccache when installed" ON)
if(ENABLE_CCACHE AND NOT CMAKE_CXX_COMPILER_LAUNCHER)
    find_program(CCACHE_PROGRAM ccache)
    if(CCACHE_PROGRAM)
        set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM} base_dir=${CMAKE_SOURCE_DIR} sloppiness=pch_defines,time_macros)
        set(CMAKE_C_COMPILER_LAUNCHER ${CMAKE_CXX_COMPILER_LAUNCHER})
        message(STATUS "Using ccache: ${CCACHE_PROGRAM}")
    endif()
endif()
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
    BYPRODUCTS ${MATLAB_MEX_DIR}/lib/libmex.lib ${MATLAB_MEX_DIR}/lib/libmx.lib # Link inputs on Windows
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
    if (ARMADILLO_FOUND)
        # find_package only searches for the files. A compiler with its own system folders (e.g. from conda) does
        # not see /usr/include, which CMake leaves out of the compile commands, so check that the header compiles.
        include(CheckIncludeFileCXX)
        set(CMAKE_REQUIRED_INCLUDES ${ARMADILLO_INCLUDE_DIRS})
        check_include_file_cxx(armadillo QD_ARMADILLO_USABLE)
        unset(CMAKE_REQUIRED_INCLUDES)
        if (NOT QD_ARMADILLO_USABLE)
            message(STATUS "Armadillo in ${ARMADILLO_INCLUDE_DIRS} is not usable with ${CMAKE_CXX_COMPILER}: using the bundled headers")
            set(ARMADILLO_FOUND OFF)
        endif()
    endif()
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

# The MEX files are compiled with their own fixed flags, and the CUDA runtime does not run under AddressSanitizer
if(ENABLE_SANITIZERS AND NOT MSVC AND (BUILD_MATLAB OR BUILD_OCTAVE OR BUILD_CUDA))
    message(STATUS "Sanitizers enabled: MATLAB, Octave and CUDA disabled!")
    set(BUILD_MATLAB OFF)
    set(BUILD_OCTAVE OFF)
    set(BUILD_CUDA OFF)
endif()

# The fuzz build is for the fuzz targets only; the Python module would need the sanitizer runtime preloaded
if(ENABLE_FUZZING AND NOT MSVC AND (BUILD_MATLAB OR BUILD_OCTAVE OR BUILD_CUDA OR BUILD_PYTHON OR BUILD_MEX_DOC))
    message(STATUS "Fuzzing enabled: MATLAB, Octave, Python and CUDA disabled!")
    set(BUILD_MATLAB OFF)
    set(BUILD_OCTAVE OFF)
    set(BUILD_MEX_DOC OFF)
    set(BUILD_PYTHON OFF)
    set(BUILD_CUDA OFF)
endif()

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

# MATLAB MEX files link the static quadriga library, so they do not depend on a libquadriga.so at run time.
if(BUILD_MATLAB)
    set(BUILD_STATIC_LIB ON)
    message(STATUS "MATLAB enabled: forcing BUILD_STATIC_LIB=ON (MEX files link libquadriga.a)")
endif()

if (BUILD_OCTAVE) # Detect Octave
    find_program(MKOCTFILE_EXECUTABLE mkoctfile)
    if(MKOCTFILE_EXECUTABLE)
        # The MEX files use the interleaved complex API, which works from Octave 8.4 on
        execute_process(COMMAND ${MKOCTFILE_EXECUTABLE} --version OUTPUT_VARIABLE OCTAVE_VERSION_TEXT ERROR_VARIABLE OCTAVE_VERSION_TEXT)
        string(REGEX MATCH "version ([0-9]+\\.[0-9]+\\.[0-9]+)" OCTAVE_VERSION_MATCH "${OCTAVE_VERSION_TEXT}")
        set(OCTAVE_VERSION_FOUND "${CMAKE_MATCH_1}")
        if(NOT OCTAVE_VERSION_FOUND)
            message(STATUS "Octave version not readable from 'mkoctfile --version'. Octave API disabled!")
            set(BUILD_OCTAVE OFF)
        elseif(OCTAVE_VERSION_FOUND VERSION_LESS 8.4)
            message(STATUS "Octave ${OCTAVE_VERSION_FOUND} found, 8.4 or later needed. Octave API disabled!")
            set(BUILD_OCTAVE OFF)
            file(GLOB OCTAVE_STALE_MEX "${CMAKE_INSTALL_PREFIX}/+quadriga_lib/*.mex")
            if(OCTAVE_STALE_MEX)
                message(WARNING "Octave MEX files of an earlier build are in ${CMAKE_INSTALL_PREFIX}/+quadriga_lib and are not updated; delete them.")
            endif()
        else()
            message(STATUS "Found mkoctfile: ${MKOCTFILE_EXECUTABLE} (Octave ${OCTAVE_VERSION_FOUND})")
        endif()
    else()
        message(STATUS "Octave not found. Octave API disabled!")
        set(BUILD_OCTAVE OFF)
    endif()
endif()

if(NOT BUILD_STATIC_LIB AND NOT BUILD_SHARED_LIB) # Validation Rules
    message(FATAL_ERROR "At least one of STATIC_LIB or SHARED_LIB must be ON")
endif()

# Build instructions for CATCH2 library
if (ENABLE_TESTS)
    if (_CATCH2_PATH)
        set(CATCH2_LIB "${_CATCH2_PATH}/lib/Catch2.lib")
        set(CATCH2_INCLUDE_DIR "${_CATCH2_PATH}/include")
        add_custom_target(catch2_dep)
    else()
        set(CATCH2_EP_ARGS
            URL ${CATCH2_ZIP}
            BINARY_DIR "${CMAKE_BINARY_DIR}/catch2-${catch2_version}_build"
            INSTALL_DIR "${CMAKE_BINARY_DIR}/catch2-${catch2_version}_bin"
            CMAKE_ARGS
                -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
                -DCMAKE_BUILD_TYPE=Release
                -DCMAKE_CXX_STANDARD=17
                -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                # Catch2 must use the same standard library and target as test_bin (e.g. -stdlib=libc++ on Linux,
                # architecture and minimum version on macOS)
                -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
                -DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}
                -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}
                -DCATCH_INSTALL_DOCS=OFF
                -DCATCH_INSTALL_EXTRAS=OFF
        )
        if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
            list(APPEND CATCH2_EP_ARGS DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
        endif()
        if(CMAKE_CXX_COMPILER_LAUNCHER) # The compiler cache, with "|" for the list separator
            string(REPLACE ";" "|" CATCH2_LAUNCHER "${CMAKE_CXX_COMPILER_LAUNCHER}")
            list(APPEND CATCH2_EP_ARGS LIST_SEPARATOR | CMAKE_ARGS -DCMAKE_CXX_COMPILER_LAUNCHER=${CATCH2_LAUNCHER})
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
        target_link_libraries(quadriga_static PRIVATE ${CUDA_LINK_LIBS})
        set_target_properties(quadriga_static PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
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
    )
    set_target_properties(quadriga_shared PROPERTIES
        OUTPUT_NAME quadriga
        VERSION ${QUADRIGA_VERSION}
        SOVERSION ${QUADRIGA_VERSION_MAJOR}.${QUADRIGA_VERSION_MINOR}
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
endif()

# Ninja starts the jobs with the longest chain of remaining steps first. An extra step after a target with long
# units (Python, tests) lets them start with the core library instead of after all MEX files.
function(qd_start_early tgt)
    add_custom_command(OUTPUT ${tgt}.stamp COMMAND ${CMAKE_COMMAND} -E touch ${tgt}.stamp DEPENDS ${tgt})
    add_custom_target(${tgt}_early ALL DEPENDS ${tgt}.stamp)
endfunction()

# --- Link Target Selection ---
if(BUILD_SHARED_LIB)
    set(QUADRIGA_LINK_TARGET quadriga_shared)
else()
    set(QUADRIGA_LINK_TARGET quadriga_static)
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
        api_python/python_sos.cpp
        api_python/python_tools.cpp
        api_python/python_gbsm.cpp
        api_python/python_layout.cpp
    )
    # The units compile in an object library, which does not wait for the core library
    add_library(quadriga_py_obj OBJECT ${PYTHON_SOURCES})
    target_include_directories(quadriga_py_obj PRIVATE
        ${QUADRIGA_CORE_INCLUDES}
        ${PYBIND11_H}
        ${Python3_INCLUDE_DIRS}
    )
    add_dependencies(quadriga_py_obj pybind11_lib pugixml_lib)
    if (NOT ARMADILLO_FOUND)
        add_dependencies(quadriga_py_obj armadillo_lib)
    endif()
    if(OpenMP_CXX_FOUND)
        target_link_libraries(quadriga_py_obj PRIVATE OpenMP::OpenMP_CXX)
    endif()
    target_compile_options(quadriga_py_obj PRIVATE ${OPT_FLAGS_API})
    set_target_properties(quadriga_py_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_library(quadriga_py MODULE $<TARGET_OBJECTS:quadriga_py_obj>)
    if(QUADRIGA_LINK_TARGET STREQUAL "quadriga_shared")
        target_link_libraries(quadriga_py PRIVATE ${QUADRIGA_LINK_TARGET} ${Python3_LIBRARIES})
    else()
        target_link_libraries(quadriga_py PRIVATE ${QUADRIGA_LINK_TARGET} ${Python3_LIBRARIES} ${CUDA_LINK_LIBS})
    endif()
    set_target_properties(quadriga_py PROPERTIES PREFIX "" OUTPUT_NAME "quadriga_lib" SUFFIX "${PYTHON_EXTENSION_SUFFIX}" POSITION_INDEPENDENT_CODE ON)
    if(MSVC)
        target_link_libraries(quadriga_py PRIVATE shlwapi.lib)
    else()
        set_target_properties(quadriga_py PROPERTIES INSTALL_RPATH "$ORIGIN" BUILD_RPATH "$ORIGIN")
    endif()
    qd_start_early(quadriga_py)
    message(STATUS "Building Python module: quadriga_lib${PYTHON_EXTENSION_SUFFIX}")
endif()

# --- MATLAB and Octave API ---
# Every MEX file is a target of its own: it is compiled once, a library change only relinks it, the compile tracks the
# included headers, and a compiler cache can store the object. One precompiled header per API holds the headers that
# all MEX files parse.
file(GLOB API_MEX_FILES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/api_mex/*.cpp")
set(MEX_PCH_HEADERS <mex.h> <armadillo> <quadriga_lib.hpp>)
if(BUILD_MATLAB OR BUILD_OCTAVE)
    file(MAKE_DIRECTORY ${MEX_OUTPUT_DIR})
endif()

# Makes the object library ${tgt} with all MEX files and one precompiled header. Object libraries do not wait for the
# core library, so they compile while it does. qd_mex_objects(<var> <tgt> <name>) gives the objects of one MEX file and,
# on MSVC, the object of the header.
function(qd_add_mex_objects tgt)
    add_library(${tgt} OBJECT ${API_MEX_FILES})
    target_include_directories(${tgt} PRIVATE ${MEX_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src
                               ${ARMADILLO_INCLUDE_DIRS})
    set_target_properties(${tgt} PROPERTIES COMPILE_OPTIONS "${MEX_COMPILE_OPTIONS}" CXX_EXTENSIONS OFF)
    target_compile_definitions(${tgt} PRIVATE ${MEX_DEFINITIONS})
    target_precompile_headers(${tgt} PRIVATE ${MEX_PCH_HEADERS})
    if(MEX_OBJECT_DEPENDENCIES)
        add_dependencies(${tgt} ${MEX_OBJECT_DEPENDENCIES})
    endif()
endfunction()
function(qd_mex_objects var tgt name)
    set(${var} "$<FILTER:$<TARGET_OBJECTS:${tgt}>,INCLUDE,/(${name}\\.cpp|cmake_pch\\.cxx)\\.o(bj)?$>" PARENT_SCOPE)
endfunction()

if (NOT ARMADILLO_FOUND)
    set(MEX_OBJECT_DEPENDENCIES armadillo_lib)
endif()

if(BUILD_MATLAB)
    # Compiled without a MATLAB installation, with the vendored headers. MATLAB always links quadriga_static
    # regardless of QUADRIGA_LINK_TARGET, so the MEX files do not need a libquadriga.so at run time. The library is
    # linked as a file, so that its OpenMP compile flag does not reach the MEX files.
    set(MEX_INCLUDE_DIRS ${MATLAB_MEX_INCLUDE_DIR})
    set(MEX_DEFINITIONS MATLAB_MEX_FILE MATLAB_DEFAULT_RELEASE=R2018a) # Interleaved complex API
    list(APPEND MEX_OBJECT_DEPENDENCIES matlab_mex_unzip)
    if(MSVC)
        # /EHs without "c": MATLAB's error functions throw through extern "C" code. CMake's default /EHsc in
        # CMAKE_CXX_FLAGS would add the "c"; the other targets get /EHsc from add_compile_options above, and Catch2
        # has taken its flags before this point.
        string(REPLACE "/EHsc" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        set(MEX_COMPILE_OPTIONS /nologo /EHs /O2 /MP)
    else()
        set(MEX_COMPILE_OPTIONS ${OPT_FLAGS_API} -fPIC -ffunction-sections -fdata-sections)
        add_library(mex_compat_shim OBJECT ${CMAKE_SOURCE_DIR}/src/mex_compat_shim.cpp)
        set_target_properties(mex_compat_shim PROPERTIES COMPILE_OPTIONS "${MEX_COMPILE_OPTIONS}" CXX_EXTENSIONS OFF)
    endif()
    qd_add_mex_objects(mexm)

    set(MEX_MATLAB_TARGETS "")
    foreach(mex_src ${API_MEX_FILES})
        get_filename_component(mex_basename ${mex_src} NAME_WE)
        set(tgt mex_${mex_basename})
        qd_mex_objects(objects mexm ${mex_basename})
        add_library(${tgt} MODULE "${objects}")
        add_dependencies(${tgt} quadriga_static)
        set_target_properties(${tgt} PROPERTIES PREFIX "" OUTPUT_NAME ${mex_basename} LINKER_LANGUAGE CXX
                              LIBRARY_OUTPUT_DIRECTORY ${MEX_OUTPUT_DIR})
        if(MSVC)
            set_target_properties(${tgt} PROPERTIES SUFFIX ".mexw64")
            target_link_libraries(${tgt} PRIVATE $<TARGET_FILE:quadriga_static> ${CUDA_MEX_LINK_LIBS} shlwapi.lib
                                  "${MATLAB_MEX_LIB_DIR}/libmex.lib" "${MATLAB_MEX_LIB_DIR}/libmx.lib")
            target_link_options(${tgt} PRIVATE /EXPORT:mexFunction)
        else()
            set_target_properties(${tgt} PROPERTIES SUFFIX ".mexa64")
            target_sources(${tgt} PRIVATE $<TARGET_OBJECTS:mex_compat_shim>)
            target_link_libraries(${tgt} PRIVATE $<TARGET_FILE:quadriga_static> ${CUDA_MEX_LINK_LIBS} gomp)
            target_link_options(${tgt} PRIVATE -Wl,--gc-sections -Wl,--version-script,${MATLAB_MEX_VERSION_SCRIPT}
                                -Wl,--no-undefined-version -Wl,--unresolved-symbols=ignore-all)
        endif()
        list(APPEND MEX_MATLAB_TARGETS ${tgt})
    endforeach()
    add_custom_target(mex_matlab ALL DEPENDS ${MEX_MATLAB_TARGETS})
endif()

if(BUILD_OCTAVE)
    # Compiled with the flags of mkoctfile; mkoctfile links them, as it adds a version stub and the Octave libraries
    execute_process(COMMAND ${MKOCTFILE_EXECUTABLE} -p INCFLAGS OUTPUT_VARIABLE OCTAVE_INCFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
    separate_arguments(OCTAVE_INCFLAGS UNIX_COMMAND "${OCTAVE_INCFLAGS}")
    list(TRANSFORM OCTAVE_INCFLAGS REPLACE "^-I" "")
    set(MEX_INCLUDE_DIRS ${OCTAVE_INCFLAGS})
    set(MEX_DEFINITIONS MEX_DEBUG MX_HAS_INTERLEAVED_COMPLEX=1)
    set(MEX_COMPILE_OPTIONS ${OPT_FLAGS_API} -fPIC -pthread -fopenmp -U_FORTIFY_SOURCE -Wall -Wpedantic -Wextra -Wfloat-conversion)
    qd_add_mex_objects(mexo)

    if(QUADRIGA_LINK_TARGET STREQUAL "quadriga_shared")
        set(QD_MEX_EXTRA_LIBS "")
        set(QD_MEX_EXTRA_LDFLAGS "-Wl,-rpath,\\$ORIGIN")  # find libquadriga.so next to the .mex
    else()
        set(QD_MEX_EXTRA_LIBS ${CUDA_MEX_LINK_LIBS})       # static: pull CUDA in explicitly
        set(QD_MEX_EXTRA_LDFLAGS "")
    endif()

    set(MEX_OCTAVE_TARGETS "")
    foreach(mex_src ${API_MEX_FILES})
        get_filename_component(mex_basename ${mex_src} NAME_WE)
        set(octave_output "${MEX_OUTPUT_DIR}/${mex_basename}.mex")
        qd_mex_objects(octave_objects mexo ${mex_basename})
        add_custom_command(
            OUTPUT ${octave_output}
            # mkoctfile links with the compiler of this build, not the one recorded when Octave was built (conda); with
            # -R2018a its version stub marks the file as using interleaved complex data
            COMMAND env "CXXFLAGS=${MEX_CXXFLAGS}" "CC=${CMAKE_C_COMPILER}" "CXX=${CMAKE_CXX_COMPILER}"
                    ${MKOCTFILE_EXECUTABLE} --mex -R2018a -o ${octave_output} "${octave_objects}"
                    -L$<TARGET_FILE_DIR:${QUADRIGA_LINK_TARGET}> -lquadriga
                    ${QD_MEX_EXTRA_LIBS} ${QD_MEX_EXTRA_LDFLAGS} -s
            DEPENDS mexo ${octave_objects} ${QUADRIGA_LINK_TARGET}
            COMMENT "Linking Octave MEX file ${mex_basename}.mex"
            VERBATIM COMMAND_EXPAND_LISTS
        )
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
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/docs_builder/extract_matlab_comments.py ${mex_src} ${m_file_output}
            DEPENDS ${mex_src} ${CMAKE_SOURCE_DIR}/docs_builder/extract_matlab_comments.py
            COMMENT "Building MEX Documentation ${m_file_output}"
            VERBATIM
        )
        list(APPEND MEX_DOC_TARGETS ${m_file_output})
    endforeach()
    add_custom_target(mex_documentation ALL DEPENDS ${MEX_DOC_TARGETS})
endif()

# --- Catch2 Tests ---
if (ENABLE_TESTS)
    set(CATCH2_MAIN_SRC "${CMAKE_CURRENT_SOURCE_DIR}/tests/quadriga_lib_catch2_tests.cpp")

    file(GLOB TEST_SOURCES CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/catch2_tests/*.cpp
    )

    # Optional interop test: files written by the library are read back with the HDF5 C library (1.12 or newer, the
    # test uses H5Ovisit3). It is compiled only if CMake finds such an HDF5 installation, which is then linked into
    # test_bin only (the library itself does not use the HDF5 C library).
    set(HDF5_INTEROP_TEST "${CMAKE_CURRENT_SOURCE_DIR}/tests/catch2_tests/test_hdf5_interop.cpp")
    list(REMOVE_ITEM TEST_SOURCES ${HDF5_INTEROP_TEST})
    find_package(HDF5 1.12 MODULE QUIET COMPONENTS C)
    if(HDF5_FOUND)
        list(APPEND TEST_SOURCES ${HDF5_INTEROP_TEST})
        set(HDF5_INTEROP_LIBS ${HDF5_LIBRARIES})
        set(HDF5_INTEROP_INCLUDE_DIRS ${HDF5_INCLUDE_DIRS})
        message(STATUS "HDF5 interop test enabled (HDF5 ${HDF5_VERSION})")
    endif()

    # The test units compile in an object library, which does not wait for the core library
    add_library(test_obj OBJECT ${CATCH2_MAIN_SRC} ${TEST_SOURCES})
    add_dependencies(test_obj catch2_dep)
    if (NOT ARMADILLO_FOUND)
        add_dependencies(test_obj armadillo_lib)
    endif()
    target_include_directories(test_obj PRIVATE
        ${QUADRIGA_CORE_INCLUDES}
        ${CATCH2_INCLUDE_DIR}
        ${HDF5_INTEROP_INCLUDE_DIRS}
    )
    if(OpenMP_CXX_FOUND)
        target_link_libraries(test_obj PRIVATE OpenMP::OpenMP_CXX)
    endif()

    add_executable(test_bin "$<FILTER:$<TARGET_OBJECTS:test_obj>,EXCLUDE,\\.gch$>")
    target_link_libraries(test_bin PRIVATE ${QUADRIGA_LINK_TARGET} ${CATCH2_LIB} ${HDF5_INTEROP_LIBS})
    if(HDF5_FOUND)
        target_compile_options(test_obj PRIVATE ${HDF5_DEFINITIONS}) # E.g. the dllimport switch of a shared HDF5
    endif()

    # The kernel tests compare AVX2 with GENERIC under #ifdef BUILD_WITH_AVX2, so the macro is defined only when
    # the library has the AVX2 kernels (the library itself gets BUILD_WITH_AVX2=0 or 1)
    if(BUILD_WITH_AVX2)
        target_compile_definitions(test_obj PRIVATE BUILD_WITH_AVX2=1)
    endif()

    # As in the library: nothing links BLAS or LAPACK, so Armadillo must not emit calls to them
    target_compile_definitions(test_obj PRIVATE ARMA_DONT_USE_BLAS ARMA_DONT_USE_LAPACK)

    if(MSVC)
        target_link_libraries(test_bin PRIVATE shlwapi.lib)
        target_compile_options(test_obj PRIVATE /EHsc /Zc:__cplusplus /nologo /MP)
        target_compile_definitions(test_obj PRIVATE _USE_MATH_DEFINES)
    else()
        target_compile_options(test_obj PRIVATE -w ${OPT_FLAGS_API})
    endif()

    qd_start_early(test_bin)

    # Catch2 and the library headers are parsed once and reused by every test file
    target_precompile_headers(test_obj PRIVATE <catch2/catch_test_macros.hpp> <quadriga_lib.hpp>)
endif()

# --- Fuzz targets: tests/fuzzing/README.md ---
if (ENABLE_FUZZING AND NOT MSVC)
    file(GLOB FUZZ_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tests/fuzzing/targets/fz_*.cpp)
    set(FUZZ_TARGET_NAMES "")
    foreach(FUZZ_SRC ${FUZZ_SOURCES})
        get_filename_component(FUZZ_NAME ${FUZZ_SRC} NAME_WE)
        add_executable(${FUZZ_NAME} ${FUZZ_SRC})
        target_include_directories(${FUZZ_NAME} PRIVATE ${QUADRIGA_CORE_INCLUDES} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fuzzing)
        target_compile_definitions(${FUZZ_NAME} PRIVATE QD_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
        target_link_libraries(${FUZZ_NAME} PRIVATE ${QUADRIGA_LINK_TARGET})
        target_link_options(${FUZZ_NAME} PRIVATE -fsanitize=fuzzer)
        list(APPEND FUZZ_TARGET_NAMES ${FUZZ_NAME})
    endforeach()
    add_custom_target(fuzz_targets DEPENDS ${FUZZ_TARGET_NAMES})
    message(STATUS "Fuzz targets: ${FUZZ_TARGET_NAMES}")
endif()

# --- Install options ---
if(CMAKE_INSTALL_PREFIX STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}")
    # Default: install to source directory (local / development layout)
    message(STATUS "Using source dir as install dir")

    set(INSTALL_LIBDIR lib)
    set(INSTALL_INCLUDEDIR include)
    set(INSTALL_MEX_DIR .)
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
    set(INSTALL_LICENSEDIR share/doc/quadriga-lib)
    set(INSTALL_ARMADIR ${CMAKE_INSTALL_INCLUDEDIR})
    set(INSTALL_IS_LOCAL FALSE)

    install(DIRECTORY include/ DESTINATION ${INSTALL_INCLUDEDIR})
    if (NOT ARMADILLO_FOUND)
        install(DIRECTORY ${ARMADILLO_INCLUDE_DIRS} DESTINATION ${INSTALL_ARMADIR})
    endif()
    install(DIRECTORY docs/ DESTINATION share/doc/quadriga-lib)
    install(FILES LICENSE DESTINATION ${INSTALL_LICENSEDIR})
    install(DIRECTORY data/config/ DESTINATION share/quadriga-lib/config FILES_MATCHING PATTERN "*.conf")
endif()

# Folder of the shipped conf files, compiled into conf_file_list and conf_file_read: data/config of the source tree for
# the local layout, share/quadriga-lib/config of the prefix for a system install; none for pip, whose Python module
# finds the folder at run time, as the MEX files do
if(DEFINED SKBUILD_PLATLIB_DIR)
    set(QD_CONF_DIR "")
elseif(INSTALL_IS_LOCAL)
    set(QD_CONF_DIR "${CMAKE_SOURCE_DIR}/data/config")
else()
    set(QD_CONF_DIR "${CMAKE_INSTALL_PREFIX}/share/quadriga-lib/config")
endif()
set_source_files_properties(${CMAKE_SOURCE_DIR}/src/qd_scenpar.cpp PROPERTIES
                            COMPILE_DEFINITIONS "QUADRIGA_LIB_CONF_DIR=\"${QD_CONF_DIR}\"")

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

if((BUILD_MATLAB OR BUILD_OCTAVE) AND NOT DEFINED SKBUILD_PLATLIB_DIR)
    install(DIRECTORY ${MEX_OUTPUT_DIR} DESTINATION ${INSTALL_MEX_DIR})
endif()

if (BUILD_PYTHON)
    if(DEFINED SKBUILD_PLATLIB_DIR) # pip / scikit-build-core install
        message(STATUS "SKBUILD_PLATLIB_DIR = '${SKBUILD_PLATLIB_DIR}'")
        install(TARGETS quadriga_py LIBRARY DESTINATION ${SKBUILD_PLATLIB_DIR})
        install(DIRECTORY data/config/ DESTINATION ${SKBUILD_DATA_DIR}/share/quadriga-lib/config
                FILES_MATCHING PATTERN "*.conf")
    elseif(INSTALL_IS_LOCAL)
        install(TARGETS quadriga_py
            ARCHIVE DESTINATION ${INSTALL_LIBDIR}
            LIBRARY DESTINATION ${INSTALL_LIBDIR})
    else()
        # Install to Python dist-packages only if they live under CMAKE_INSTALL_PREFIX.
        # For custom prefixes (e.g. /sjc/qrt/quadriga_lib), fall back to INSTALL_LIBDIR.
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

        # Only use the Python dist path if it's actually under the install prefix
        string(FIND "${PYTHON3_DIST_PACKAGES}" "${CMAKE_INSTALL_PREFIX}/" _dist_under_prefix)
        if(_dist_under_prefix EQUAL 0)
            string(REGEX REPLACE "^${CMAKE_INSTALL_PREFIX}/?" "" PYTHON3_DIST_REL "${PYTHON3_DIST_PACKAGES}")
            message(STATUS "Python install dir: ${PYTHON3_DIST_REL}")
            install(TARGETS quadriga_py LIBRARY DESTINATION ${PYTHON3_DIST_REL})
        else()
            message(STATUS "Python install dir: ${INSTALL_LIBDIR} (custom prefix, not system Python)")
            install(TARGETS quadriga_py LIBRARY DESTINATION ${INSTALL_LIBDIR})
        endif()
    endif()
endif()
