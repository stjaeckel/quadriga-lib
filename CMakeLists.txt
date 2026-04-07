cmake_minimum_required(VERSION 3.18)
project(QuadrigaLib LANGUAGES C CXX)

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
option(ENABLE_BIN "Build supporting binary files" OFF)

# Path to the HDF5 library for static linking (include files and .a/.lib file)
# If left empty, HDF location is detected automatically, if not found, HDF5 is built from sources
option(HDF5_PATH "Location of the HDF5 library" "")

# If HDF5_STATIC is ON, HDF5 is build from sources and statically linked
# Static linking the HDF5 library may cause Octave to crash but usually works fine with MATLAB
option(HDF5_STATIC "Link HDF5 statically" OFF)

# Cmake tries to detect Armadillo include files on the system and link to these files.
# If Armadillo is not found, the version provided in the "external" folder is used instead.
option(ARMA_EXT "Use armadillo headers provided with quadriga-lib" OFF)

# --- Validation Rules ---
if(NOT ENABLE_STATIC_LIB AND NOT ENABLE_SHARED_LIB)
    message(FATAL_ERROR "At least one of ENABLE_STATIC_LIB or ENABLE_SHARED_LIB must be ON")
endif()

# --- Platform Guards ---
if(MSVC)
    if(ENABLE_SHARED_LIB)
        message(WARNING "Shared library not supported on Windows, forcing ENABLE_SHARED_LIB=OFF")
        set(ENABLE_SHARED_LIB OFF)
    endif()
    if(ENABLE_CUDA)
        message(WARNING "CUDA not supported on Windows, forcing ENABLE_CUDA=OFF")
        set(ENABLE_CUDA OFF)
    endif()
endif()

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
    set(MEX_CXXFLAGS "/std:c++17 /nologo $<$<CONFIG:Debug>:/MDd> $<$<CONFIG:Release>:/MD>")
else()
    add_compile_options(-O3 -fPIC -Wall -Wpedantic -Wextra)
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

# Components to build
set(BUILD_HDF5 ${HDF5_STATIC})
set(BUILD_MATLAB ${ENABLE_MATLAB})
if(NOT MSVC)
    set(BUILD_OCTAVE ${ENABLE_OCTAVE})
endif()
set(BUILD_MEX_DOC ${ENABLE_MEX_DOC})
if(NOT MSVC)
    set(BUILD_PYTHON ${ENABLE_PYTHON})
endif()
set(BUILD_SHARED_LIB ${ENABLE_SHARED_LIB})
set(BUILD_CATCH2 ${ENABLE_TESTS})

if (NOT ENABLE_PYTHON)
    set(BUILD_MEX_DOC OFF)
endif()

if (MSVC)
    set(BUILD_MEX_DOC OFF)
endif()

# Build instructions for HDF5 library
if (NOT HDF5_PATH AND NOT BUILD_HDF5) # Detect HDF5 automatically
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

if (HDF5_PATH OR BUILD_HDF5) # Disable Python and Octave
    if (BUILD_PYTHON)
        message(STATUS "Python API does not support static linking HDF5 library. Python disabled!")
        set(BUILD_PYTHON OFF)
    endif()
    if (BUILD_OCTAVE)
        message(STATUS "Octave does not support static linking HDF5 library. Octave disabled!")
        set(BUILD_OCTAVE OFF)
    endif()
endif()

if (HDF5_PATH) # Use given HDF5 location before building it
    set(BUILD_HDF5 OFF)
    get_filename_component(HDF5_PATH_ABSOLUTE "${HDF5_PATH}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
    set(HDF5_INCLUDE_DIRS "${HDF5_PATH}/include")
    if (MSVC)
        set(HDF5_LIB "${HDF5_PATH_ABSOLUTE}/lib/libhdf5.lib")
    else()
        set(HDF5_LIB "${HDF5_PATH_ABSOLUTE}/lib/libhdf5.a")
    endif()
endif()

if (BUILD_HDF5)
    set(HDF5_EP_ARGS
        URL ${HDF5_ZIP}
        BINARY_DIR "${CMAKE_BINARY_DIR}/hdf5-${hdf5_version}_build"
        INSTALL_DIR "${CMAKE_BINARY_DIR}/hdf5-${hdf5_version}_bin"
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
            -DBUILD_SHARED_LIBS=OFF
            -DHDF5_ENABLE_Z_LIB_SUPPORT=OFF
            -DBUILD_TESTING=OFF
            -DHDF5_BUILD_TOOLS=OFF
            -DHDF5_BUILD_EXAMPLES=OFF
            -DHDF5_BUILD_HL_LIB=OFF
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    )
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
        list(APPEND HDF5_EP_ARGS DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    endif()

    if (MSVC)
        set(HDF5_LIB "${CMAKE_BINARY_DIR}/hdf5-${hdf5_version}_bin/lib/libhdf5.lib")
    else()
        set(HDF5_LIB "${CMAKE_BINARY_DIR}/hdf5-${hdf5_version}_bin/lib/libhdf5.a")
    endif()
    set(HDF5_INCLUDE_DIRS "${CMAKE_BINARY_DIR}/hdf5-${hdf5_version}_bin/include")

    ExternalProject_Add(hdf5_dep ${HDF5_EP_ARGS})
endif()

# Build instructions for CATCH2 library
if (BUILD_CATCH2)
    set(CATCH2_EP_ARGS
        URL ${CATCH2_ZIP}
        BINARY_DIR "${CMAKE_BINARY_DIR}/catch2-${catch2_version}_build"
        INSTALL_DIR "${CMAKE_BINARY_DIR}/catch2-${catch2_version}_bin"
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
    )
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
        list(APPEND CATCH2_EP_ARGS DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    endif()

    if (MSVC) 
        set(CATCH2_LIB "${CMAKE_BINARY_DIR}/catch2-${catch2_version}_bin/lib/Catch2.lib")
    else()
        set(CATCH2_LIB "${CMAKE_BINARY_DIR}/catch2-${catch2_version}_bin/lib/libCatch2.a")
    endif()
    set(CATCH2_INCLUDE_DIR "${CMAKE_BINARY_DIR}/catch2-${catch2_version}_bin/include")

    ExternalProject_Add(catch2_dep ${CATCH2_EP_ARGS})
endif()

# Detect MATLAB MEX headers (vendored in external/matlab_mex.zip, no MATLAB installation needed)
if(BUILD_MATLAB)
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

# Detect Octave
if (BUILD_OCTAVE)
    find_program(MKOCTFILE_EXECUTABLE mkoctfile)
    if(MKOCTFILE_EXECUTABLE)
        message(STATUS "Found mkoctfile: ${MKOCTFILE_EXECUTABLE}")
    else()
        message(STATUS "Octave not found, Octave API disabled!")
        set(BUILD_OCTAVE OFF)
    endif()
else()
    message(STATUS "Octave disabled")
endif()

# Detect python
if (BUILD_PYTHON OR BUILD_MEX_DOC)
    find_package(Python3 COMPONENTS Interpreter Development)
    if (NOT Python3_FOUND)
        set(BUILD_PYTHON OFF)
        set(BUILD_MEX_DOC OFF)
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
    set(SRC_AVX2 "")
endif()

# For files that require CUDA, detect nvcc and add CUDA language support.
set(BUILD_WITH_CUDA 0)
set(CUDA_LINK_LIBS "")
set(CUDA_MEX_LINK_LIBS "")
if(ENABLE_CUDA)
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

# Static library
if(ENABLE_STATIC_LIB)
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
    if (BUILD_HDF5 AND NOT HDF5_PATH)
        add_dependencies(quadriga_static hdf5_dep)
    endif()
    if (NOT ARMADILLO_FOUND)
        add_dependencies(quadriga_static armadillo_lib)
    endif()
endif()

# Shared library
if (BUILD_SHARED_LIB)
    add_library(quadriga_shared SHARED ${SRC_FILES} ${SRC_AVX2} ${SRC_CUDA})
    target_include_directories(quadriga_shared PRIVATE
        ${QUADRIGA_CORE_INCLUDES}
        ${HDF5_INCLUDE_DIRS}
    )
    set_target_properties(quadriga_shared PROPERTIES OUTPUT_NAME quadriga)
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
    if (BUILD_HDF5 AND NOT HDF5_PATH)
        add_dependencies(quadriga_shared hdf5_dep)
        target_link_libraries(quadriga_shared PUBLIC ${HDF5_LIB})
    elseif(HDF5_PATH)
        target_link_libraries(quadriga_shared PUBLIC ${HDF5_LIB})
    elseif(HDF5_FOUND)
        target_link_libraries(quadriga_shared PUBLIC HDF5::HDF5)
    endif()
endif()

# --- Link Target Selection ---
if(ENABLE_SHARED_LIB)
    set(QUADRIGA_LINK_TARGET quadriga_shared)
else()
    set(QUADRIGA_LINK_TARGET quadriga_static)
endif()

# --- MEX link variables (shared by MATLAB and Octave sections) ---
if(BUILD_MATLAB OR BUILD_OCTAVE)
    if(HDF5_FOUND)
        set(HDF5_MEX_LINK_LIBS ${HDF5_LIBRARIES})
    else()
        set(HDF5_MEX_LINK_LIBS "")
    endif()

    if(QUADRIGA_LINK_TARGET STREQUAL "quadriga_shared")
        set(MEX_HDF5_LINK_LIBS "")
        set(MEX_CUDA_LINK_LIBS "")
        set(MEX_RPATH_FLAGS "-Wl,-rpath,$ORIGIN")
    else()
        set(MEX_HDF5_LINK_LIBS ${HDF5_LIB} ${HDF5_MEX_LINK_LIBS})
        set(MEX_CUDA_LINK_LIBS ${CUDA_MEX_LINK_LIBS})
        set(MEX_RPATH_FLAGS "")
    endif()
endif()

# --- MATLAB API ---
# Compiles MEX files directly with g++ (Linux) or cl.exe (Windows). No MATLAB installation required
if(BUILD_MATLAB)
    file(MAKE_DIRECTORY ${MEX_OUTPUT_DIR})
    file(GLOB API_MEX_FILES "${CMAKE_SOURCE_DIR}/api_mex/*.cpp")
    set(MEX_MATLAB_TARGETS "")

    foreach(mex_src ${API_MEX_FILES})
        get_filename_component(mex_basename ${mex_src} NAME_WE)

        # Uniform dependency: all MEX files depend on the link target
        set(mex_deps ${mex_src})
        list(APPEND mex_deps ${QUADRIGA_LINK_TARGET})
        if(mex_basename STREQUAL "ieee_chan_indoor")
            list(APPEND mex_deps ${CMAKE_SOURCE_DIR}/src/ieee_channel_models.hpp)
        endif()

        if(MSVC)
            set(mex_output "${MEX_OUTPUT_DIR}/${mex_basename}.mexw64")
            add_custom_command(
                OUTPUT ${mex_output}
                COMMAND ${CMAKE_CXX_COMPILER}
                        /nologo /std:c++17 /EHs /O2 /MD /LD
                        /DMATLAB_MEX_FILE
                        /I${MATLAB_MEX_INCLUDE_DIR}
                        /I${CMAKE_SOURCE_DIR}/include
                        /I${CMAKE_SOURCE_DIR}/src
                        /I${ARMADILLO_INCLUDE_DIRS}
                        ${mex_src}
                        /link
                        $<TARGET_FILE:${QUADRIGA_LINK_TARGET}>
                        ${MEX_HDF5_LINK_LIBS} ${MEX_CUDA_LINK_LIBS}
                        shlwapi.lib
                        "${MATLAB_MEX_LIB_DIR}/libmex.lib"
                        "${MATLAB_MEX_LIB_DIR}/libmx.lib"
                        /OUT:${mex_output}
                        /DLL
                        /EXPORT:mexFunction
                DEPENDS ${mex_deps}
                COMMENT "Building MATLAB MEX file ${mex_basename}.mexw64"
                VERBATIM
            )
        else()
            set(mex_output "${MEX_OUTPUT_DIR}/${mex_basename}.mexa64")
            add_custom_command(
                OUTPUT ${mex_output}
                COMMAND ${CMAKE_CXX_COMPILER}
                        -std=c++17 -O3 -shared -fPIC
                        -DMATLAB_MEX_FILE
                        -I${MATLAB_MEX_INCLUDE_DIR}
                        -I${CMAKE_SOURCE_DIR}/include
                        -I${CMAKE_SOURCE_DIR}/src
                        -I${ARMADILLO_INCLUDE_DIRS}
                        ${mex_src}
                        $<TARGET_FILE:${QUADRIGA_LINK_TARGET}>
                        ${MEX_HDF5_LINK_LIBS} ${MEX_CUDA_LINK_LIBS}
                        -lgomp
                        ${MEX_RPATH_FLAGS}
                        -Wl,--version-script,${MATLAB_MEX_VERSION_SCRIPT}
                        -Wl,--no-undefined-version
                        -Wl,--unresolved-symbols=ignore-all
                        -o ${mex_output}
                DEPENDS ${mex_deps}
                COMMENT "Building MATLAB MEX file ${mex_basename}.mexa64"
                VERBATIM
            )
        endif()

        add_custom_target(mex_${mex_basename} ALL DEPENDS ${mex_output})
        add_dependencies(mex_${mex_basename} matlab_mex_unzip)
        list(APPEND MEX_MATLAB_TARGETS mex_${mex_basename})
    endforeach()

    add_custom_target(mex_matlab ALL DEPENDS ${MEX_MATLAB_TARGETS})
endif()

# --- Octave API ---
if(BUILD_OCTAVE)
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

        if(QUADRIGA_LINK_TARGET STREQUAL "quadriga_shared")
            execute_process(COMMAND ${MKOCTFILE_EXECUTABLE} -p INCFLAGS OUTPUT_VARIABLE OCTAVE_INCFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
            execute_process(COMMAND ${MKOCTFILE_EXECUTABLE} -p OCTINCLUDEDIR OUTPUT_VARIABLE OCTAVE_OCTINCLUDEDIR OUTPUT_STRIP_TRAILING_WHITESPACE)
            execute_process(COMMAND ${MKOCTFILE_EXECUTABLE} -p OCTLIBDIR OUTPUT_VARIABLE OCTAVE_LIBDIR OUTPUT_STRIP_TRAILING_WHITESPACE)

            # Shared lib: use direct g++ (CUDA already resolved inside the .so)
            add_custom_command(
                OUTPUT ${octave_output}
                COMMAND ${CMAKE_CXX_COMPILER}
                        -std=c++17 -O3 -shared -fPIC -DMATLAB_MEX_FILE ${OCTAVE_INCFLAGS}
                        -I${OCTAVE_OCTINCLUDEDIR} -I${CMAKE_SOURCE_DIR}/include -I${CMAKE_SOURCE_DIR}/src -I${ARMADILLO_INCLUDE_DIRS}
                        ${mex_src} $<TARGET_FILE:${QUADRIGA_LINK_TARGET}>
                        ${MEX_HDF5_LINK_LIBS} ${MEX_CUDA_LINK_LIBS} -L${OCTAVE_LIBDIR} -loctinterp -loctave -lgomp
                        ${MEX_RPATH_FLAGS} -o ${octave_output}
                DEPENDS ${mex_deps}
                COMMENT "Building Octave MEX file ${octave_output}"
                VERBATIM
            )
        else()
            # Static lib: must use mkoctfile to handle CUDA correctly
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
            COMMAND python3 ${CMAKE_SOURCE_DIR}/tools/extract_matlab_comments.py 
                    ${mex_src} ${m_file_output}
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
    set_target_properties(quadriga_py PROPERTIES
        PREFIX ""
        OUTPUT_NAME "quadriga_lib"
        SUFFIX "${PYTHON_EXTENSION_SUFFIX}"
        POSITION_INDEPENDENT_CODE ON
        INSTALL_RPATH "$ORIGIN"
        BUILD_RPATH "$ORIGIN"
    )
    if(QUADRIGA_LINK_TARGET STREQUAL "quadriga_shared")
        target_link_libraries(quadriga_py PRIVATE ${QUADRIGA_LINK_TARGET} ${Python3_LIBRARIES})
    else()
        target_link_libraries(quadriga_py PRIVATE ${QUADRIGA_LINK_TARGET} ${Python3_LIBRARIES} HDF5::HDF5 ${CUDA_LINK_LIBS})
    endif()
    target_compile_options(quadriga_py PRIVATE -Wno-unused-function)

    file(GLOB PYTHON_INCLUDED_FILES "api_python/*.hpp" "api_python/*.cpp")
    set_source_files_properties(${PYTHON_SOURCES} PROPERTIES
        OBJECT_DEPENDS "${PYTHON_INCLUDED_FILES}"
    )
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
        if (BUILD_HDF5)
            target_link_libraries(test_bin PRIVATE ${HDF5_LIB})
        else()
            target_link_libraries(test_bin PRIVATE HDF5::HDF5)
        endif()
    endif()
    
    if(MSVC)
        target_link_libraries(test_bin PRIVATE shlwapi.lib)
        target_compile_options(test_bin PRIVATE /EHsc /Zc:__cplusplus /nologo /MP)
        target_compile_definitions(test_bin PRIVATE _USE_MATH_DEFINES ARMA_DONT_USE_BLAS ARMA_DONT_USE_LAPACK)
    else()
        target_compile_options(test_bin PRIVATE -O3 -fPIC -w)
    endif()
endif()

# --- Binaries ---
if (ENABLE_BIN)
    file(GLOB SRC_BIN "${CMAKE_SOURCE_DIR}/src/bin/*.cpp")
    foreach(source IN LISTS SRC_BIN)
        get_filename_component(exeName ${source} NAME_WE)

        add_executable(${exeName} ${source})
        add_dependencies(${exeName} ${QUADRIGA_LINK_TARGET})
        target_include_directories(${exeName} PRIVATE ${QUADRIGA_CORE_INCLUDES})
        
        target_link_libraries(${exeName} PRIVATE ${QUADRIGA_LINK_TARGET})
        if(QUADRIGA_LINK_TARGET STREQUAL "quadriga_static")
            if (BUILD_HDF5)
                target_link_libraries(${exeName} PRIVATE ${HDF5_LIB})
            else()
                target_link_libraries(${exeName} PRIVATE HDF5::HDF5)
            endif()
        endif()
    endforeach()
endif()

# Install options
if(CMAKE_INSTALL_PREFIX STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}")
    # Default: install to source directory (local / development layout)
    message(STATUS "Using source dir as install dir")

    set(INSTALL_LIBDIR lib)
    set(INSTALL_BINDIR bin)
    set(INSTALL_INCLUDEDIR include)
    set(INSTALL_MEX_DIR .)
    set(INSTALL_DOCDIR html_docu)
    set(INSTALL_LICENSEDIR .)
    set(INSTALL_ARMADIR .)
    set(INSTALL_IS_LOCAL TRUE)
else()
    # System install: use GNUInstallDirs for Debian/FHS-standard paths
    include(GNUInstallDirs)
    message(STATUS "Installing to: ${CMAKE_INSTALL_PREFIX}")

    set(INSTALL_LIBDIR ${CMAKE_INSTALL_LIBDIR})
    set(INSTALL_BINDIR ${CMAKE_INSTALL_BINDIR})
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

if(ENABLE_STATIC_LIB)
    install(TARGETS quadriga_static
            ARCHIVE DESTINATION ${INSTALL_LIBDIR}
            LIBRARY DESTINATION ${INSTALL_LIBDIR}
            RUNTIME DESTINATION ${INSTALL_BINDIR})
endif()

if (BUILD_SHARED_LIB)
    install(TARGETS quadriga_shared
            ARCHIVE DESTINATION ${INSTALL_LIBDIR}
            LIBRARY DESTINATION ${INSTALL_LIBDIR}
            RUNTIME DESTINATION ${INSTALL_BINDIR})
    # For local installs, copy shared lib into MEX directory so MATLAB/Octave find it
    if(INSTALL_IS_LOCAL)
        install(FILES $<TARGET_FILE:quadriga_shared> DESTINATION ${INSTALL_MEX_DIR}/+quadriga_lib)
    endif()
endif()

if(BUILD_MATLAB OR BUILD_OCTAVE)
    install(DIRECTORY ${MEX_OUTPUT_DIR} DESTINATION ${INSTALL_MEX_DIR})
endif()

if (BUILD_PYTHON)
    if(INSTALL_IS_LOCAL)
        install(TARGETS quadriga_py
            ARCHIVE DESTINATION ${INSTALL_LIBDIR}
            LIBRARY DESTINATION ${INSTALL_LIBDIR}
            RUNTIME DESTINATION ${INSTALL_BINDIR})
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

if (BUILD_HDF5 OR HDF5_PATH)
    install(FILES ${HDF5_LIB} DESTINATION ${INSTALL_LIBDIR})
endif()