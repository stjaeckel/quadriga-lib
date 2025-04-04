cmake_minimum_required(VERSION 3.18)
project(QuadrigaLib LANGUAGES C CXX)

# Windows:
# rmdir /s /q build_win
# cmake -B build_win -D HDF5_STATIC_PATH=external/hdf5-1.14.2-win64

# Option to enable MATLAB/Octace MEX compilation (default: ON)
# For octave, make sure that 'mkoctfile' is installed and can be called
option(ENABLE_MATLAB "Enable MATLAB MEX API" ON)
option(ENABLE_OCTAVE "Enable Octave MEX API" ON)
option(ENABLE_MEX_DOC "Enable MEX Documentation" ON)
option(ENABLE_PYTHON "Enable Python API" ON)

# Option to specify MATLAB path manually (leave empty for auto-detect)
option(MATLAB_PATH "Path to MATLAB installation" "")

# If HDF5_STATIC_LINK is ON, we assume static linking (e.g. HDF5 version provided locally)
# Static linking the HDF5 library may cause Octave to crash but usually works fine with MATLAB
option(HDF5_STATIC_LINK "Link HDF5 statically" OFF)
option(HDF5_STATIC_PATH "Location of the HDF5 library" "")

# Versions for external libraries (adjust as needed)
set(armadillo_version "14.2.2")
set(pugixml_version "1.13")
set(pybind11_version "2.12.0")
set(hdf5_version "1.14.2")

# Set the C++ standard and basic compile flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(MEX_OUTPUT_DIR "${CMAKE_BINARY_DIR}/+quadriga_lib")

if(MSVC)
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /EHsc /Zc:__cplusplus /nologo /openmp /MP /FS")
    set(CMAKE_CXX_FLAGS "/EHsc /Zc:__cplusplus /nologo /openmp /MP")
    set(CMAKE_CXX_FLAGS_RELEASE "")
    set(MEX_CXXFLAGS "/std:c++17 /nologo $<$<CONFIG:Debug>:/MDd> $<$<CONFIG:Release>:/MD>")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC -fopenmp -Wall -Wconversion -Wpedantic -Wextra")
    set(MEX_CXXFLAGS "-std=c++17 ${CMAKE_CXX_FLAGS}")
endif()

include(ExternalProject)

# External library files
set(ARMADILLO_ZIP "${CMAKE_SOURCE_DIR}/external/armadillo-${armadillo_version}.zip")
set(PUGIXML_ZIP "${CMAKE_SOURCE_DIR}/external/pugixml-${pugixml_version}.zip")
set(PYBIND11_ZIP "${CMAKE_SOURCE_DIR}/external/pybind11-${pybind11_version}.zip")
set(HDF5_ZIP "${CMAKE_SOURCE_DIR}/external/hdf5-${hdf5_version}.zip")

# External packages
set(ARMADILLO_SRC_DIR "${CMAKE_BINARY_DIR}/armadillo-${armadillo_version}")
set(PUGIXML_SRC_DIR "${CMAKE_BINARY_DIR}/pugixml-${pugixml_version}")
set(PYBIND11_SRC_DIR "${CMAKE_BINARY_DIR}/pybind11-${pybind11_version}")

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

# Create a custom target for the unzip step.
add_custom_target(armadillo_lib ALL DEPENDS ${ARMADILLO_SRC_DIR}/.stamp)
add_custom_target(pugixml_lib ALL DEPENDS ${PUGIXML_SRC_DIR}/.stamp)
add_custom_target(pybind11_lib ALL DEPENDS ${PYBIND11_SRC_DIR}/.stamp)

# External include directories
set(ARMA_H      "${ARMADILLO_SRC_DIR}/include")
set(PUGIXML_H   "${PUGIXML_SRC_DIR}/src")
set(PYBIND11_H  "${PYBIND11_SRC_DIR}/include")

# Components to build
set(BUILD_HDF5 ${HDF5_STATIC_LINK})
set(BUILD_MATLAB ${ENABLE_MATLAB})
set(BUILD_OCTAVE ${ENABLE_OCTAVE})
set(BUILD_MEX_DOC ${ENABLE_MEX_DOC})
set(BUILD_PYTHON ${ENABLE_PYTHON})

if (MSVC)
    set(BUILD_PYTHON OFF)
    set(BUILD_MEX_DOC OFF)
endif()

# Build instructions for HDF5 library
if (NOT BUILD_HDF5)
    find_package(HDF5 COMPONENTS C HL)
    if(HDF5_FOUND)
        get_filename_component(HDF5_LIB_DIR "${HDF5_LIBRARIES}" DIRECTORY)
        message(STATUS "HDF5 Include dir: ${HDF5_INCLUDE_DIRS}")
        message(STATUS "HDF5 Library dir: ${HDF5_LIB_DIR}")
    else()
        set(BUILD_HDF5 ON)
    endif()
endif()
if (BUILD_HDF5)
    if (BUILD_PYTHON)
        message(STATUS "Python API does not support static linking HDF5 library. Python disabled!")
        set(BUILD_PYTHON OFF)
    endif()
    if (HDF5_STATIC_PATH)
        get_filename_component(HDF5_STATIC_ABS_PATH "${HDF5_STATIC_PATH}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
        set(HDF5_INCLUDE_DIRS "${HDF5_STATIC_PATH}/include")
        if (MSVC)
            set(HDF5_A "${HDF5_STATIC_ABS_PATH}/lib/libhdf5.lib")
        else()
            set(HDF5_A "${HDF5_STATIC_ABS_PATH}/lib/libhdf5.a")
        endif()
    else()
        set(HDF5_INCLUDE_DIRS "${CMAKE_BINARY_DIR}/hdf5-${hdf5_version}_bin/include")
        if (MSVC)
            set(HDF5_A "${CMAKE_BINARY_DIR}/hdf5-${hdf5_version}_bin/lib/libhdf5.lib")
        else()
            set(HDF5_A "${CMAKE_BINARY_DIR}/hdf5-${hdf5_version}_bin/lib/libhdf5.a")
        endif()

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
        )

        if(MSVC)
            list(APPEND HDF5_EP_ARGS DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
        endif()

        ExternalProject_Add(hdf5_dep ${HDF5_EP_ARGS})
    endif()
endif()

# Detect MATLAB
if(BUILD_MATLAB AND MATLAB_PATH)
    message(STATUS "Using user-specified MATLAB at: ${MATLAB_PATH}")
    set(MEX_EXECUTABLE "${MATLAB_PATH}/bin/mex")
    message(STATUS "Using MEX compiler at: ${MEX_EXECUTABLE}")
elseif(BUILD_MATLAB)
    find_package(Matlab REQUIRED COMPONENTS MEX_COMPILER)
    if(Matlab_FOUND)
        message(STATUS "Found MEX compiler at: ${Matlab_MEX_COMPILER}")
        set(MEX_EXECUTABLE "${Matlab_MEX_COMPILER}")
    else()
        message(STATUS "MATLAB not found, MATLAB API disabled!")
        set(BUILD_MATLAB OFF)
    endif()
else()
    message(STATUS "MATLAB disabled")
endif()

# Detect Ocatve
if (MSVC AND BUILD_OCTAVE)
    message(STATUS "Octave for Windows is not supported!")
    set(BUILD_OCTAVE OFF)
endif()
if (BUILD_OCTAVE)
    find_program(MKOCTFILE_EXECUTABLE mkoctfile)
    if(MKOCTFILE_EXECUTABLE)
        message(STATUS "Found mkoctfile: ${MKOCTFILE_EXECUTABLE}")
    else()
        message(STATUS "Octave not found, Octave API disabled!")
        set(BUILD_OCTAVE OFF)
    endif()
    if (BUILD_OCTAVE AND BUILD_HDF5)
        message(STATUS "Octave is not compatible with statically linked HDF5 library. Octave compilation disabled!")
        set(BUILD_OCTAVE OFF)
    endif()
else()
    message(STATUS "Octave disabled")
endif()

# Detect python
if (NOT ENABLE_PYTHON)
    set(BUILD_MEX_DOC OFF)
endif()
if (BUILD_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development)
    if (NOT Python3_FOUND)
        set(BUILD_PYTHON OFF)
        set(BUILD_MEX_DOC OFF)
    endif()
endif()

# Add include directories common to all targets
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${ARMA_H}
    ${PUGIXML_H}
    ${HDF5_INCLUDE_DIRS}
)

# Locate all core source files
file(GLOB SRC_FILES "${CMAKE_SOURCE_DIR}/src/*.cpp")
file(GLOB SRC_AVX2 "${CMAKE_SOURCE_DIR}/src/*_avx2.cpp")

# For files that require AVX2/FMA, add the extra flags.
foreach(src ${SRC_AVX2})
    if (MSVC)
        set_source_files_properties(${src} PROPERTIES COMPILE_FLAGS "/arch:AVX2")
        list(REMOVE_ITEM SRC_FILES "${CMAKE_SOURCE_DIR}/src/baseband_freq_response_avx2.cpp")
        list(REMOVE_ITEM SRC_AVX2 "${CMAKE_SOURCE_DIR}/src/baseband_freq_response_avx2.cpp")
    else()
        set_source_files_properties(${src} PROPERTIES COMPILE_FLAGS "-mavx2 -mfma")
    endif()
endforeach()

# Static library
add_library(quadriga_static STATIC ${SRC_FILES} ${SRC_AVX2})
set_target_properties(quadriga_static PROPERTIES OUTPUT_NAME quadriga)
add_dependencies(quadriga_static armadillo_lib pugixml_lib pybind11_lib)

if (BUILD_HDF5 AND NOT HDF5_STATIC_PATH)
    add_dependencies(quadriga_static hdf5_dep)
endif()

# Shared library
if (NOT BUILD_HDF5)
    add_library(quadriga_shared SHARED ${SRC_FILES} ${SRC_AVX2})
    set_target_properties(quadriga_shared PROPERTIES OUTPUT_NAME quadriga)
    add_dependencies(quadriga_shared armadillo_lib pugixml_lib pybind11_lib)
endif()

# --- MATLAB API ---
if(BUILD_MATLAB)
    file(MAKE_DIRECTORY ${MEX_OUTPUT_DIR})

    if (NOT HDF5_STATIC_LINK AND NOT MSVC)
        set(HDF_L "-L${HDF5_LIB_DIR}")
        set(HDF_l "-lhdf5")
    endif()

    file(GLOB API_MEX_FILES "${CMAKE_SOURCE_DIR}/api_mex/*.cpp")
    set(MEX_MATLAB_TARGETS "")

    foreach(mex_src ${API_MEX_FILES})
        get_filename_component(mex_basename ${mex_src} NAME_WE)

        if (MSVC)
            set(mex_output "${MEX_OUTPUT_DIR}/${mex_basename}.mexw64")
            add_custom_command(
                OUTPUT ${mex_output}
                COMMAND ${MEX_EXECUTABLE} -silent -outdir ${MEX_OUTPUT_DIR} 
                        COMPFLAGS="${MEX_CXXFLAGS}"
                        ${mex_src} $<TARGET_FILE:quadriga_static> ${HDF5_A} shlwapi.lib 
                        -I${CMAKE_SOURCE_DIR}/include -I${CMAKE_SOURCE_DIR}/src -I${ARMA_H}
                DEPENDS ${mex_src} quadriga_static
                COMMENT "Building MATLAB MEX file ${mex_output}"
                VERBATIM
            )
            add_custom_target(mex_${mex_basename} ALL DEPENDS ${mex_output})
        else()
            set(mex_output "${MEX_OUTPUT_DIR}/${mex_basename}.mexa64")
            add_custom_command(
                OUTPUT ${mex_output}
                COMMAND env CXXFLAGS="${MEX_CXXFLAGS}" ${MEX_EXECUTABLE} -silent -outdir ${MEX_OUTPUT_DIR} 
                        ${mex_src} $<TARGET_FILE:quadriga_static> ${HDF5_A}
                        -I${CMAKE_SOURCE_DIR}/include -I${CMAKE_SOURCE_DIR}/src -I${ARMA_H}
                        -lgomp ${HDF_L} ${HDF_l}
                DEPENDS ${mex_src} quadriga_static
                COMMENT "Building MATLAB MEX file ${mex_output}"
                VERBATIM
            )
            list(APPEND MEX_MATLAB_TARGETS ${mex_output})
        endif()
    endforeach()
    if (NOT MSVC)
        add_custom_target(mex_matlab ALL DEPENDS ${MEX_MATLAB_TARGETS})
    endif()
endif()

# --- Octave API ---
if(BUILD_OCTAVE)
    file(MAKE_DIRECTORY ${MEX_OUTPUT_DIR})
    file(GLOB API_MEX_FILES "${CMAKE_SOURCE_DIR}/api_mex/*.cpp")
    set(MEX_OCTAVE_TARGETS "")

    foreach(mex_src ${API_MEX_FILES})
        get_filename_component(mex_basename ${mex_src} NAME_WE)
        set(octave_output "${MEX_OUTPUT_DIR}/${mex_basename}.mex")
        
        add_custom_command(
            OUTPUT ${octave_output}
            COMMAND env "CXXFLAGS=${MEX_CXXFLAGS}" ${MKOCTFILE_EXECUTABLE} --mex -o ${octave_output}
                    ${mex_src} $<TARGET_FILE:quadriga_static>
                    -I${CMAKE_SOURCE_DIR}/include -I${CMAKE_SOURCE_DIR}/src -I${ARMA_H} -s
            DEPENDS ${mex_src} quadriga_static
            COMMENT "Building Octave MEX file ${octave_output}"
            VERBATIM
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

    set(PYTHON_MAIN_SRC "api_python/python_main.cpp")
    add_library(quadriga_py MODULE ${PYTHON_MAIN_SRC})
    set_target_properties(quadriga_py PROPERTIES
        PREFIX ""
        OUTPUT_NAME "quadriga_lib"
        SUFFIX "${PYTHON_EXTENSION_SUFFIX}"
    )
    target_include_directories(quadriga_py PRIVATE ${Python3_INCLUDE_DIRS})
    target_link_libraries(quadriga_py PRIVATE quadriga_static ${Python3_LIBRARIES} HDF5::HDF5)

    file(GLOB PYTHON_INCLUDED_FILES "api_python/*.hpp" "api_python/*.cpp")
    set_source_files_properties(${PYTHON_MAIN_SRC} PROPERTIES
        OBJECT_DEPENDS "${PYTHON_INCLUDED_FILES}"
    )
    message(STATUS "Building Python module: quadriga_lib${PYTHON_EXTENSION_SUFFIX}")
endif()

# Install options
if(CMAKE_INSTALL_PREFIX STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}")
    message(STATUS "Using source dir as install dir")
else()    
    install(DIRECTORY include/ DESTINATION include)
endif()

install(TARGETS quadriga_static
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)

if (NOT BUILD_HDF5)
    install(TARGETS quadriga_shared
            ARCHIVE DESTINATION lib
            LIBRARY DESTINATION lib
            RUNTIME DESTINATION bin) 
endif()

if(BUILD_MATLAB OR BUILD_OCTAVE)
    install(DIRECTORY ${MEX_OUTPUT_DIR} DESTINATION .)
endif()

if (BUILD_PYTHON)
    install(TARGETS quadriga_py
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)
endif()

if (BUILD_HDF5)
    install(FILES ${HDF5_A} DESTINATION lib)
endif()