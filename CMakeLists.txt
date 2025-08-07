cmake_minimum_required(VERSION 3.18)
project(QuadrigaLib LANGUAGES C CXX)

# Option to enable MATLAB/Octace MEX compilation (default: ON)
# For octave, make sure that 'mkoctfile' is installed and can be called
option(ENABLE_MATLAB "Enable MATLAB MEX API" ON)
option(ENABLE_OCTAVE "Enable Octave MEX API" ON)
option(ENABLE_MEX_DOC "Enable MEX Documentation" ON)
option(ENABLE_PYTHON "Enable Python API" ON)
option(ENABLE_SHARED_LIB "Enable build of shared library" OFF)
option(ENABLE_AVX2 "Enable AVX2 Acceleration" OFF)
option(ENABLE_TESTS "Build tests" OFF)
option(ENABLE_BIN "Build supporting binary files" OFF)

# Option to specify MATLAB path manually (leave empty for auto-detect)
option(MATLAB_PATH "Path to MATLAB" "")

# Path to the HDF5 library for static linking (include files and .a/.lib file)
# If left empty, HDF location is detected automatically, if not found, HDF5 is built from sources
option(HDF5_PATH "Location of the HDF5 library" "")

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
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /EHsc /Zc:__cplusplus /nologo /openmp /MP /FS")
    set(CMAKE_CXX_FLAGS "/EHsc /Zc:__cplusplus /nologo /openmp /MP")
    set(CMAKE_CXX_FLAGS_RELEASE "")
    set(MEX_CXXFLAGS "/std:c++17 /nologo $<$<CONFIG:Debug>:/MDd> $<$<CONFIG:Release>:/MD>")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC -fopenmp -Wall -Wpedantic -Wextra")
    set(MEX_CXXFLAGS "-std=c++17 ${CMAKE_CXX_FLAGS}")
endif()

include(ExternalProject)
include(CheckCXXCompilerFlag)

# External library files
set(ARMADILLO_ZIP "${CMAKE_SOURCE_DIR}/external/armadillo-${armadillo_version}.zip")
set(PUGIXML_ZIP "${CMAKE_SOURCE_DIR}/external/pugixml-${pugixml_version}.zip")
set(PYBIND11_ZIP "${CMAKE_SOURCE_DIR}/external/pybind11-${pybind11_version}.zip")
set(HDF5_ZIP "${CMAKE_SOURCE_DIR}/external/hdf5-${hdf5_version}.zip")
set(CATCH2_ZIP "${CMAKE_SOURCE_DIR}/external/Catch2-${catch2_version}.zip")

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
add_custom_target(pugixml_lib ALL DEPENDS ${PUGIXML_SRC_DIR}/.stamp)
add_custom_target(pybind11_lib ALL DEPENDS ${PYBIND11_SRC_DIR}/.stamp)

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
set(BUILD_OCTAVE ${ENABLE_OCTAVE})
set(BUILD_MEX_DOC ${ENABLE_MEX_DOC})
set(BUILD_PYTHON ${ENABLE_PYTHON})
set(BUILD_SHARED_LIB ${ENABLE_SHARED_LIB})
set(BUILD_CATCH2 ${ENABLE_TESTS})

if (NOT ENABLE_PYTHON)
    set(BUILD_MEX_DOC OFF)
endif()

if (MSVC)
    set(BUILD_PYTHON OFF)
    set(BUILD_MEX_DOC OFF)
    set(BUILD_OCTAVE OFF)
endif()

if (NOT BUILD_HDF5 AND NOT HDF5_PATH)
    set(BUILD_SHARED_LIB OFF)
endif()

# Build instructions for HDF5 library
if (NOT HDF5_PATH AND NOT BUILD_HDF5) # Detect HDF5 automatically
    find_package(HDF5 COMPONENTS C HL)
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

# Detect MATLAB
if(BUILD_MATLAB AND MATLAB_PATH)
    message(STATUS "Using user-specified MATLAB at: ${MATLAB_PATH}")
    set(MEX_EXECUTABLE "${MATLAB_PATH}/bin/mex")
    message(STATUS "Using MEX compiler at: ${MEX_EXECUTABLE}")
elseif(BUILD_MATLAB)
    find_package(Matlab COMPONENTS MEX_COMPILER)
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

# Add include directories common to all targets
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${ARMADILLO_INCLUDE_DIRS}
    ${PUGIXML_H}
    ${PYBIND11_H}
    ${HDF5_INCLUDE_DIRS}
)

# Locate all core source files
file(GLOB SRC_FILES "${CMAKE_SOURCE_DIR}/src/*.cpp")
file(GLOB SRC_AVX2 "${CMAKE_SOURCE_DIR}/src/*_avx2.cpp")

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
            list(REMOVE_ITEM SRC_FILES "${CMAKE_SOURCE_DIR}/src/baseband_freq_response_avx2.cpp")
            list(REMOVE_ITEM SRC_AVX2 "${CMAKE_SOURCE_DIR}/src/baseband_freq_response_avx2.cpp")
        else()
            set_source_files_properties(${src} PROPERTIES COMPILE_FLAGS "-mavx2 -mfma")
        endif()
    endforeach()
    message(STATUS "AVX2 acceleration enabled")
else()
    foreach(src ${SRC_AVX2})
        # This does not work. Please fix
        list(REMOVE_ITEM SRC_FILES ${src})
        list(REMOVE_ITEM SRC_AVX2 ${src})
        #message(STATUS ${src})
    endforeach()
    set(BUILD_WITH_AVX2 0)
    message(STATUS "AVX2 acceleration disabled")
endif()

# Static library
add_library(quadriga_static STATIC ${SRC_FILES} ${SRC_AVX2})
set_target_properties(quadriga_static PROPERTIES OUTPUT_NAME quadriga)
add_dependencies(quadriga_static pugixml_lib pybind11_lib)
target_compile_definitions(quadriga_static PRIVATE BUILD_WITH_AVX2=${BUILD_WITH_AVX2})

if (BUILD_HDF5 AND NOT HDF5_PATH)
    add_dependencies(quadriga_static hdf5_dep)
endif()
if (NOT ARMADILLO_FOUND)
    add_dependencies(quadriga_static armadillo_lib)
endif()

# Shared library
if (BUILD_SHARED_LIB)
    add_library(quadriga_shared SHARED ${SRC_FILES} ${SRC_AVX2})
    set_target_properties(quadriga_shared PROPERTIES OUTPUT_NAME quadriga)
    add_dependencies(quadriga_shared pugixml_lib pybind11_lib)
    target_compile_definitions(quadriga_shared PRIVATE BUILD_WITH_AVX2=${BUILD_WITH_AVX2})
    if (NOT ARMADILLO_FOUND)
        add_dependencies(quadriga_shared armadillo_lib)
    endif()
endif()

# --- MATLAB API ---
if(BUILD_MATLAB)
    file(MAKE_DIRECTORY ${MEX_OUTPUT_DIR})

    if (HDF5_FOUND)
        set(HDF5_MEX_LINK_LIBS ${HDF5_LIBRARIES})
    else()
        set(HDF5_MEX_LINK_LIBS "")
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
                        ${mex_src} $<TARGET_FILE:quadriga_static> ${HDF5_LIB} shlwapi.lib 
                        -I${CMAKE_SOURCE_DIR}/include -I${CMAKE_SOURCE_DIR}/src -I${ARMADILLO_INCLUDE_DIRS}
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
                        ${mex_src} $<TARGET_FILE:quadriga_static> ${HDF5_LIB} ${HDF5_MEX_LINK_LIBS}
                        -I${CMAKE_SOURCE_DIR}/include -I${CMAKE_SOURCE_DIR}/src -I${ARMADILLO_INCLUDE_DIRS}
                        -lgomp
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
                    -I${CMAKE_SOURCE_DIR}/include -I${CMAKE_SOURCE_DIR}/src -I${ARMADILLO_INCLUDE_DIRS} -s
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

    set(PYTHON_SOURCES 
        api_python/python_main.cpp
        api_python/python_arrayant.cpp
        api_python/python_channel.cpp
        api_python/python_RTtools.cpp
        api_python/python_tools.cpp
    )
    add_library(quadriga_py MODULE ${PYTHON_SOURCES})
    set_target_properties(quadriga_py PROPERTIES
        PREFIX ""
        OUTPUT_NAME "quadriga_lib"
        SUFFIX "${PYTHON_EXTENSION_SUFFIX}"
        POSITION_INDEPENDENT_CODE ON
    )
    target_include_directories(quadriga_py PRIVATE ${Python3_INCLUDE_DIRS})
    target_link_libraries(quadriga_py PRIVATE quadriga_static ${Python3_LIBRARIES} HDF5::HDF5)
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

    # Watch test files (but don't compile them)
    file(GLOB TEST_HEADERS CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/catch2_tests/*.cpp
    )
    
    add_executable(test_bin ${CATCH2_MAIN_SRC})
    add_dependencies(test_bin catch2_dep)
    target_include_directories(test_bin PRIVATE ${CATCH2_INCLUDE_DIR})

    if (BUILD_HDF5)
        target_link_libraries(test_bin PRIVATE quadriga_static ${CATCH2_LIB} ${HDF5_LIB})
    else()
        target_link_libraries(test_bin PRIVATE quadriga_static ${CATCH2_LIB} HDF5::HDF5)
    endif()
    
    if(MSVC)
        target_link_libraries(test_bin PRIVATE shlwapi.lib)
        target_compile_options(test_bin PRIVATE /EHsc /Zc:__cplusplus /nologo /openmp /MP)
    else()
        target_compile_options(test_bin PRIVATE -O3 -fPIC -fopenmp -w)
    endif()

    
endif()

# --- Binaries ---
if (ENABLE_BIN)
    file(GLOB SRC_BIN "${CMAKE_SOURCE_DIR}/src/bin/*.cpp")
    foreach(source IN LISTS SRC_BIN)
        get_filename_component(exeName ${source} NAME_WE)

        add_executable(${exeName} ${source})
        add_dependencies(${exeName} quadriga_static)
        
        if (BUILD_HDF5)
            target_link_libraries(${exeName} PRIVATE quadriga_static ${HDF5_LIB})
        else()
            target_link_libraries(${exeName} PRIVATE quadriga_static HDF5::HDF5)
        endif()
    endforeach()
endif()

# Install options
if(CMAKE_INSTALL_PREFIX STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}")
    message(STATUS "Using source dir as install dir")
else()    
    message(STATUS "Installing to: ${CMAKE_INSTALL_PREFIX}")
    install(DIRECTORY include/ DESTINATION include)
    if (NOT ARMADILLO_FOUND)
        install(DIRECTORY ${ARMADILLO_INCLUDE_DIRS} DESTINATION .)
    endif()
    if(NOT CMAKE_INSTALL_PREFIX STREQUAL "/usr/local")
        install(DIRECTORY html_docu/ DESTINATION html_docu)
        install(FILES LICENSE DESTINATION .)
    endif()
endif()

install(TARGETS quadriga_static
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)

if (BUILD_SHARED_LIB)
    install(TARGETS quadriga_shared
            ARCHIVE DESTINATION lib
            LIBRARY DESTINATION lib
            RUNTIME DESTINATION bin) 
endif()

if(NOT CMAKE_INSTALL_PREFIX STREQUAL "/usr/local")
    if(BUILD_MATLAB OR BUILD_OCTAVE)
        install(DIRECTORY ${MEX_OUTPUT_DIR} DESTINATION .)
    endif()
else()
    if(BUILD_MATLAB OR BUILD_OCTAVE)
        install(DIRECTORY ${MEX_OUTPUT_DIR} DESTINATION MATLAB_quadriga-lib)
    endif()
endif()

if (BUILD_PYTHON)
    install(TARGETS quadriga_py
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)
endif()

if (BUILD_HDF5 OR HDF5_PATH)
    install(FILES ${HDF5_LIB} DESTINATION lib)
endif()