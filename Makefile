# This Makefile is for Windows / MSVC environments

# Steps for compiling Quadriga-Lib (Linux):
# - Get Build Tools for Visual Studio
# - Compile HDF5 library by "nmake hdf5lib"
# - Set MATLAB path below
# - Run "nmake"

# Set path to your MATLAB installation (optional):
# Leave this empty if you don't want to use MATLAB
MATLAB_PATH = C:\Program Files\MATLAB\R2022b

# External libraries
# External libraries are located in the 'external' folder. Set the version numbers here.
# You need to compile the HDF5 and Catch2 libraries (e.g. using 'make hdf5lib' or 'make catch2lib' )
armadillo_version = 14.2.2
hdf5_version      = 1.14.2
catch2_version    = 3.4.0
pugixml_version   = 1.13

# nmake cheat sheet:
#	$@    Current target's full name (path, base name, extension)
#	$*    Current target's path and base name minus file extension.
#	$**   All dependents of the current target
#	$(@B) Current target's base name (no path, no extension)
#	$(@F) Current target's base name + estension (no path)

# Compilers
CC    = cl
MEX   = "$(MATLAB_PATH)\bin\win64\mex.exe"

# Header files
ARMA_H      = external\armadillo-$(armadillo_version)\include
PUGIXML_H   = external\pugixml-$(pugixml_version)\src
CATCH2      = external\Catch2-$(catch2_version)-win64
HDF5        = external\hdf5-$(hdf5_version)-win64

# Configurations
CCFLAGS     = /EHsc /std:c++17 /Zc:__cplusplus /nologo /MD #/Wall 
MEXFLAGS    = /std:c++17 /MD

all:   dirs \
       +quadriga_lib\arrayant_calc_directivity.mexw64 \
       +quadriga_lib\arrayant_combine_pattern.mexw64 \
	   +quadriga_lib\arrayant_generate.mexw64 \
	   +quadriga_lib\arrayant_interpolate.mexw64 \
	   +quadriga_lib\arrayant_qdant_read.mexw64 \
	   +quadriga_lib\arrayant_qdant_write.mexw64 \
	   +quadriga_lib\arrayant_rotate_pattern.mexw64 \
	   +quadriga_lib/baseband_freq_response.mexw64 \
	   +quadriga_lib\calc_diffraction_gain.mexw64 \
	   +quadriga_lib\calc_rotation_matrix.mexw64 \
	   +quadriga_lib\cart2geo.mexw64 \
	   +quadriga_lib\generate_diffraction_paths.mexw64 \
	   +quadriga_lib\geo2cart.mexw64 \
	   +quadriga_lib\get_channels_planar.mexw64 \
	   +quadriga_lib\get_channels_spherical.mexw64 \
	   +quadriga_lib\hdf5_create_file.mexw64 \
	   +quadriga_lib\hdf5_read_channel.mexw64 \
	   +quadriga_lib\hdf5_read_dset.mexw64 \
	   +quadriga_lib\hdf5_read_dset_names.mexw64 \
	   +quadriga_lib\hdf5_read_layout.mexw64 \
	   +quadriga_lib\hdf5_reshape_layout.mexw64 \
	   +quadriga_lib\hdf5_write_channel.mexw64 \
	   +quadriga_lib\hdf5_write_dset.mexw64 \
	   +quadriga_lib\hdf5_version.mexw64 \
	   +quadriga_lib\icosphere.mexw64 \
	   +quadriga_lib\interp.mexw64 \
	   +quadriga_lib\obj_file_read.mexw64 \
	   +quadriga_lib\point_cloud_aabb.mexw64 \
	   +quadriga_lib\point_cloud_segmentation.mexw64 \
	   +quadriga_lib\ray_mesh_interact.mexw64 \
	   +quadriga_lib\ray_point_intersect.mexw64 \
	   +quadriga_lib\ray_triangle_intersect.mexw64 \
	   +quadriga_lib\subdivide_triangles.mexw64 \
	   +quadriga_lib\triangle_mesh_aabb.mexw64 \
	   +quadriga_lib\triangle_mesh_segmentation.mexw64 \
	   +quadriga_lib\version.mexw64

dirs:
	- mkdir build
	- mkdir lib
	- mkdir +quadriga_lib

test:   tests\test.exe
	tests\test.exe

tests\test.exe:   tests\quadriga_lib_catch2_tests.cpp   lib\quadriga_lib.lib
	$(CC) $(CCFLAGS) /Fetests\test.exe $** /Iinclude /I$(ARMA_H) /I$(CATCH2)\include /link $(CATCH2)\lib\Catch2.lib
	del quadriga_lib_catch2_tests.obj

# Individual Library files
build\calc_diffraction_gain.obj:   src\calc_diffraction_gain.cpp   include\quadriga_tools.hpp
	$(CC) $(CCFLAGS) /c src\$(@B).cpp /Fo$@ /Iinclude /I$(ARMA_H)

build\qd_arrayant.obj:   src\qd_arrayant.cpp   include\quadriga_arrayant.hpp
	$(CC) $(CCFLAGS) /openmp /c src\$(@B).cpp /Fo$@ /Iinclude /I$(ARMA_H)

build\qd_arrayant_qdant.obj:   src\qd_arrayant_qdant.cpp   src\qd_arrayant_functions.hpp
	$(CC) $(CCFLAGS) /c src\$(@B).cpp /Fo$@ /Iinclude /I$(PUGIXML_H) /I$(ARMA_H)

build\qd_arrayant_interpolate.obj:   src\qd_arrayant_interpolate.cpp   src\qd_arrayant_functions.hpp
	$(CC) $(CCFLAGS) /openmp /c src\$(@B).cpp /Fo$@ /Iinclude /I$(ARMA_H)

build\baseband_freq_response.obj:   src\baseband_freq_response.cpp   include\quadriga_channel.hpp
	$(CC) /openmp $(CCFLAGS) /c src\$(@B).cpp /Fo$@ /Iinclude /I$(ARMA_H)

build\qd_channel.obj:   src\qd_channel.cpp   include\quadriga_channel.hpp
	$(CC) $(CCFLAGS) /c src\$(@B).cpp /Fo$@ /Iinclude /I$(ARMA_H) /I$(HDF5)\include

build\quadriga_tools.obj:   src\quadriga_tools.cpp   include\quadriga_tools.hpp
	$(CC) $(CCFLAGS) /c src\$(@B).cpp /Fo$@ /Iinclude /I$(ARMA_H)

build\quadriga_lib.obj:   src\quadriga_lib.cpp   include\quadriga_lib.hpp
	$(CC) $(CCFLAGS) /c src\$(@B).cpp /Fo$@ /Iinclude /I$(ARMA_H)

build\ray_mesh_interact.obj:   src\ray_mesh_interact.cpp   include\quadriga_tools.hpp
	$(CC) $(CCFLAGS) /openmp /c src\$(@B).cpp /Fo$@ /Iinclude /I$(ARMA_H)

build\ray_point_intersect.obj:   src\ray_point_intersect.cpp   include\quadriga_tools.hpp
	$(CC) /openmp $(CCFLAGS) /c src\$(@B).cpp /Fo$@ /Iinclude /I$(ARMA_H)

build\ray_triangle_intersect.obj:   src\ray_triangle_intersect.cpp   include\quadriga_tools.hpp
	$(CC)  /openmp $(CCFLAGS) /c src\$(@B).cpp /Fo$@ /Iinclude /I$(ARMA_H)

# AVX2 library files
build\quadriga_lib_test_avx.obj:   src\quadriga_lib_test_avx.cpp   src\quadriga_lib_test_avx.hpp
	$(CC) /arch:AVX2 $(CCFLAGS) /c src\$(@B).cpp /Fo$@ /Isrc

build\ray_triangle_intersect_avx2.obj:   src\ray_triangle_intersect_avx2.cpp   src\ray_triangle_intersect_avx2.hpp
	$(CC) /arch:AVX2 /openmp $(CCFLAGS) /c src\$(@B).cpp /Fo$@ /Isrc /I$(ARMA_H)

build\ray_point_intersect_avx2.obj:   src\ray_point_intersect_avx2.cpp   src\ray_point_intersect_avx2.hpp
	$(CC) /arch:AVX2 /openmp $(CCFLAGS) /c src\$(@B).cpp /Fo$@ /Isrc

# Archive file for static linking
build\libhdf5.lib:
   lib /OUT:$@ $(HDF5)\lib\libhdf5.lib Shlwapi.lib

lib\quadriga_lib.lib:   build\quadriga_lib.obj   build\quadriga_lib_test_avx.obj   build\qd_arrayant.obj   build\qd_channel.obj   \
                        build\quadriga_tools.obj   build\qd_arrayant_interpolate.obj   build\qd_arrayant_qdant.obj   \
						build\ray_mesh_interact.obj   build\ray_triangle_intersect.obj   build\calc_diffraction_gain.obj \
						build\libhdf5.lib   build\ray_point_intersect.obj   build\baseband_freq_response.obj   \
						build\ray_triangle_intersect_avx2.obj   build\ray_point_intersect_avx2.obj
    lib /OUT:$@ $**

# Dependencies
dep_quadriga_tools = build\quadriga_tools.obj   build\ray_triangle_intersect.obj   build\ray_triangle_intersect_avx2.obj
dep_arrayant = build\qd_arrayant.obj   build\qd_arrayant_interpolate.obj   build\qd_arrayant_qdant.obj   $(dep_quadriga_tools)

# MEX MATLAB interface
+quadriga_lib\arrayant_calc_directivity.mexw64:   api_mex\arrayant_calc_directivity.cpp   $(dep_arrayant)
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\arrayant_combine_pattern.mexw64:   api_mex\arrayant_combine_pattern.cpp   $(dep_arrayant)
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\arrayant_generate.mexw64:   api_mex\arrayant_generate.cpp   $(dep_arrayant)
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\arrayant_interpolate.mexw64:   api_mex\arrayant_interpolate.cpp   $(dep_arrayant)
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\arrayant_qdant_read.mexw64:   api_mex\arrayant_qdant_read.cpp   $(dep_arrayant)
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\arrayant_qdant_write.mexw64:   api_mex\arrayant_qdant_write.cpp   $(dep_arrayant)
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\baseband_freq_response.mexw64:   api_mex\baseband_freq_response.cpp   build\baseband_freq_response.obj
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\calc_diffraction_gain.mexw64:   api_mex\calc_diffraction_gain.cpp   build\calc_diffraction_gain.obj   $(dep_quadriga_tools)  build\ray_mesh_interact.obj
   $(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\arrayant_rotate_pattern.mexw64:   api_mex\arrayant_rotate_pattern.cpp   $(dep_arrayant)
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\calc_rotation_matrix.mexw64:   api_mex\calc_rotation_matrix.cpp   $(dep_quadriga_tools)
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\cart2geo.mexw64:   api_mex\cart2geo.cpp   $(dep_quadriga_tools)
 	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\generate_diffraction_paths.mexw64:   api_mex\generate_diffraction_paths.cpp   $(dep_quadriga_tools)
 	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\geo2cart.mexw64:   api_mex\geo2cart.cpp   $(dep_quadriga_tools)
 	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\get_channels_planar.mexw64:   api_mex\get_channels_planar.cpp   $(dep_arrayant)
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\get_channels_spherical.mexw64:   api_mex\get_channels_spherical.cpp   $(dep_arrayant)
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\hdf5_create_file.mexw64:   api_mex\hdf5_create_file.cpp   build\qd_channel.obj   $(dep_quadriga_tools)   build\libhdf5.lib
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\hdf5_read_channel.mexw64:   api_mex\hdf5_read_channel.cpp   build\qd_channel.obj   $(dep_quadriga_tools)   build\libhdf5.lib
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\hdf5_read_dset.mexw64:   api_mex\hdf5_read_dset.cpp   build\qd_channel.obj   $(dep_quadriga_tools)   build\libhdf5.lib
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\hdf5_read_dset_names.mexw64:   api_mex\hdf5_read_dset_names.cpp   build\qd_channel.obj   $(dep_quadriga_tools)   build\libhdf5.lib
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\hdf5_read_layout.mexw64:   api_mex\hdf5_read_layout.cpp   build\qd_channel.obj   $(dep_quadriga_tools)   build\libhdf5.lib
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\hdf5_reshape_layout.mexw64:   api_mex\hdf5_reshape_layout.cpp   build\qd_channel.obj   $(dep_quadriga_tools)   build\libhdf5.lib
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\hdf5_write_channel.mexw64:   api_mex\hdf5_write_channel.cpp   build\qd_channel.obj   $(dep_quadriga_tools)   build\libhdf5.lib
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\hdf5_write_dset.mexw64:   api_mex\hdf5_write_dset.cpp   build\qd_channel.obj   $(dep_quadriga_tools)   build\libhdf5.lib
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\hdf5_version.mexw64:   api_mex\hdf5_version.cpp   build\qd_channel.obj   $(dep_quadriga_tools)   build\libhdf5.lib
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\icosphere.mexw64:   api_mex\icosphere.cpp   $(dep_quadriga_tools)
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\interp.mexw64:   api_mex\interp.cpp   $(dep_quadriga_tools)
 	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\obj_file_read.mexw64:   api_mex\obj_file_read.cpp   $(dep_quadriga_tools)
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\point_cloud_aabb.mexw64:   api_mex\point_cloud_aabb.cpp   $(dep_quadriga_tools)
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\point_cloud_segmentation.mexw64:   api_mex\point_cloud_segmentation.cpp   $(dep_quadriga_tools)
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\ray_mesh_interact.mexw64:   api_mex\ray_mesh_interact.cpp   build\ray_mesh_interact.obj
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\ray_point_intersect.mexw64:   api_mex\ray_point_intersect.cpp   build\ray_point_intersect.obj   $(dep_quadriga_tools)   build\ray_point_intersect_avx2.obj
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\ray_triangle_intersect.mexw64:   api_mex\ray_triangle_intersect.cpp   build\ray_triangle_intersect.obj   build\ray_triangle_intersect_avx2.obj
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\subdivide_triangles.mexw64:   api_mex\subdivide_triangles.cpp   $(dep_quadriga_tools)
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\triangle_mesh_aabb.mexw64:   api_mex\triangle_mesh_aabb.cpp   $(dep_quadriga_tools)
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\triangle_mesh_segmentation.mexw64:   api_mex\triangle_mesh_segmentation.cpp   $(dep_quadriga_tools)
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\version.mexw64:   api_mex\version.cpp   build\quadriga_lib.obj   build\quadriga_lib_test_avx.obj
 	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

# Maintainance section
external:   armadillo-lib   pugixml-lib   hdf5-lib   catch2-lib   moxunit-lib

armadillo-lib:
	- rmdir /s /q external\armadillo-$(armadillo_version)
	tar -xf external/armadillo-$(armadillo_version).zip
	move armadillo-$(armadillo_version) external

pugixml-lib:
	- rmdir /s /q external\pugixml-$(pugixml_version)
	tar -xf external/pugixml-$(pugixml_version).zip
	move pugixml-$(pugixml_version) external

hdf5-lib:
	- rmdir /s /q external\build
	- rmdir /s /q external\hdf5-$(hdf5_version)
	- rmdir /s /q external\hdf5-$(hdf5_version)-win64
	tar -xf external/hdf5-$(hdf5_version).zip
	move hdf5-$(hdf5_version) external
	mkdir external\build
	cmake -S external\hdf5-$(hdf5_version) -B external\build -D CMAKE_INSTALL_PREFIX=external\hdf5-$(hdf5_version)-win64 -D BUILD_SHARED_LIBS=OFF -D HDF5_ENABLE_Z_LIB_SUPPORT=OFF -D BUILD_TESTING=OFF
	cmake --build external\build --config Release --target install
	rmdir /s /q external\build
	rmdir /s /q external\hdf5-$(hdf5_version)

catch2-lib:
	- rmdir /s /q external\build
	- rmdir /s /q external\Catch2-$(catch2_version)
	- rmdir /s /q external\Catch2-$(catch2_version)-win64
	tar -xf external/Catch2-$(catch2_version).zip
	move Catch2-$(catch2_version) external
	mkdir external\build
	cmake -S external\Catch2-$(catch2_version) -B external\build
	cmake --build external\build --config Release --target package
	move external\build\_CPack_Packages\win64\NSIS\Catch2-$(catch2_version)-win64 external
	rmdir /s /q external\build
	rmdir /s /q external\Catch2-$(catch2_version)

moxunit-lib:
	- rmdir /s /q external\MOxUnit-master
	tar -xf external/MOxUnit.zip
	move MOxUnit-master external

clean:
	- rmdir /s /q external\build
	- rmdir /s /q external\Catch2-$(catch2_version)
	- rmdir /s /q external\hdf5-$(hdf5_version)
	- rmdir /s /q build
	- del "+quadriga_lib"\*.manifest
	- del "+quadriga_lib"\*.exp
	- del "+quadriga_lib"\*.lib
	- del tests\test.exe
	
	
tidy:   clean
	- rmdir /s /q external\Catch2-$(catch2_version)-win64
	- rmdir /s /q external\hdf5-$(hdf5_version)-win64
	- rmdir /s /q external\armadillo-$(armadillo_version)
	- rmdir /s /q external\pugixml-$(pugixml_version)
	- rmdir /s /q external\MOxUnit-master
	- rmdir /s /q +quadriga_lib
	- rmdir /s /q lib

build\quadriga-lib-version.exe:   src/version.cpp   lib/quadriga_lib.lib
	$(CC) $(CCFLAGS) /Febuild\quadriga-lib-version.exe $** /Iinclude /I$(ARMA_H)

releasex:   all   build\quadriga-lib-version.exe
	@setlocal
	@for /f %%i in ('build\quadriga-lib-version.exe') do \
		tar -c -f release\quadrigalib-v%%i-Win64.zip lib\*.lib +quadriga_lib\*.m +quadriga_lib\*.mexw64 include
	@endlocal



