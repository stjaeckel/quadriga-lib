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
armadillo_version = 12.6.3
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

all:   +quadriga_lib\arrayant_calc_directivity.mexw64 \
       +quadriga_lib\arrayant_combine_pattern.mexw64 \
	   +quadriga_lib\arrayant_generate.mexw64 \
	   +quadriga_lib\arrayant_interpolate.mexw64 \
	   +quadriga_lib\arrayant_qdant_read.mexw64 \
	   +quadriga_lib\arrayant_qdant_write.mexw64 \
	   +quadriga_lib\arrayant_rotate_pattern.mexw64 \
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
	   +quadriga_lib\icosphere.mexw64 \
	   +quadriga_lib\interp.mexw64 \
	   +quadriga_lib\obj_file_read.mexw64 \
	   +quadriga_lib\ray_mesh_interact.mexw64 \
	   +quadriga_lib\ray_triangle_intersect.mexw64 \
	   +quadriga_lib\subdivide_triangles.mexw64 \
	   +quadriga_lib\version.mexw64

test:   tests\test.exe
	tests\test.exe

tests\test.exe:   tests\quadriga_lib_catch2_tests.cpp   lib\quadriga_lib.lib
	$(CC) $(CCFLAGS) /Fetests\test.exe $** /Iinclude /I$(ARMA_H) /I$(CATCH2)\include /link $(CATCH2)\lib\Catch2.lib
	del quadriga_lib_catch2_tests.obj

# Individual Library files
build\qd_arrayant.obj:   src\qd_arrayant.cpp   include\quadriga_arrayant.hpp
	$(CC) $(CCFLAGS) /c src\$(@B).cpp /Fo$@ /Iinclude /I$(ARMA_H)

build\qd_arrayant_qdant.obj:   src\qd_arrayant_qdant.cpp   src\qd_arrayant_qdant.hpp
	$(CC) $(CCFLAGS) /c src\$(@B).cpp /Fo$@ /Iinclude /I$(PUGIXML_H) /I$(ARMA_H)

build\qd_arrayant_interpolate.obj:   src\qd_arrayant_interpolate.cpp   src\qd_arrayant_interpolate.hpp
	$(CC) $(CCFLAGS) /openmp /c src\$(@B).cpp /Fo$@ /Iinclude /I$(ARMA_H)

build\qd_channel.obj:   src\qd_channel.cpp   include\quadriga_channel.hpp
	$(CC) $(CCFLAGS) /c src\$(@B).cpp /Fo$@ /Iinclude /I$(ARMA_H) /I$(HDF5)\include

build\quadriga_tools.obj:   src\quadriga_tools.cpp   include\quadriga_tools.hpp
	$(CC) $(CCFLAGS) /c src\$(@B).cpp /Fo$@ /Iinclude /I$(ARMA_H)

build\quadriga_lib.obj:   src\quadriga_lib.cpp   include\quadriga_lib.hpp
	$(CC) /arch:AVX2 $(CCFLAGS) /c src\$(@B).cpp /Fo$@ /Iinclude /I$(ARMA_H)

build\ray_mesh_interact.obj:   src\ray_mesh_interact.cpp   include\quadriga_tools.hpp
	$(CC) $(CCFLAGS) /openmp /c src\$(@B).cpp /Fo$@ /Iinclude /I$(ARMA_H)

build\ray_triangle_intersect.obj:   src\ray_triangle_intersect.cpp   include\quadriga_tools.hpp
	$(CC) /arch:AVX2 /openmp $(CCFLAGS) /c src\$(@B).cpp /Fo$@ /Iinclude /I$(ARMA_H)

# Archive file for static linking
build\libhdf5.lib:
   lib /OUT:$@ $(HDF5)\lib\libhdf5.lib Shlwapi.lib

lib\quadriga_lib.lib:   build\quadriga_lib.obj   build\qd_arrayant.obj   build\qd_channel.obj   \
                        build\quadriga_tools.obj   build\qd_arrayant_interpolate.obj   build\qd_arrayant_qdant.obj   \
						build\ray_mesh_interact.obj   build\ray_triangle_intersect.obj   build\libhdf5.lib
    lib /OUT:$@ $**

# MEX MATLAB interface
+quadriga_lib\arrayant_calc_directivity.mexw64:   mex\arrayant_calc_directivity.cpp   build\qd_arrayant.obj   build\qd_arrayant_interpolate.obj   build\qd_arrayant_qdant.obj   build\quadriga_tools.obj
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\arrayant_combine_pattern.mexw64:   mex\arrayant_combine_pattern.cpp   build\qd_arrayant.obj   build\qd_arrayant_interpolate.obj   build\qd_arrayant_qdant.obj   build\quadriga_tools.obj
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\arrayant_generate.mexw64:   mex\arrayant_generate.cpp   build\qd_arrayant.obj   build\qd_arrayant_interpolate.obj   build\qd_arrayant_qdant.obj   build\quadriga_tools.obj
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\arrayant_interpolate.mexw64:   mex\arrayant_interpolate.cpp   build\qd_arrayant.obj   build\qd_arrayant_interpolate.obj   build\qd_arrayant_qdant.obj   build\quadriga_tools.obj
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\arrayant_qdant_read.mexw64:   mex\arrayant_qdant_read.cpp   build\qd_arrayant.obj   build\qd_arrayant_interpolate.obj   build\qd_arrayant_qdant.obj   build\quadriga_tools.obj
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\arrayant_qdant_write.mexw64:   mex\arrayant_qdant_write.cpp   build\qd_arrayant.obj   build\qd_arrayant_interpolate.obj   build\qd_arrayant_qdant.obj   build\quadriga_tools.obj
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\arrayant_rotate_pattern.mexw64:   mex\arrayant_rotate_pattern.cpp   build\qd_arrayant.obj   build\qd_arrayant_interpolate.obj   build\qd_arrayant_qdant.obj   build\quadriga_tools.obj
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\calc_rotation_matrix.mexw64:   mex\calc_rotation_matrix.cpp   build\quadriga_tools.obj
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\cart2geo.mexw64:   mex\cart2geo.cpp   build\quadriga_tools.obj
 	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\generate_diffraction_paths.mexw64:   mex\generate_diffraction_paths.cpp   build\quadriga_tools.obj
 	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\geo2cart.mexw64:   mex\geo2cart.cpp   build\quadriga_tools.obj
 	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\get_channels_planar.mexw64:   mex\get_channels_planar.cpp   build\qd_arrayant.obj   build\qd_arrayant_interpolate.obj   build\qd_arrayant_qdant.obj   build\quadriga_tools.obj
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\get_channels_spherical.mexw64:   mex\get_channels_spherical.cpp   build\qd_arrayant.obj   build\qd_arrayant_interpolate.obj   build\qd_arrayant_qdant.obj   build\quadriga_tools.obj
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\hdf5_create_file.mexw64:   mex\hdf5_create_file.cpp   build\qd_channel.obj   build\libhdf5.lib
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\hdf5_read_channel.mexw64:   mex\hdf5_read_channel.cpp   build\qd_channel.obj   build\libhdf5.lib
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\hdf5_read_dset.mexw64:   mex\hdf5_read_dset.cpp   build\qd_channel.obj   build\libhdf5.lib
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\hdf5_read_dset_names.mexw64:   mex\hdf5_read_dset_names.cpp   build\qd_channel.obj   build\libhdf5.lib
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\hdf5_read_layout.mexw64:   mex\hdf5_read_layout.cpp   build\qd_channel.obj   build\libhdf5.lib
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\hdf5_reshape_layout.mexw64:   mex\hdf5_reshape_layout.cpp   build\qd_channel.obj   build\libhdf5.lib
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\hdf5_write_channel.mexw64:   mex\hdf5_write_channel.cpp   build\qd_channel.obj   build\libhdf5.lib
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\hdf5_write_dset.mexw64:   mex\hdf5_write_dset.cpp   build\qd_channel.obj   build\libhdf5.lib
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\icosphere.mexw64:   mex\icosphere.cpp   build\quadriga_tools.obj
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\interp.mexw64:   mex\interp.cpp   build\quadriga_tools.obj
 	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\obj_file_read.mexw64:   mex\obj_file_read.cpp   build\quadriga_tools.obj
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\ray_mesh_interact.mexw64:   mex\ray_mesh_interact.cpp   build\ray_mesh_interact.obj
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\ray_triangle_intersect.mexw64:   mex\ray_triangle_intersect.cpp   build\ray_triangle_intersect.obj
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\subdivide_triangles.mexw64:   mex\subdivide_triangles.cpp   build\quadriga_tools.obj
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib\version.mexw64:   mex\version.cpp   build\quadriga_lib.obj
 	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

# Maintainance section
hdf5lib:
	- rmdir /s /q external\build
	- rmdir /s /q external\hdf5-$(hdf5_version)
	- rmdir /s /q external\hdf5-$(hdf5_version)-win64
	tar -xf external/hdf5-$(hdf5_version).zip
	move hdf5-$(hdf5_version) external
	mkdir external\build
	cmake -S external\hdf5-$(hdf5_version) -B external\build -D CMAKE_INSTALL_PREFIX=external\hdf5-$(hdf5_version)-win64 -D BUILD_SHARED_LIBS=OFF -D HDF5_ENABLE_Z_LIB_SUPPORT=OFF
	cmake --build external\build --config Release --target install
	rmdir /s /q external\build
	rmdir /s /q external\hdf5-$(hdf5_version)

catch2lib:
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

clean:
	del build\*.obj
	del build\*.lib
	del "+quadriga_lib"\*.manifest
	del "+quadriga_lib"\*.exp
	del "+quadriga_lib"\*.lib
	del "+quadriga_lib"\*.mexw64

tidy:   clean
	- rmdir /s /q external\Catch2-$(catch2_version)-win64
	- rmdir /s /q external\hdf5-$(hdf5_version)-win64

build\quadriga-lib-version.exe:   src/version.cpp   lib/quadriga_lib.lib
	$(CC) $(CCFLAGS) /Febuild\quadriga-lib-version.exe $** /Iinclude /I$(ARMA_H)

releasex:   all   build\quadriga-lib-version.exe
	@setlocal
	@for /f %%i in ('build\quadriga-lib-version.exe') do \
		tar -c -f release\quadrigalib-v%%i-Win64.zip lib\*.lib +quadriga_lib\*.m +quadriga_lib\*.mexw64 include
	@endlocal



