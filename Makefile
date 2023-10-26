# This Makefile is for Windows / MSVC environments
# Cheat sheet:
#	$@    Current target's full name (path, base name, extension)
#	$*    Current target's path and base name minus file extension.
#	$**   All dependents of the current target
#	$(@B) Current target's base name (no path, no extension)
#	$(@F) Current target's base name + estension (no path)

# Compilers
CC    = cl
MEX   = "C:\Program Files\MATLAB\R2022b\bin\win64\mex.exe"

# External libraries
hdf5version    = 1.14.2
catch2version  = 3.4.0

ARMA_H      = external\armadillo-12.6.3\include
PUGIXML_H   = external\pugixml-1.13\src
CATCH2      = external\Catch2-$(catch2version)-win64
HDF5        = external\hdf5-$(hdf5version)-win64

# Configurations
CCFLAGS     = /EHsc /std:c++17 /Zc:__cplusplus /nologo /MD #/Wall 
MEXFLAGS    = /std:c++17 /MD

all:   +quadriga_lib/calc_rotation_matrix.mexw64   +quadriga_lib/cart2geo.mexw64   +quadriga_lib/geo2cart.mexw64   \
       +quadriga_lib/arrayant_interpolate.mexw64   +quadriga_lib/arrayant_qdant_read.mexw64      +quadriga_lib/arrayant_qdant_write.mexw64   \
	   +quadriga_lib/version.mexw64   +quadriga_lib/arrayant_combine_pattern.mexw64   +quadriga_lib/interp.mexw64   \
	   +quadriga_lib/arrayant_generate.mexw64   +quadriga_lib/arrayant_calc_directivity.mexw64   +quadriga_lib/arrayant_rotate_pattern.mexw64 \
	   +quadriga_lib/get_channels_spherical.mexw64   +quadriga_lib/get_channels_planar.mexw64 \
	   +quadriga_lib/hdf5_create_file.mexw64   +quadriga_lib/hdf5_read_channel.mexw64   +quadriga_lib/hdf5_read_dset.mexw64   +quadriga_lib/hdf5_read_dset_names.mexw64   \
	   +quadriga_lib/hdf5_read_layout.mexw64   +quadriga_lib/hdf5_reshape_layout.mexw64   +quadriga_lib/hdf5_write_channel.mexw64   +quadriga_lib/hdf5_write_dset.mexw64   \
	   +quadriga_lib/icosphere.mexw64   +quadriga_lib/subdivide_triangles.mexw64   +quadriga_lib/obj_file_read.mexw64

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
	$(CC) $(CCFLAGS) /c src\$(@B).cpp /Fo$@ /Iinclude /I$(ARMA_H)

# Archive file for static linking
lib\quadriga_lib.lib:   build\quadriga_lib.obj   build\qd_arrayant.obj   build\qd_channel.obj   \
                        build\quadriga_tools.obj   build\qd_arrayant_interpolate.obj   build\qd_arrayant_qdant.obj
	lib /OUT:temp_quadriga_lib.lib $**
    lib /OUT:$@ temp_quadriga_lib.lib $(HDF5)\lib\libhdf5.lib Shlwapi.lib
    del temp_quadriga_lib.lib

# MEX MATLAB interface
+quadriga_lib/arrayant_combine_pattern.mexw64:   mex\arrayant_combine_pattern.cpp   lib\quadriga_lib.lib
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib/arrayant_interpolate.mexw64:   mex\arrayant_interpolate.cpp   lib\quadriga_lib.lib
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib/arrayant_qdant_read.mexw64:   mex\arrayant_qdant_read.cpp   lib\quadriga_lib.lib
 	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib/arrayant_qdant_write.mexw64:   mex\arrayant_qdant_write.cpp   lib\quadriga_lib.lib
 	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib/calc_rotation_matrix.mexw64:   mex\calc_rotation_matrix.cpp   lib\quadriga_lib.lib
 	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib/cart2geo.mexw64:   mex\cart2geo.cpp   lib\quadriga_lib.lib
 	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib/geo2cart.mexw64:   mex\geo2cart.cpp   lib\quadriga_lib.lib
 	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib/version.mexw64:   mex\version.cpp   lib\quadriga_lib.lib
 	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib/interp.mexw64:   mex\interp.cpp   lib\quadriga_lib.lib
 	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib/arrayant_generate.mexw64:   mex\arrayant_generate.cpp   lib\quadriga_lib.lib
 	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib/arrayant_calc_directivity.mexw64:   mex\arrayant_calc_directivity.cpp   lib\quadriga_lib.lib
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib/arrayant_rotate_pattern.mexw64:   mex\arrayant_rotate_pattern.cpp   lib\quadriga_lib.lib
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib/get_channels_spherical.mexw64:   mex\get_channels_spherical.cpp   lib\quadriga_lib.lib
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib/get_channels_planar.mexw64:   mex\get_channels_planar.cpp   lib\quadriga_lib.lib
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib/hdf5_create_file.mexw64:   mex\hdf5_create_file.cpp   lib\quadriga_lib.lib
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib/hdf5_read_channel.mexw64:   mex\hdf5_read_channel.cpp   lib\quadriga_lib.lib
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib/hdf5_read_dset.mexw64:   mex\hdf5_read_dset.cpp   lib\quadriga_lib.lib
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib/hdf5_read_dset_names.mexw64:   mex\hdf5_read_dset_names.cpp   lib\quadriga_lib.lib
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib/hdf5_read_layout.mexw64:   mex\hdf5_read_layout.cpp   lib\quadriga_lib.lib
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib/hdf5_reshape_layout.mexw64:   mex\hdf5_reshape_layout.cpp   lib\quadriga_lib.lib
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib/hdf5_write_channel.mexw64:   mex\hdf5_write_channel.cpp   lib\quadriga_lib.lib
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib/hdf5_write_dset.mexw64:   mex\hdf5_write_dset.cpp   lib\quadriga_lib.lib
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib/icosphere.mexw64:   mex\icosphere.cpp   lib\quadriga_lib.lib
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib/subdivide_triangles.mexw64:   mex\subdivide_triangles.cpp   lib\quadriga_lib.lib
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib/obj_file_read.mexw64:   mex\obj_file_read.cpp   lib\quadriga_lib.lib
	$(MEX) COMPFLAGS="$(MEXFLAGS)" -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

# Maintainance section
hdf5lib:
	- rmdir /s /q external\build
	- rmdir /s /q external\hdf5-$(hdf5version)
	- rmdir /s /q external\hdf5-$(hdf5version)-win64
	tar -xf external/hdf5-$(hdf5version).zip
	move hdf5-$(hdf5version) external
	mkdir external\build
	cmake -S external\hdf5-$(hdf5version) -B external\build -D CMAKE_INSTALL_PREFIX=external\hdf5-$(hdf5version)-win64 -D BUILD_SHARED_LIBS=OFF -D HDF5_ENABLE_Z_LIB_SUPPORT=OFF
	cmake --build external\build --config Release --target install
	rmdir /s /q external\build
	rmdir /s /q external\hdf5-$(hdf5version)

catch2lib:
	- rmdir /s /q external\build
	- rmdir /s /q external\Catch2-$(catch2version)
	- rmdir /s /q external\Catch2-$(catch2version)-win64
	tar -xf external/Catch2-$(catch2version).zip
	move Catch2-$(catch2version) external
	mkdir external\build
	cmake -S external\Catch2-$(catch2version) -B external\build
	cmake --build external\build --config Release --target package
	move external\build\_CPack_Packages\win64\NSIS\Catch2-$(catch2version)-win64 external
	rmdir /s /q external\build
	rmdir /s /q external\Catch2-$(catch2version)

clean:
	del build\*.obj
	del build\*.lib
	del "+quadriga_lib"\*.manifest
	del "+quadriga_lib"\*.exp
	del "+quadriga_lib"\*.lib
	del "+quadriga_lib"\*.mexw64

tidy: clean
	- rmdir /s /q external\Catch2-$(catch2version)-win64
	- rmdir /s /q external\hdf5-$(hdf5version)-win64
