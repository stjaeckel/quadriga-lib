# This Makefile is for Windows / MSVC environments
# Cheat sheet:
#	$@    Current target's full name (path, base name, extension)
#	$*    Current target's path and base name minus file extension.
#	$**   All dependents of the current target
#	$(@B) Current target's base name (no path, no extension)
#	$(@F) Current target's base name + estension (no path)

# Compilers
CC    = cl
MEX   = "C:\Program Files\MATLAB\R2022b\bin\mex"

# External headers
ARMA_H      = external\armadillo-11.4.2\include
PUGIXML_H   = external\pugixml-1.13\src

# External pre-compiled libraries
CATCH2_LIB  = /Iexternal\Catch2-3.3.2-win64\include /link Catch2.lib /LIBPATH:external\Catch2-3.3.2-win64\lib
ARMA_LIB    = /Iexternal\armadillo-11.4.2-win64\include /link armadillo.lib /LIBPATH:external\armadillo-11.4.2-win64\lib

# Configurations
CCFLAGS     = /EHsc /std:c++17 /nologo /MD /MP #/Wall 

all:   +quadriga_lib/calc_rotation_matrix.mexw64   +quadriga_lib/cart2geo.mexw64   +quadriga_lib/geo2cart.mexw64   \
       +quadriga_lib/arrayant_interpolate.mexw64   +quadriga_lib/arrayant_qdant_read.mexw64      +quadriga_lib/arrayant_qdant_write.mexw64   \
	   +quadriga_lib/version.mexw64   +quadriga_lib/arrayant_combine_pattern.mexw64   +quadriga_lib/interp.mexw64   \
	   +quadriga_lib/arrayant_generate.mexw64   +quadriga_lib/arrayant_calc_directivity.mexw64   +quadriga_lib/arrayant_rotate_pattern.mexw64 \
	   +quadriga_lib/get_channels_spherical.mexw64

test:   tests\test.exe
	tests\test.exe

tests\test.exe:   tests\quadriga_lib_catch2_tests.cpp   lib\quadriga_lib.lib
	$(CC) $(CCFLAGS) /Fetests\test.exe $** /Iinclude /I$(ARMA_H) $(CATCH2_LIB)
	del quadriga_lib_catch2_tests.obj

# Library files
build\quadriga_lib.obj:   src\quadriga_lib.cpp   include\quadriga_lib.hpp
	$(CC) $(CCFLAGS) /c src\$(@B).cpp /Fo$@ /Iinclude $(ARMA_LIB)

build\quadriga_tools.obj:   src\quadriga_tools.cpp   include\quadriga_tools.hpp
	$(CC) $(CCFLAGS) /c src\$(@B).cpp /Fo$@ /Iinclude $(ARMA_LIB)

build\qd_arrayant_interpolate.obj:   src\qd_arrayant_interpolate.cpp   src\qd_arrayant_interpolate.hpp
	$(CC) $(CCFLAGS) /openmp /c src\$(@B).cpp /Fo$@ /Iinclude $(ARMA_LIB)

build\qd_arrayant_qdant.obj:   src\qd_arrayant_qdant.cpp   src\qd_arrayant_qdant.hpp
	$(CC) $(CCFLAGS) /c src\$(@B).cpp /Fo$@ /Iinclude /I$(PUGIXML_H) $(ARMA_LIB) 

lib\quadriga_lib.lib:   build\quadriga_lib.obj   build\quadriga_tools.obj   build\qd_arrayant_interpolate.obj   build\qd_arrayant_qdant.obj
 	lib /OUT:$@ $**

# MEX interface files
+quadriga_lib/arrayant_combine_pattern.mexw64:   mex\arrayant_combine_pattern.cpp   lib\quadriga_lib.lib
	$(MEX) -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib/arrayant_interpolate.mexw64:   mex\arrayant_interpolate.cpp   lib\quadriga_lib.lib
	$(MEX) -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib/arrayant_qdant_read.mexw64:   mex\arrayant_qdant_read.cpp   lib\quadriga_lib.lib
 	$(MEX) -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib/arrayant_qdant_write.mexw64:   mex\arrayant_qdant_write.cpp   lib\quadriga_lib.lib
 	$(MEX) -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib/calc_rotation_matrix.mexw64:   mex\calc_rotation_matrix.cpp   lib\quadriga_lib.lib
 	$(MEX) -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib/cart2geo.mexw64:   mex\cart2geo.cpp   lib\quadriga_lib.lib
 	$(MEX) -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib/geo2cart.mexw64:   mex\geo2cart.cpp   lib\quadriga_lib.lib
 	$(MEX) -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib/version.mexw64:   mex\version.cpp   lib\quadriga_lib.lib
 	$(MEX) -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib/interp.mexw64:   mex\interp.cpp   lib\quadriga_lib.lib
 	$(MEX) -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib/arrayant_generate.mexw64:   mex\arrayant_generate.cpp   lib\quadriga_lib.lib
 	$(MEX) -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib/arrayant_calc_directivity.mexw64:   mex\arrayant_calc_directivity.cpp   lib\quadriga_lib.lib
	$(MEX) -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib/arrayant_rotate_pattern.mexw64:   mex\arrayant_rotate_pattern.cpp   lib\quadriga_lib.lib
	$(MEX) -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib/get_channels_spherical.mexw64:   mex\get_channels_spherical.cpp   lib\quadriga_lib.lib
	$(MEX) -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

# Clean up instructions
clean:
	del build\*.obj
	del build\*.lib
	del "+quadriga_lib"\*.manifest
	del "+quadriga_lib"\*.exp
	del "+quadriga_lib"\*.lib

tidy: clean
	del "+quadriga_lib"\*.mexw64
