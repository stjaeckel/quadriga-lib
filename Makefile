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

# Static library headers
ARMA_H      = external\armadillo-11.4.2\include
PUGIXML_H   = external\pugixml-1.13\src

# Configurations
CCFLAGS     = /EHsc /std:c++17 /nologo /MD /MP #/Wall 

all:   +quadriga_lib/calc_rotation_matrix.mexw64   +quadriga_lib/cart2geo.mexw64   +quadriga_lib/geo2cart.mexw64   \
       +quadriga_lib/arrayant_interpolate.mexw64   +quadriga_lib/arrayant_qdant_read.mexw64   +quadriga_lib/version.mexw64

# Library files
build\quadriga_lib.obj:   src\quadriga_lib.cpp   include\quadriga_lib.hpp
	$(CC) $(CCFLAGS) /c src\$(@B).cpp /Fo$@ /Iinclude /I$(ARMA_H)

build\quadriga_tools.obj:   src\quadriga_tools.cpp   include\quadriga_tools.hpp
	$(CC) $(CCFLAGS) /c src\$(@B).cpp /Fo$@ /Iinclude /I$(ARMA_H)

build\qd_arrayant_interpolate.obj:   src\qd_arrayant_interpolate.cpp   src\qd_arrayant_interpolate.hpp
	$(CC) $(CCFLAGS) /openmp /c src\$(@B).cpp /Fo$@ /Iinclude /I$(ARMA_H)

build\qd_arrayant_qdant.obj:   src\qd_arrayant_qdant.cpp   src\qd_arrayant_qdant.hpp
	$(CC) $(CCFLAGS) /c src\$(@B).cpp /Fo$@ /Iinclude /I$(ARMA_H) /I$(PUGIXML_H)

build\quadriga_lib_combined.lib:   build\quadriga_lib.obj   build\quadriga_tools.obj   build\qd_arrayant_interpolate.obj   build\qd_arrayant_qdant.obj
 	lib /OUT:$@ $**

# MEX interface files
+quadriga_lib/arrayant_interpolate.mexw64:   mex\arrayant_interpolate.cpp   build\quadriga_lib_combined.lib
	$(MEX) -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib/arrayant_qdant_read.mexw64:   mex\arrayant_qdant_read.cpp   build\quadriga_lib_combined.lib
 	$(MEX) -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib/calc_rotation_matrix.mexw64:   mex\calc_rotation_matrix.cpp   build\quadriga_tools.obj
 	$(MEX) -outdir +quadriga_lib $** -Iinclude -I$(ARMA_H)

+quadriga_lib/cart2geo.mexw64:   mex\cart2geo.cpp   build\quadriga_tools.obj
 	$(MEX) -outdir +quadriga_lib $** -Iinclude -I$(ARMA_H)

+quadriga_lib/geo2cart.mexw64:   mex\geo2cart.cpp   build\quadriga_tools.obj
 	$(MEX) -outdir +quadriga_lib $** -Iinclude -I$(ARMA_H)

+quadriga_lib/version.mexw64:   mex\version.cpp   build\quadriga_lib_combined.lib
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

