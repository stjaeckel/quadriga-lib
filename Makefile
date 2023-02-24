# This Makefile is for Windows / MSVC environments

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
build\quadriga_lib.lib:   src\quadriga_lib.cpp   include\quadriga_lib.hpp
	$(CC) $(CCFLAGS) /c src\$(@B).cpp /Fo$*.obj /Iinclude /I$(ARMA_H)
	lib $*.obj

build\quadriga_tools.lib:   src\quadriga_tools.cpp   include\quadriga_tools.hpp
	$(CC) $(CCFLAGS) /c src\$(@B).cpp /Fo$*.obj /Iinclude /I$(ARMA_H)
	lib $*.obj

build\qd_arrayant_interpolate.lib:   src\qd_arrayant_interpolate.cpp   src\qd_arrayant_interpolate.hpp
	$(CC) $(CCFLAGS) /openmp /c src\$(@B).cpp /Fo$*.obj /Iinclude /I$(ARMA_H)
	lib $*.obj

build\qd_arrayant_qdant.lib:   src\qd_arrayant_qdant.cpp   src\qd_arrayant_qdant.hpp
	$(CC) $(CCFLAGS) /c src\$(@B).cpp /Fo$*.obj /Iinclude /I$(ARMA_H) /I$(PUGIXML_H)
	lib $*.obj

# MEX interface files
+quadriga_lib/arrayant_interpolate.mexw64:   mex\arrayant_interpolate.cpp   build\quadriga_lib.lib   build\quadriga_tools.lib   build\qd_arrayant_interpolate.lib
	$(MEX) -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib/arrayant_qdant_read.mexw64:   mex\arrayant_qdant_read.cpp   build\quadriga_lib.lib   build\qd_arrayant_qdant.lib
	$(MEX) -outdir +quadriga_lib $** -Iinclude -Isrc -I$(ARMA_H)

+quadriga_lib/calc_rotation_matrix.mexw64:   mex\calc_rotation_matrix.cpp   build\quadriga_tools.lib
	$(MEX) -outdir +quadriga_lib $** -Iinclude -I$(ARMA_H)

+quadriga_lib/cart2geo.mexw64:   mex\cart2geo.cpp   build\quadriga_tools.lib
	$(MEX) -outdir +quadriga_lib $** -Iinclude -I$(ARMA_H)

+quadriga_lib/geo2cart.mexw64:   mex\geo2cart.cpp   build\quadriga_tools.lib
	$(MEX) -outdir +quadriga_lib $** -Iinclude -I$(ARMA_H)

+quadriga_lib/version.mexw64:   mex\version.cpp   build\quadriga_lib.lib
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

