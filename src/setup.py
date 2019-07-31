from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

import numpy

extensions = Extension("expt", sources = ["expt.pyx", "Matrix.cpp", "Data.cpp", "MutualInformationMatrix.cpp", "Filter.cpp", "Math.cpp"], 
                       language = "c++", 
                       extra_compile_args = ["-stdlib=libc++"],
                       extra_link_args = ["-stdlib=libc++"],
                       #define_macros = [('DL_IMPORT')],
                       include_dirs=[numpy.get_include()],)

setup(name = "expt",
      #cmdclass = {'build_ext': build_ext},
      ext_modules = cythonize(extensions))
# Need to run $ python setup.py build_ext --inplace
# Then "import expt" in other python files