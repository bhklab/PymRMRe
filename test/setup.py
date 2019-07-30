from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

import numpy

extensions = Extension("dum", sources = ["dum.pyx"], language = "c++", 
                       extra_compile_args = ["-stdlib=libc++"],
                       extra_link_args = ["-stdlib=libc++"],
                       #define_macros = [('DL_IMPORT')],
                       include_dirs=[numpy.get_include()],)

setup(name = "dum",
      #cmdclass = {'build_ext': build_ext},
      ext_modules = cythonize(extensions))

'''
setup(
    name = "dum",
    #cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("dum",
                             sources=["dum.pyx"],
                             language = "c++",
                             extra_compile_args = ["-stdlib=libc++"],
                             extra_link_args = ["-stdlib=libc++"],
                             include_dirs=[numpy.get_include()])],
)
'''
