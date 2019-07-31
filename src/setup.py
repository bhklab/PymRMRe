from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize


extensions = Extension("expt", sources = ["expt.pyx"], language = "c++", 
                       extra_compile_args = ["-stdlib=libc++"],
                       extra_link_args = ["-stdlib=libc++"],)

setup(name = "expt",
      ext_modules = cythonize(extensions))
# Need to run $ python setup.py build_ext --inplace
# Then "import expt" in other python files