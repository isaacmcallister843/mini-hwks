import io
import os
import platform

try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

try:
    from Cython.Build import cythonize
    import numpy
except:
    raise SystemExit(
        "Cython>=0.28 and numpy are required. Please install before proceeding"
    )

REQUIRED = ["numpy>=1.14", "Cython>=0.28", "scipy>=1.1", "numdifftools>=0.9"]
EXTRA_REQUIRED = {"test": ["mock", "nose"], "docs": ["Sphinx", "sphinx-rtd-theme>=0.4"]}
DESCRIPTION = "parameter estimation for simple Hawkes (self-exciting) processes"
CLASSIFIERS = [
    "Programming Language :: Python",
    "Programming Language :: Python :: 2",
    "Programming Language :: Cython",
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Information Analysis",
]

# make description
try:
    here = os.path.abspath(os.path.dirname(__file__))
    with io.open(os.path.join(here, "README.markdown"), encoding="utf-8") as f:
        LONG_DESCRIPTION = "\n" + f.read()
except:
    LONG_DESCRIPTION = DESCRIPTION


setup(
    name="hawkeslib",
    version="0.2.2",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="Caner Turkmen",
    author_email="caner.turkmen@boun.edu.tr",
    url="http://hawkeslib.rtfd.io",
    ext_modules=ext_mods,
    packages=["hawkeslib", "hawkeslib.model", "hawkeslib.model.c", "hawkeslib.util"],
    install_requires=REQUIRED,
    extras_require=EXTRA_REQUIRED,
    setup_requires=["numpy", "cython", "scipy"],
    license="MIT",
    python_requires=">=2.7.5",
    classifiers=CLASSIFIERS,
)
