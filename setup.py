import os
import platform
import re
import subprocess
import sys
from distutils.version import LooseVersion
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        cmake_version = LooseVersion(
            re.search(r"version\s*([\d.]+)", out.decode()).group(1)
        )
        if cmake_version < "3.11.0":
            raise RuntimeError("CMake >= 3.11.0 required")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DPYTHON_EXECUTABLE=" + sys.executable,
            "-DPIP_INSTALL=ON",
        ]

        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        if platform.system() == "Windows":
            cmake_args += [
                "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir)
            ]
            if sys.maxsize > 2**32:
                cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]
        else:
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
            build_args += ["--", "-j10"]

        env = os.environ.copy()
        env["CXXFLAGS"] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get("CXXFLAGS", ""), self.distribution.get_version()
        )
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )


setup(
    name="pyplayground",
    version=Path("version.txt").read_text().strip(),
    description="Playground code.",
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    license="MIT License",  # license_files default value should include the license file
    # The list of python packages
    py_modules=["pyplayground"],
    # The package is found in ./python/src
    package_dir={"": os.path.join(os.getcwd(), "python", "src")},
    # A list of instances of setuptools.Extension providing the list of Python extensions to be built.
    ext_modules=[CMakeExtension("playground_bindings")],
    # A dictionary providing a mapping of command names to Command subclasses.
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)

# Sources:
# - Single-sourcing the package version https://packaging.python.org/en/latest/guides/single-sourcing-package-version/
# - Making a PyPI-friendly README https://packaging.python.org/en/latest/guides/making-a-pypi-friendly-readme/
# - License https://setuptools.pypa.io/en/latest/references/keywords.html?highlight=license_files#keywords
