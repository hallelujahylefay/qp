import sys

import setuptools

# READ README.md for long description on PyPi.
try:
    long_description = open("README.md", encoding="utf-8").read()
except Exception as e:
    sys.stderr.write(f"Failed to read README.md:\n  {e}\n")
    sys.stderr.flush()
    long_description = ""

setuptools.setup(
    name="qp",
    author="Yvann Le Fay",
    description="Barrier methods for quadratic programs",
    long_description=long_description,
    version="0.1",
    packages=setuptools.find_packages(),
    install_requires=[
        "jax>=0.3.25",
        "jaxlib>=0.3.25",
        "pytest",
        "numpy>=1.24.3",
    ],
    long_description_content_type="text/markdown",
    keywords="quadratic-programming barrier-methods convex-optimization numerical-optimization newton-method",
    license="MIT",
    license_files=("LICENSE",),
)