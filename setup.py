#! /usr/bin/env python
#
# License: new BSD

import re
from setuptools import find_packages, setup

import orthoconv

DISTNAME = 'orthoconv'
PACKAGE_NAME = 'orthoconv'
VERSION = orthoconv.__version__
DESCRIPTION = ('A Python toolbox for orthonormal convolutions')
with open('README.rst') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'Alex Hepburn'
MAINTAINER_EMAIL = 'alex.hepburn@bristol.ac.uk'
LICENSE = 'new BSD'
PACKAGES = find_packages(exclude=['*.tests', '*.tests.*', 'tests.*', 'tests'])
PYTHON_REQUIRES = '~=3.5'
INCLUDE_PACKAGE_DATA = True

def setup_package():
    metadata = dict(name=DISTNAME,
                    maintainer=MAINTAINER,
                    maintainer_email=MAINTAINER_EMAIL,
                    description=DESCRIPTION,
                    license=LICENSE,
                    version=VERSION,
                    long_description=LONG_DESCRIPTION,
                    include_package_data=INCLUDE_PACKAGE_DATA,
                    python_requires=PYTHON_REQUIRES,
                    packages=PACKAGES)

    setup(**metadata)

if __name__ == "__main__":
    setup_package()