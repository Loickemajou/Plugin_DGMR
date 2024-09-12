# -*- coding: utf-8 -*-

"""
Setup script for installing the dgmr model plugin.
"""

# IMPORTANT: Note that for this example we are referring to the plugin also as the
# "package".


# This setup script uses setuptools to handle the package installation and distribution.

from setuptools import setup, find_packages

# The long description is used to explain to the users how the package is installed and
# how it is used. Typically, a python package includes this long description in a
# separate README file.
# The following lines use the contents of the readme file as long
# description.
with open("README.md") as readme_file:
    long_description = readme_file.read()

# Add the plugin dependencies here. This dependencies will be installed along with your
# package.
requirements = ['numpy','pandas','pyproj','pysteps','tensorflow','tensorflow_hub','tensorflow_intel','wradlib','xarray','huggingface_hub']

# Add the dependencies needed to build the package.
# For example, if the package use compile extensions (like Cython), they can be included
# here.
setup_requirements = []
test_requirements = ['pytest>=3']

setup(
    author="pysteps",
    author_email=" ",
    python_requires=">=3.6",  # Pysteps supports python versions >3.6
    # Add the classifiers to the package.
    classifiers=[
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Deep learning model for nowcasting on radar images.",  # short description
    install_requires=requirements,
    license="MIT license",
    long_description=long_description,
    test_suite='tests',
    tests_require=test_requirements,
    include_package_data=True,
    keywords=["dgmr", "pysteps", "plugin", "importer"],
    name="pysteps_dgmr_nowcasts",
    packages=find_packages(),
    setup_requires=setup_requirements,
    # Entry points
    # ~~~~~~~~~~~~
    #
    # This is the most important part of the plugin setup script.
    # Entry points are a mechanism for an installed python distribution to advertise
    # some of the components installed (packages, modules, and scripts) to other
    # applications (in our case, pysteps).
    # https://packaging.python.org/specifications/entry-points/
    #
    # An entry point is defined by three properties:
    # - The group that an entry point belongs indicate the kind of functionality that
    #   provides. For the pysteps importers use the "pysteps.plugins.importers" group.
    # - The unique name that is used to identify this entry point in the
    #   "pysteps.plugins.importers" group.
    # - A reference to a Python object. For the pysteps importers, the object should
    #   point to a importer function, and should have the following form:
    #   package_name.module:function.
    # The setup script uses a dictionary mapping the entry point group names to a list
    # of strings defining the importers provided by this package (our plugin).
    # The general form of the entry points dictionary is:
    # entry_points={
    #     "group_name": [
    #         "entry_point_name=package_name.module:function",
    #         "entry_point_name=package_name.module:function2",
    #     ]
    # },
    # For this particular example, the entry points are defined as:
    entry_points={
        'pysteps.plugins.nowcasts': [
            'dgmr=dgmr_module_plugin.dgmr:forecast',

        ]
    },
    version="0.1.0",  # Indicate the version of the plugin
    # https://setuptools.readthedocs.io/en/latest/userguide/miscellaneous.html?highlight=zip_safe#setting-the-zip-safe-flag
    zip_safe=False,  # Do not compress the package.
)
