from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path


here = path.abspath(path.dirname(__file__))


with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


version_namespace = {}
with open(path.join(here, "docoskin/version.py")) as f:
    exec(f.read(), version_namespace)


setup(
    name="docoskin",
    version=version_namespace["__version__"],

    description='"Onion-skin" visual differences between a reference document image and a scanned copy',
    long_description=long_description,
    long_description_content_type="text/markdown",

    url='https://github.com/risicle/docoskin',

    author="Robert Scott",
    author_email="code@humanleg.org.uk",

    license="GPLv3",

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Legal Industry',
        'Topic :: Multimedia :: Graphics :: Viewers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',

        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
    ],
    keywords="document scan compare visual diff signature",
    packages=["docoskin"],
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    install_requires=[
        # omitting opencv >= 3.1.0 dependency as there is no standard python package for it and therefore no universal
        # way to detect it. i advocate using nix to satisfy all these dependencies, hence the included default.nix.
        "six",
    ],
    extras_require={
        ':python_version<"3"': ["futures"],
        "pypi_opencv": ["opencv-python >= 3.1.0"],
    },

    entry_points={
        'console_scripts': [
            'docoskin=docoskin.docoskin:main',
        ],
    },
)
