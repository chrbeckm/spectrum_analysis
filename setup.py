from setuptools import setup, find_packages
from zipfile import ZipFile
setup(
        name="spectrum_analysis",
        version="0.2",
        description="Python package for spectrum analysis",
        author="stefangri, hmoldenhauer",
        url="https://github.com/hmoldenhauer/spectrum_analysis",
        download_url="https://github.com/hmoldenhauer/spectrum_analysis",
        license="MIT",
        package_dir={"": "lib"},
        packages=find_packages("lib"),
        classifiers=["Programming Language :: Python :: 3",
                     "License :: OSI Approved :: MIT License",
                     "Operating System :: OS Independent",
                     ],
        install_requires=[
            "lmfit>=0.9.11",
            "matplotlib>=3.1.1",
            "numpy>=1.16.4",
            "PyWavelets>=1.0.1",
            "scipy>=1.3.0",
            "statsmodels>=0.9.0",
            "pandas>=0.24.0",
        ],

        # matplotlib has C/C++ extensions, so it's not zip safe.
        # Telling setuptools this prevents it from doing an automatic
        # check for zip safety.
        zip_safe=False,
)
