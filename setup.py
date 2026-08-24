from setuptools import setup, find_packages

setup(
    name="quilt-substrate",
    version="0.2.0",
    description="The Quilt substrate, as a working Python library. 11-primitive cells, tensor encoding, Schrödinger pattern, fog-of-war decay, convoy consensus, witness log, opener layer, Opener ABC.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="SuperInstance",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "quilt-substrate=quilt_substrate.substrate:_cli",
        ],
    },
)
