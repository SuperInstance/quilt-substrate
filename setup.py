from setuptools import setup, find_packages

setup(
    name="quilt-substrate",
    version="0.1.0",
    description="The Quilt substrate, as a working Python library. 11-primitive cells, tensor encoding, Schrödinger pattern, fog-of-war decay, convoy consensus, witness log, opener layer.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="SuperInstance",
    license="MIT",
    py_modules=["substrate"],
    package_dir={"": "src"},
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    entry_points={
        "console_scripts": [
            "quilt-substrate=substrate:_cli",
        ],
    },
)
