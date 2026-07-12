from setuptools import setup, find_packages

setup(
    name="functional_supervised_classification",
    version="0.1.0",
    packages=find_packages(
        include=["functional_supervised_classification", "functional_supervised_classification.*"]
    ),
    description="Python programm for functional supervised classification",
    author="Hippolyte Guigon",
    author_email="hippolyte.auguste.guigon@gmail.com",
    url="https://github.com/HippolyteGuigon/functional_supervised_classification",
)
