from setuptools import setup

setup(
    name="ops-translator",
    version="2.0.0",
    packages=["ops-translator"],
    install_requires=open("requirements.txt").read().splitlines(),
    python_requires=">=3.8",
)