from setuptools import setup

setup(
    name="ops-translator",
    version="2.0.0",
    packages=["ops-translator"],
    install_requires=open("requirements.txt").read().splitlines(),
    python_requires=">=3.8",
    #entry_points={"console_scripts": ["ops-translator = ops-translator.__main__:main"]},
)