from distutils.core import setup

setup(
    name="amktools",
    version="",
    packages=["amktools.wav2brr"],
    py_modules=["amktools.mmkparser"],
    url="",
    license="BSD-3-Clause",
    author="nyanpasu64",
    author_email="",
    description="Preprocessor/compiler and automated sample extractor/tuner for AddMusicK",
    install_requires=[
        "sf2utils",
        "plumbum",
        "ruamel.yaml",
        "click",
        "more-itertools" 'dataclasses;python_version<"3.7"',
        "pygtrie",
    ],
    tests_require=["pytest", "pytest-mock", "pytest-pycharm"],
)
