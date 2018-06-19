from distutils.core import setup

setup(
    name='amktools',
    version='',
    packages=['amktools.wav2brr'],
    py_modules=['amktools.mmkparser'],
    url='',
    license='BSD-3-Clause',
    author='nyanpasu64',
    author_email='',
    description='Preprocessor/compiler and automated sample extractor/tuner for AddMusicK',
    requires=['sf2utils', 'plumbum', 'ruamel.yaml', 'click'],
    # TODO https://python-packaging.readthedocs.io/en/latest/command-line-scripts.html#the-console-scripts-entry-point
)
