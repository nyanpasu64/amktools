from contextlib import contextmanager
from os import mkdir, chdir as cd
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from amktools import wav2brr

WAV = 'lol wav'
AMK = '../lol addmusic'
SAMPLE = 'lol sample'

AMK_SAMPLES = Path(AMK, 'samples')
PROJECT = 'project'


@pytest.fixture(scope='function')
def filesystem_tree():
    """ Creates an isolated filesystem tree, with proper directory structure.
    isolated_filesystem is an effectively static method, so yielding `runner`
    is unnecessary! """
    runner = CliRunner()
    with runner.isolated_filesystem():
        mkdir(PROJECT)
        cd(PROJECT)

        mkdir(WAV)
        mkdir(AMK)
        # with pushd(AMK):
        #     mkdir('samples')
        AMK_SAMPLES.mkdir()

        yield


def test_pushd():
    SUB = 'sub~folder'

    runner = CliRunner()
    with runner.isolated_filesystem():
        root = Path().resolve()
        mkdir(SUB)

        with wav2brr.pushd(SUB):
            sub = Path().resolve()
            assert root / SUB == sub

        with wav2brr.pushd(Path(SUB)):
            sub = Path().resolve()
            assert root / SUB == sub


@contextmanager
def is_moved(dir: Path, name, method: str, is_moved: bool) -> None:
    before = dir / name
    after = dir / wav2brr.BACKUP_ROOT / name

    getattr(before, method)(exist_ok=True)
    yield

    assert after.exists() == is_moved
    assert before.exists() != is_moved


def test_sample_dir(filesystem_tree):
    runner = CliRunner()

    # Valid sample path
    result = runner.invoke(wav2brr.main, [WAV, AMK, SAMPLE], catch_exceptions=False)
    assert result.exit_code == 0
    assert 'Creating sample folder' in result.output
    assert Path(AMK, 'samples', SAMPLE).exists()

    # Ensure existing files are backed up, not deleted
    assert '/' not in wav2brr.BACKUP_ROOT
    sample_dir = AMK_SAMPLES / SAMPLE

    for i in range(2):  # Ensure backups are overwritten without errors
        with is_moved(sample_dir, 'file', 'touch', True), \
             is_moved(sample_dir, 'directory', 'mkdir', False):
            result = runner.invoke(wav2brr.main, [WAV, AMK, SAMPLE], catch_exceptions=False)
            assert result.exit_code == 0
            assert 'Creating sample folder' not in result.output

    # Invalid sample path (attempting to mangle sample-root folder)
    with pytest.raises(click.BadParameter):
        # click is weird.
        # catch_exceptions=False stops the test runner from eating exceptions.
        # standalone_mode=False stops main() from pretty-printing BadParameter.
        runner.invoke(wav2brr.main, [WAV, AMK, '.'], catch_exceptions=False,
                      standalone_mode=False)
