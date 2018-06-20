import click
from click.testing import CliRunner
# from amktools import wav2brr

WAV = 'lol wav'
AMK = 'lol addmusic'
PROJECT = 'lol project'



def test_hello_world():
    @click.command()
    @click.argument('name', required=False)
    def hello(name):
        click.echo('Hello %s!' % name)

    runner = CliRunner()
    result = runner.invoke(hello, [])
    assert result.exit_code == 0
    assert result.output == 'Hello None!\n'
