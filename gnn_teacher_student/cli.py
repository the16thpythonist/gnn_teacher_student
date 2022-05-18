"""Console script for gnn_teacher_student."""
import sys
import click

from gnn_teacher_student.version import __version__ as version


@click.group('gnnst')
def cli():
    pass


@click.command('version', help='')
def version():
    click.secho(version)


cli.add_command(version)

if __name__ == "__main__":
    sys.exit(cli)  # pragma: no cover
