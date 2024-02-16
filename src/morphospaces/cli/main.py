import typer

from morphospaces.cli import skeletonize_cli

app = typer.Typer(
    help="The morphospaces command line interface.", no_args_is_help=True
)
app.add_typer(skeletonize_cli.app, name="skeletonize")

if __name__ == "__main__":
    app()
