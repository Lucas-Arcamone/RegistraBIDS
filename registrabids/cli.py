import click

@click.group()
def cli():
    pass

@cli.command()
@click.argument("bids_path")
def index(bids_path):
    print(f"Indexing {bids_path}")

if __name__ == "__main__":
    cli()