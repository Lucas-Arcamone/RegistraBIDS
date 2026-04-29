import click

@click.group()
def cli():
    pass

@click.command()
@click.argument("bids_root")
@click.option("--config", required=True, help="Path to YAML config")

def index(bids_path):
    print(f"Indexing {bids_path}")

if __name__ == "__main__":
    cli()