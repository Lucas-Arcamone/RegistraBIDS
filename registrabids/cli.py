import logging
from pathlib import Path

import click
import yaml

from registrabids.pipeline.runner import run_pipeline


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Ajout utile pour ton resolver
    config["_config_path"] = str(Path(config_path).resolve())
    return config


@click.command()
@click.argument(
    "bids_root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to YAML configuration file",
)
@click.option(
    "-v", "--verbose",
    is_flag=True,
    help="Enable debug logging",
)
def main(bids_root: Path, config_path: Path, verbose: bool):
    """
    Run the RegistraBIDS pipeline on a BIDS dataset.
    """
    setup_logging(verbose)

    config = load_config(str(config_path))

    run_pipeline(
        bids_root=str(bids_root),
        config=config,
    )


if __name__ == "__main__":
    main()