import logging
from pathlib import Path

import click
import yaml

from registrabids.pipeline.runner import run_pipeline


def setup_logging(log_level: str):
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
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
    "--output-dir",
    "output_dir",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help="Output directory for derivatives. Defaults to <bids_root>/derivatives/registrabids/",
)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    help="Logging level (default: INFO).",
    show_default=True,
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Force re-execution of all steps, ignoring existing outputs.",
)

def main(bids_root: Path, config_path: Path,  log_level: str,  output_dir: Path, force: bool):
    """
    Run the RegistraBIDS pipeline on a BIDS dataset.
    """
    setup_logging(log_level)

    config = load_config(str(config_path))

    run_pipeline(
        bids_root=str(bids_root),
        config=config,
        output_dir=output_dir,
        force=force,
    )


if __name__ == "__main__":
    main()