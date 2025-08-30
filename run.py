"""
Module for executing the application script with logging and error handling.

This module serves as the entry point for running the application script,
utilizing subprocess to execute the Python script located in the app directory.
It implements logging for tracking script execution and handles errors.

Attributes:
    logger: Logger instance for logging script execution status and errors.
    repo_root: Path object representing the root directory of the repository.
    src_dir: Path object representing the resolved path to the app directory.
    scripts: List of tuples containing script names and their corresponding file names.

Raises:
    SystemExit: Exits with code 1 if a subprocess error occurs during script execution.
"""
import subprocess
import sys
from pathlib import Path

from scripts.logging_config import get_logger, setup_logging

logger = get_logger(__name__)
setup_logging()

repo_root = Path(__file__).parent
src_dir = Path(repo_root / "app").resolve()

scripts = [
        ('Dash app', 'app.py')
    ]

try:
    for script_name, script_file in scripts:
        logger.info(f"Running {str(script_name)}...")

        script_path = src_dir / script_file
        # Run the script
        command = [sys.executable, str(script_path)]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
except subprocess.CalledProcessError as e:
    logger.error(f"Error occurred while running {str(script_name)}: {e.stderr}")
    sys.exit(1)
