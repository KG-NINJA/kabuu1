"""Ensure heavy third-party dependencies are available before test collection."""
from __future__ import annotations

import importlib
import subprocess
import sys
from typing import Iterable, Tuple


REQUIRED_PACKAGES: Tuple[Tuple[str, str], ...] = (
    ("numpy", "1.26.0"),
    ("yfinance", "0.2.66"),
    ("holidays", "0.34"),
    ("scikit-learn", "1.5.0"),
    ("pandas", "2.3.3"),
    ("schedule", "1.2.1"),
    ("PyYAML", "6.0.2"),

)


def _install_package(requirement: str) -> None:
    """Install a package using pip in the current Python environment."""

    subprocess.check_call([sys.executable, "-m", "pip", "install", requirement])


def _ensure_dependencies(packages: Iterable[Tuple[str, str]]) -> None:
    """Ensure each named dependency can be imported, installing it if required."""

    for package, version in packages:
        requirement = f"{package}=={version}" if version else package
        try:
            importlib.import_module(package)
        except ModuleNotFoundError:
            _install_package(requirement)
        finally:
            importlib.invalidate_caches()


def pytest_configure() -> None:
    """Install required packages before pytest starts collecting modules."""

    _ensure_dependencies(REQUIRED_PACKAGES)

