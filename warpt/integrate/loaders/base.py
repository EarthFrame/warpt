"""Abstract base class and auto-detection for document loaders."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod


class DocLoader(ABC):
    """Abstract base class for SDK documentation loaders."""

    @abstractmethod
    def load(self, source: str) -> str:
        """Load and return normalized text from the source.

        Parameters
        ----------
        source : str
            Path, URL, or other identifier for the docs.

        Returns
        -------
        str
            Concatenated, normalized text content.
        """

    @staticmethod
    def detect_and_load(source: str) -> str:
        """Auto-detect source type and load documentation.

        Parameters
        ----------
        source : str
            Local path (file or directory), PDF path,
            git URL, or web URL.

        Returns
        -------
        str
            Loaded and normalized text.

        Raises
        ------
        ValueError
            If source type cannot be determined.
        FileNotFoundError
            If a local path does not exist.
        """
        # PDF file
        if source.lower().endswith(".pdf"):
            from warpt.integrate.loaders.pdf import PDFLoader

            return PDFLoader().load(source)

        # Git repository URL
        if source.endswith(".git") or source.startswith(
            ("https://github.com", "https://gitlab.com")
        ):
            from warpt.integrate.loaders.git import GitLoader

            return GitLoader().load(source)

        # Web URL
        if source.startswith(("http://", "https://")):
            from warpt.integrate.loaders.web import WebLoader

            return WebLoader().load(source)

        # Local directory
        if os.path.isdir(source):
            from warpt.integrate.loaders.directory import (
                DirectoryLoader,
            )

            return DirectoryLoader().load(source)

        # Local file
        if os.path.isfile(source):
            with open(source, encoding="utf-8", errors="replace") as f:
                return f.read()

        raise ValueError(
            f"Cannot determine source type for: {source!r}\n"
            "Expected: local directory, local file, .pdf path, "
            "git URL (.git or github.com/gitlab.com), "
            "or web URL (http/https)."
        )
