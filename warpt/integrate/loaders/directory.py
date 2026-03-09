"""Load documentation from a local directory tree."""

from __future__ import annotations

from pathlib import Path

from warpt.integrate.loaders.base import DocLoader

# File extensions to include when scanning directories
SUPPORTED_EXTENSIONS = {
    ".md",
    ".rst",
    ".txt",
    ".py",
    ".h",
    ".c",
    ".cpp",
    ".hpp",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".cfg",
    ".ini",
}


class DirectoryLoader(DocLoader):
    """Recursively load text files from a local directory."""

    def __init__(
        self,
        extensions: set[str] | None = None,
        max_file_size: int = 512_000,
    ) -> None:
        """Initialize the directory loader.

        Parameters
        ----------
        extensions : set[str] | None
            File extensions to include. Defaults to
            SUPPORTED_EXTENSIONS.
        max_file_size : int
            Skip files larger than this (bytes). Default 512 KB.
        """
        self.extensions = extensions or SUPPORTED_EXTENSIONS
        self.max_file_size = max_file_size

    def load(self, source: str) -> str:
        """Load all matching files from a directory tree.

        Parameters
        ----------
        source : str
            Path to the root directory.

        Returns
        -------
        str
            Concatenated file contents with path headers.

        Raises
        ------
        FileNotFoundError
            If the directory does not exist.
        """
        root = Path(source).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Directory not found: {root}")

        parts: list[str] = []
        files = sorted(root.rglob("*"))

        for filepath in files:
            if not filepath.is_file():
                continue
            if filepath.suffix.lower() not in self.extensions:
                continue
            if filepath.stat().st_size > self.max_file_size:
                continue
            # Skip hidden files and directories
            if any(p.startswith(".") for p in filepath.relative_to(root).parts):
                continue

            try:
                content = filepath.read_text(encoding="utf-8", errors="replace")
            except (OSError, UnicodeDecodeError):
                continue

            rel_path = filepath.relative_to(root)
            parts.append(f"--- {rel_path} ---\n{content}")

        if not parts:
            raise ValueError(
                f"No supported files found in {root}. "
                f"Supported extensions: {self.extensions}"
            )

        return "\n\n".join(parts)
