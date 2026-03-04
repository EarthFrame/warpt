"""Load documentation from a git repository."""

from __future__ import annotations

import shutil
import tempfile

from warpt.integrate.loaders.base import DocLoader
from warpt.integrate.loaders.directory import DirectoryLoader


class GitLoader(DocLoader):
    """Clone a git repo and load documentation from it."""

    def load(self, source: str) -> str:
        """Clone and load documentation from a git repository.

        Parameters
        ----------
        source : str
            Git repository URL (HTTPS or SSH).

        Returns
        -------
        str
            Concatenated text from supported files.

        Raises
        ------
        ImportError
            If gitpython is not installed.
        RuntimeError
            If cloning fails.
        """
        try:
            import git as gitmodule
        except ImportError as exc:
            raise ImportError(
                "gitpython is required for git repo loading. "
                "Install with: pip install 'warpt[integrate]'"
            ) from exc

        tmpdir = tempfile.mkdtemp(prefix="warpt_integrate_")
        try:
            gitmodule.Repo.clone_from(
                source,
                tmpdir,
                depth=1,
                single_branch=True,
            )
            return DirectoryLoader().load(tmpdir)
        except gitmodule.GitCommandError as e:
            raise RuntimeError(
                f"Failed to clone {source}: {e}"
            ) from e
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
