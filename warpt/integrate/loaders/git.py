"""Load documentation from a git repository."""

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path

from warpt.integrate.loaders.base import DocLoader
from warpt.integrate.loaders.directory import DirectoryLoader

# Matches GitHub/GitLab browser URLs with /tree/branch/path
# e.g. https://github.com/ROCm/amdsmi/tree/amd-mainline/py-interface
_TREE_URL_PATTERN = re.compile(
    r"^(https://(?:github|gitlab)\.com/[^/]+/[^/]+)" r"/tree/([^/]+)(?:/(.+))?$"
)


def _parse_github_url(
    url: str,
) -> tuple[str, str | None, str | None]:
    """Parse a GitHub/GitLab URL into repo URL, branch, and subdir.

    Parameters
    ----------
    url : str
        A git or GitHub browser URL.

    Returns
    -------
    tuple[str, str | None, str | None]
        (clone_url, branch, subdirectory)
    """
    match = _TREE_URL_PATTERN.match(url)
    if match:
        repo_url = match.group(1)
        branch = match.group(2)
        subdir = match.group(3)  # may be None
        return repo_url, branch, subdir

    return url, None, None


class GitLoader(DocLoader):
    """Clone a git repo and load documentation from it."""

    def load(self, source: str) -> str:
        """Clone and load documentation from a git repository.

        Supports both standard git URLs and GitHub/GitLab
        browser URLs with /tree/branch/path.

        Parameters
        ----------
        source : str
            Git repository URL (HTTPS or SSH), or a GitHub
            browser URL pointing to a subdirectory.

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

        clone_url, branch, subdir = _parse_github_url(source)

        tmpdir = tempfile.mkdtemp(prefix="warpt_integrate_")
        try:
            clone_kwargs: dict[str, object] = {
                "depth": 1,
                "single_branch": True,
            }
            if branch:
                clone_kwargs["branch"] = branch

            gitmodule.Repo.clone_from(
                clone_url,
                tmpdir,
                **clone_kwargs,
            )

            load_path = tmpdir
            if subdir:
                load_path = str(Path(tmpdir) / subdir)
                if not Path(load_path).is_dir():
                    raise RuntimeError(
                        f"Subdirectory '{subdir}' not found "
                        f"in {clone_url} (branch: {branch})."
                    )

            return DirectoryLoader().load(load_path)
        except gitmodule.GitCommandError as exc:
            raise RuntimeError(
                f"Failed to clone {clone_url}: " f"{exc.stderr.strip()}"
            ) from exc
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
