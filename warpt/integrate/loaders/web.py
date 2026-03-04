"""Load documentation from a web URL."""

from __future__ import annotations

from warpt.integrate.loaders.base import DocLoader


class WebLoader(DocLoader):
    """Fetch a URL and convert HTML to markdown."""

    def load(self, source: str) -> str:
        """Fetch and convert a web page to text.

        Parameters
        ----------
        source : str
            HTTP or HTTPS URL.

        Returns
        -------
        str
            Page content converted to markdown text.

        Raises
        ------
        ImportError
            If httpx or markdownify is not installed.
        RuntimeError
            If the fetch fails.
        """
        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "httpx is required for web loading. "
                "Install with: pip install 'warpt[integrate]'"
            ) from exc

        try:
            from markdownify import markdownify as md
        except ImportError:
            md = None

        try:
            response = httpx.get(
                source,
                follow_redirects=True,
                timeout=30.0,
                headers={
                    "User-Agent": (
                        "warpt-integrate/1.0 "
                        "(documentation loader)"
                    )
                },
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise RuntimeError(
                f"Failed to fetch {source}: {e}"
            ) from e

        content_type = response.headers.get(
            "content-type", ""
        )

        # If it's HTML, convert to markdown
        if "html" in content_type and md is not None:
            return md(response.text, strip=["script", "style"])

        # If it's HTML but markdownify isn't available,
        # do basic tag stripping
        if "html" in content_type:
            return _strip_html(response.text)

        # Plain text or other — return as-is
        return response.text


def _strip_html(html: str) -> str:
    """Strip HTML tags and return plain text."""
    import re

    # Remove script and style blocks
    text = re.sub(
        r"<(script|style)[^>]*>.*?</\1>",
        "",
        html,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    # Restore some line breaks
    text = text.replace(". ", ".\n")
    return text.strip()
