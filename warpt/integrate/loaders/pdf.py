"""Load documentation from PDF files using PyMuPDF."""

from __future__ import annotations

from pathlib import Path

from warpt.integrate.loaders.base import DocLoader


class PDFLoader(DocLoader):
    """Extract text from PDF files using pymupdf (fitz)."""

    def load(self, source: str) -> str:
        """Load text content from a PDF file.

        Parameters
        ----------
        source : str
            Path to the PDF file.

        Returns
        -------
        str
            Extracted text from all pages.

        Raises
        ------
        FileNotFoundError
            If the PDF file does not exist.
        ImportError
            If pymupdf is not installed.
        """
        path = Path(source).resolve()
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        try:
            import fitz  # pymupdf
        except ImportError as exc:
            raise ImportError(
                "pymupdf is required for PDF loading. "
                "Install with: pip install 'warpt[integrate]'"
            ) from exc

        parts: list[str] = []

        with fitz.open(str(path)) as doc:
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text()
                if text.strip():
                    parts.append(
                        f"--- Page {page_num} ---\n{text}"
                    )

        if not parts:
            raise ValueError(
                f"No text content extracted from PDF: {path}"
            )

        return "\n\n".join(parts)
