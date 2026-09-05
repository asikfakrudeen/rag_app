import os
import fitz          # PyMuPDF  — PDF
import docx          # python-docx — DOCX
import pandas as pd  # pandas + openpyxl — CSV / XLSX


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pages_from_text(raw_text: str, source: str) -> list[dict]:
    """
    Wraps plain text (no page concept) into the standard page-dict format
    used by create_chunks(). Treats the entire content as a single 'page 1'.
    """
    text = raw_text.strip()
    if not text:
        return []
    return [{"text": text, "page": 1, "source": source}]


# ── Per-format loaders ────────────────────────────────────────────────────────

def _load_pdf(path: str) -> list[dict]:
    """
    Extracts text page-by-page from a PDF using PyMuPDF.
    Returns one dict per non-empty page, preserving page numbers.
    """
    source = os.path.basename(path)
    document = fitz.open(path)
    pages = []
    for page_number, page in enumerate(document):
        text = page.get_text()
        if text.strip():
            pages.append({
                "text": text,
                "page": page_number + 1,
                "source": source
            })
    document.close()
    return pages


def _load_docx(path: str) -> list[dict]:
    """
    Extracts text from a .docx file using python-docx.
    Joins all paragraphs into a single block (no page boundary metadata
    is available in the .docx format without rendering the document).
    """
    source = os.path.basename(path)
    doc = docx.Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    raw_text = "\n".join(paragraphs)
    return _pages_from_text(raw_text, source)


def _load_txt(path: str) -> list[dict]:
    """
    Reads a plain-text or Markdown file with UTF-8 encoding.
    Falls back to latin-1 if a decoding error occurs.
    """
    source = os.path.basename(path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_text = f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            raw_text = f.read()
    return _pages_from_text(raw_text, source)


def _load_csv(path: str) -> list[dict]:
    """
    Reads a CSV file with pandas and converts every row into a readable
    'column: value' sentence so the RAG can reason about tabular data naturally.
    """
    source = os.path.basename(path)
    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        row_text = " | ".join(f"{col}: {val}" for col, val in row.items() if str(val).strip())
        if row_text.strip():
            rows.append(row_text)
    raw_text = "\n".join(rows)
    return _pages_from_text(raw_text, source)


def _load_xlsx(path: str) -> list[dict]:
    """
    Reads an Excel file with pandas (all sheets).
    Each sheet is treated as a separate 'page', converting rows to readable text.
    """
    source = os.path.basename(path)
    xl = pd.ExcelFile(path)
    pages = []
    for sheet_number, sheet_name in enumerate(xl.sheet_names, start=1):
        df = xl.parse(sheet_name)
        rows = []
        for _, row in df.iterrows():
            row_text = " | ".join(f"{col}: {val}" for col, val in row.items() if str(val).strip())
            if row_text.strip():
                rows.append(row_text)
        raw_text = "\n".join(rows)
        if raw_text.strip():
            pages.append({
                "text": raw_text,
                "page": sheet_number,      # sheet number acts as page
                "source": source
            })
    return pages


# ── Public API ────────────────────────────────────────────────────────────────

# Maps lowercase file extensions to their loader functions
_LOADERS = {
    ".pdf":  _load_pdf,
    ".docx": _load_docx,
    ".txt":  _load_txt,
    ".md":   _load_txt,   # Markdown is plain text
    ".csv":  _load_csv,
    ".xlsx": _load_xlsx,
}

SUPPORTED_EXTENSIONS = list(_LOADERS.keys())


def load_document(path: str) -> list[dict]:
    """
    Universal entry-point. Detects the file extension and delegates to the
    appropriate loader. Raises ValueError for unsupported types.

    Returns:
        List of dicts: [{"text": str, "page": int, "source": str}, ...]
    """
    ext = os.path.splitext(path)[1].lower()
    loader = _LOADERS.get(ext)

    if loader is None:
        raise ValueError(
            f"Unsupported file type: '{ext}'. "
            f"Supported types: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    return loader(path)
