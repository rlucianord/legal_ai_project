import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def test_extract_text_uses_docling_when_available(monkeypatch):
    processfiles = importlib.import_module("processfiles")

    class DummyDocument:
        def export_to_text(self):
            return "Texto extraído con Docling"

    class DummyResult:
        def __init__(self):
            self.document = DummyDocument()

    class DummyConverter:
        def convert(self, file_path):
            return DummyResult()

    monkeypatch.setattr(processfiles, "DocumentConverter", DummyConverter, raising=False)
    monkeypatch.setattr(processfiles, "pdfminer_extract_text", lambda _: None, raising=False)
    monkeypatch.setattr(processfiles, "PdfReader", None, raising=False)

    assert processfiles.extract_text("/tmp/ejemplo.pdf") == "Texto extraído con Docling"
