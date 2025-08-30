import sys, os
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from document_processors import extract_text_from_file


def test_extract_text_from_file_text_and_fallback(tmp_path):
    # UTF-8
    p1 = tmp_path / 'utf8.txt'
    p1.write_text('Hello, UTF-8!')
    assert extract_text_from_file(p1) == 'Hello, UTF-8!'

    # Latin-1 fallback
    p2 = tmp_path / 'latin1.txt'
    p2.write_bytes('Café crème'.encode('latin-1'))
    assert extract_text_from_file(p2) == 'Café crème'

    # Binary bytes: function falls back to latin-1 and returns a decoded string
    p3 = tmp_path / 'bin.bin'
    p3.write_bytes(b"\xff\xfe\x00\x00")
    decoded = extract_text_from_file(p3)
    assert isinstance(decoded, str)
