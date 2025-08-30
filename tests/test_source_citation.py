import sys, os
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from source_citation import SourceCitationManager


def make_doc(path: Path, text: str, start=10, end=20, distance=0.2, is_code=True, size_kb=4, modified_date="2025-08-01T12:34:56"):
    return {
        'path': str(path),
        'start': start,
        'end': end,
        'distance': distance,
        'text': text,
        'is_code_file': is_code,
        'file_size_kb': size_kb,
        'modified_date': modified_date,
    }


def test_format_sources_and_inline(tmp_path):
    src_root = tmp_path / 'src'
    src_root.mkdir()
    f1 = src_root / 'a.py'
    f1.write_text('print("hello")\n')
    f2 = src_root / 'b.md'
    f2.write_text('# Title\nBody text\n')

    mgr = SourceCitationManager(str(src_root))

    docs = [
        make_doc(f1, 'Function foo does bar quickly.'),
        make_doc(f2, 'Guide to bar configuration and setup.', start=5, end=5, distance=0.4, is_code=False),
    ]

    # Enhanced list formatting
    formatted = mgr.format_sources_for_response(docs, response_text='Irrelevant')
    assert '**[1]**' in formatted and '**[2]**' in formatted
    assert 'a.py' in formatted and 'b.md' in formatted
    assert 'Relevance' in formatted

    # Terminal formatting
    term = mgr.format_sources_for_terminal(docs, show_preview=True, max_sources=5)
    assert 'Sources:' in term and '[1]' in term and 'a.py' in term

    # VS Code links
    links = mgr.generate_vscode_links(docs)
    assert len(links) == 2 and links[0].startswith('vscode://file/')

    # Inline citations get added for overlapping keywords
    response = 'Bar configuration and setup are documented thoroughly. Foo implementation details and configuration exist.'
    with_citations = mgr.format_inline_citations(response, docs)
    assert with_citations.endswith('.')
    assert '[' in with_citations and ']' in with_citations
