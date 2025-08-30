import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from query_transformer import QueryTransformer, transform_query


def test_transformer_methods_without_ollama():
    qt = QueryTransformer(llm_model='dummy')

    # HyDE should gracefully return None when ollama is unavailable
    assert qt.generate_hyde_query('What is RAG?') is None

    # Expand should at least include the original query
    expanded = qt.expand_query('optimize indexing speed')
    assert isinstance(expanded, list) and 'optimize indexing speed' in expanded

    # Enhance adds context hints and technical qualifiers
    enhanced = qt.enhance_query_with_context('how to install on windows', context_hints=['C#', 'kb'])
    assert 'in context of: C# kb' in enhanced
    assert 'documentation' in enhanced and 'installation' in enhanced

    # Key terms extracts technical tokens and quoted phrases
    terms = qt.extract_key_terms('Fix "IndexError" in split_csharp_code function')
    assert 'IndexError' in terms or 'indexerror' in [t.lower() for t in terms]
    assert 'split_csharp_code' in terms


def test_transform_query_variants():
    # enhance
    res = transform_query('how to configure reranker', method='enhance', context_hints=['BM25'])
    assert res['method'] == 'enhance'
    assert 'enhanced_query' in res and isinstance(res['transformed_queries'], list)

    # expand
    res2 = transform_query('semantic chunking', method='expand')
    assert res2['method'] == 'expand'
    assert isinstance(res2['transformed_queries'], list)

    # extract
    res3 = transform_query('Troubleshoot hybrid_score mismatch', method='extract')
    assert 'key_terms' in res3 and len(res3['transformed_queries']) >= 2
