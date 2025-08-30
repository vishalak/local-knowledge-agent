#!/usr/bin/env python3
"""
Streamlit Web UI for the Local Knowledge Base Agent.
Provides an easy-to-use web interface for querying the knowledge base.
"""

import streamlit as st
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from kb import KnowledgeBase

# Page configuration
st.set_page_config(
    page_title="Local Knowledge Base",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .source-title {
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .source-meta {
        font-size: 0.8rem;
        color: #666;
        margin-bottom: 0.5rem;
    }
    .source-preview {
        font-style: italic;
        color: #444;
        background-color: #fff;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin-top: 0.5rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    .metric-container {
        text-align: center;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'kb' not in st.session_state:
    st.session_state.kb = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'last_sources' not in st.session_state:
    st.session_state.last_sources = []

@st.cache_resource
def load_knowledge_base(config_path: str = "config.yaml"):
    """Load the knowledge base (cached for performance)."""
    try:
        return KnowledgeBase(config_path)
    except Exception as e:
        st.error(f"Failed to load knowledge base: {e}")
        return None

def format_source_for_display(source: Dict, index: int) -> str:
    """Format a source for web display."""
    path = source.get('path', 'Unknown')
    rel_path = Path(path).name if path else "Unknown"
    
    # Build source info
    info_parts = []
    if source.get('start') and source.get('end'):
        if source.get('start') == source.get('end'):
            info_parts.append(f"Line {source.get('start')}")
        else:
            info_parts.append(f"Lines {source.get('start')}-{source.get('end')}")
    
    if source.get('distance') is not None:
        relevance = max(0, 1 - source.get('distance'))
        info_parts.append(f"Relevance: {relevance:.1%}")
    
    # File metadata
    metadata_parts = []
    if source.get('is_code_file'):
        metadata_parts.append("📁 Code")
    elif source.get('is_documentation'):
        metadata_parts.append("📚 Documentation")
    elif source.get('is_config_file'):
        metadata_parts.append("⚙️ Configuration")
    
    if source.get('file_size_kb'):
        size_kb = source.get('file_size_kb')
        if size_kb < 1:
            metadata_parts.append(f"📏 {source.get('file_size', 0)} bytes")
        else:
            metadata_parts.append(f"📏 {size_kb} KB")
    
    if source.get('modified_date'):
        try:
            date_str = source.get('modified_date', '').split('T')[0]
            metadata_parts.append(f"🕒 Modified: {date_str}")
        except:
            pass
    
    # Content preview
    preview = ""
    if source.get('text'):
        text = source.get('text', '').strip()
        if len(text) > 150:
            preview = text[:150] + "..."
        else:
            preview = text
    
    return {
        'title': f"[{index}] {rel_path}",
        'info': " | ".join(info_parts),
        'metadata': " | ".join(metadata_parts),
        'preview': preview,
        'path': path
    }

def display_sources(sources: List[Dict]):
    """Display sources in the sidebar."""
    if not sources:
        return
    
    st.sidebar.markdown("### 📚 Sources Used")
    
    for i, source in enumerate(sources[:5], 1):
        formatted = format_source_for_display(source, i)
        
        with st.sidebar.expander(f"📄 {formatted['title']}", expanded=False):
            if formatted['info']:
                st.markdown(f"**Location:** {formatted['info']}")
            if formatted['metadata']:
                st.markdown(f"**Metadata:** {formatted['metadata']}")
            if formatted['preview']:
                st.markdown(f"**Preview:**")
                st.code(formatted['preview'], language=None)
            
            # VS Code link button
            if formatted['path']:
                start_line = source.get('start', 1)
                vscode_link = f"vscode://file/{formatted['path']}:{start_line}:1"
                st.markdown(f"[📝 Open in VS Code]({vscode_link})")
    
    if len(sources) > 5:
        st.sidebar.markdown(f"*... and {len(sources) - 5} more sources*")

def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 Local Knowledge Base Assistant</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.markdown("## ⚙️ Configuration")
    
    # Load knowledge base
    if st.session_state.kb is None:
        with st.spinner("Loading knowledge base..."):
            st.session_state.kb = load_knowledge_base()
    
    if st.session_state.kb is None:
        st.error("Could not load knowledge base. Please check your configuration.")
        return
    
    # Sidebar settings
    st.sidebar.markdown("### 🔧 Query Settings")
    k_value = st.sidebar.slider("Number of sources to retrieve", 1, 20, 6)
    show_sources = st.sidebar.checkbox("Show detailed sources", value=True)
    use_streaming = st.sidebar.checkbox("Enable streaming responses", value=True)
    
    # Knowledge base stats
    if hasattr(st.session_state.kb, 'collection'):
        try:
            count = st.session_state.kb.collection.count()
            st.sidebar.markdown("### 📊 Knowledge Base Stats")
            st.sidebar.metric("Total Documents", f"{count:,}")
        except:
            pass
    
    # Clear chat button
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.last_sources = []
        if st.session_state.kb and st.session_state.kb.chat_enabled:
            st.session_state.kb.clear_conversation()
        st.rerun()
    
    # Main chat interface
    st.markdown("## 💬 Chat with your Knowledge Base")
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message assistant-message"><strong>Assistant:</strong> {message["content"]}</div>', 
                       unsafe_allow_html=True)
    
    # Query input
    query = st.chat_input("Ask a question about your knowledge base...")
    
    if query:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        # Display user message
        st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {query}</div>', 
                   unsafe_allow_html=True)
        
        # Process query
        with st.spinner("Searching knowledge base..."):
            try:
                start_time = time.time()
                
                if use_streaming:
                    result = st.session_state.kb.ask_with_context(query, k=k_value, stream=True)
                    
                    # Create placeholder for streaming response
                    response_placeholder = st.empty()
                    full_response = ""
                    
                    # Stream the response
                    for chunk in result['response_stream']:
                        full_response += chunk
                        response_placeholder.markdown(
                            f'<div class="chat-message assistant-message"><strong>Assistant:</strong> {full_response}▌</div>', 
                            unsafe_allow_html=True
                        )
                    
                    # Final response without cursor
                    response_placeholder.markdown(
                        f'<div class="chat-message assistant-message"><strong>Assistant:</strong> {full_response}</div>', 
                        unsafe_allow_html=True
                    )
                    
                    # Finalize streaming response
                    st.session_state.kb.finalize_streaming_response(
                        query, 
                        full_response, 
                        result['context_docs'],
                        result['enhanced_query'],
                        result['start_time']
                    )
                    
                    response_text = full_response
                    sources = result['context_docs']
                    
                else:
                    result = st.session_state.kb.ask_with_context(query, k=k_value, stream=False)
                    response_text = result['response']
                    sources = result['context_docs']
                    
                    # Display response
                    st.markdown(f'<div class="chat-message assistant-message"><strong>Assistant:</strong> {response_text}</div>', 
                               unsafe_allow_html=True)
                
                # Add assistant response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                st.session_state.last_sources = sources
                
                # Show performance metrics
                response_time = time.time() - start_time
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Response Time", f"{response_time:.2f}s")
                with col2:
                    st.metric("Sources Found", len(sources))
                with col3:
                    if sources:
                        avg_relevance = sum(max(0, 1 - s.get('distance', 1)) for s in sources) / len(sources)
                        st.metric("Avg Relevance", f"{avg_relevance:.1%}")
                
            except Exception as e:
                st.error(f"Error processing query: {e}")
                return
    
    # Display sources in sidebar
    if show_sources and st.session_state.last_sources:
        display_sources(st.session_state.last_sources)
    
    # Footer
    st.markdown("---")
    st.markdown("💡 **Tips:** Use specific questions for better results. Try asking about code patterns, configurations, or documentation topics.")

if __name__ == "__main__":
    main()
