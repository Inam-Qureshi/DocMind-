
import requests
import streamlit as st

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="DocuMind",
    page_icon="📚",
    layout="wide"
)

st.title("📚 DocMind")
st.caption("Chat with your PDF documents using RAG")

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "ui_notice" not in st.session_state:
    st.session_state.ui_notice = None

# --------------------------------------------------
# API CONFIGURATION
# --------------------------------------------------

st.sidebar.header("Backend Configuration")

API_BASE_URL = st.sidebar.text_input(
    "FastAPI Base URL",
    value="http://127.0.0.1:8000"
).strip().rstrip("/")

def api_request(method, endpoint, timeout=60, **kwargs):
    """Send an HTTP request to the FastAPI backend."""

    try:
        response = requests.request(
            method=method,
            url=f"{API_BASE_URL}{endpoint}",
            timeout=timeout,
            **kwargs
        )

    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to FastAPI. "
            "Make sure the backend server is running."
        )
        return None

    except requests.exceptions.Timeout:
        st.error("The backend request timed out.")
        return None

    except requests.exceptions.RequestException as error:
        st.error(f"Request failed: {error}")
        return None

    try:
        data = response.json() if response.content else {}
    except ValueError:
        data = {"detail": response.text}

    if not response.ok:
        detail = (
            data.get("detail", response.text)
            if isinstance(data, dict)
            else response.text
        )

        st.error(f"API Error ({response.status_code}): {detail}")
        return None

    return data

# --------------------------------------------------
# BACKEND HEALTH CHECK
# --------------------------------------------------

health = api_request(
    "GET",
    "/health",
    timeout=5
)

if health and health.get("status") == "healthy":
    st.sidebar.success("Backend connected")
else:
    st.sidebar.error("Backend unavailable")

# --------------------------------------------------
# LOAD DOCUMENTS
# --------------------------------------------------

documents_response = api_request(
    "GET",
    "/documents/",
    timeout=15
)

if documents_response is not None:
    st.session_state.documents = documents_response.get(
        "documents", []
    )
elif "documents" not in st.session_state:
    st.session_state.documents = []

documents = st.session_state.documents

document_lookup = {
    document["id"]: document
    for document in documents
}

document_options = [None] + list(document_lookup.keys())

# --------------------------------------------------
# DOCUMENT FILTER
# --------------------------------------------------

st.sidebar.divider()

st.sidebar.header("Document Selection")

selected_document_id = st.sidebar.selectbox(
    "Search within",
    options=document_options,
    format_func=lambda document_id: (
        "All documents"
        if document_id is None
        else document_lookup[document_id].get(
            "original_filename",
            document_id
        )
    ),
    key="document_scope"
)

top_k = st.sidebar.slider(
    "Number of retrieved chunks",
    min_value=1,
    max_value=10,
    value=3
)

# --------------------------------------------------
# DISPLAY NOTIFICATIONS
# --------------------------------------------------

if st.session_state.ui_notice:
    st.success(st.session_state.ui_notice)
    st.session_state.ui_notice = None

# --------------------------------------------------
# APPLICATION TABS
# --------------------------------------------------

tab_chat, tab_documents, tab_search = st.tabs(
    [
        "💬 Chat",
        "📄 Documents",
        "🔍 Semantic Search"
    ]
)

# ==================================================
# TAB 1: CHAT WITH DOCUMENTS
# ==================================================

with tab_chat:

    st.subheader("Ask questions about your documents")

    for message in st.session_state.chat_history:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            if message.get("sources"):

                source_labels = [
                    f'{source["filename"]}, page {source["page_number"]}'
                    for source in message["sources"]
                ]

                st.caption(
                    "Sources: " + " | ".join(source_labels)
                )

    with st.form("chat_form", clear_on_submit=True):

        question = st.text_input(
            "Your question",
            placeholder="Ask something about your PDF documents..."
        )

        submitted = st.form_submit_button(
            "Send question",
            type="primary"
        )

    if submitted:

        question = question.strip()

        if not question:

            st.warning("Please enter a question.")

        else:

            st.session_state.chat_history.append({
                "role": "user",
                "content": question
            })

            request_data = {
                "question": question,
                "document_id": selected_document_id,
                "top_k": top_k
            }

            with st.spinner("Searching documents and generating answer..."):

                result = api_request(
                    "POST",
                    "/chat/ask",
                    json=request_data,
                    timeout=(10, 240)
                )

            if result:

                assistant_message = {
                    "role": "assistant",
                    "content": result.get(
                        "answer",
                        "No answer was returned."
                    ),
                    "sources": result.get("sources", [])
                }

            else:

                assistant_message = {
                    "role": "assistant",
                    "content": (
                        "I couldn't get a response from the backend. "
                        "Please check the API error above and try again."
                    ),
                    "sources": []
                }

            st.session_state.chat_history.append(
                assistant_message
            )

            st.rerun()

    if st.button("Clear chat history"):
        st.session_state.chat_history = []
        st.rerun()

# ==================================================
# TAB 2: DOCUMENT MANAGEMENT
# ==================================================

with tab_documents:

    st.subheader("Upload and manage PDF documents")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        accept_multiple_files=False
    )

    if st.button(
        "Upload and index PDF",
        type="primary"
    ):

        if uploaded_file is None:

            st.warning("Please select a PDF first.")

        else:

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type or "application/pdf"
                )
            }

            with st.spinner(
                "Extracting text, generating embeddings and indexing document..."
            ):

                result = api_request(
                    "POST",
                    "/documents/upload",
                    files=files,
                    timeout=(10, 600)
                )

            if result:

                document = result.get("document", {})

                st.session_state.ui_notice = (
                    f"Successfully indexed {document.get('original_filename', uploaded_file.name)}. "
                    f"Pages: {document.get('total_pages', 'N/A')}, "
                    f"Chunks: {document.get('total_chunks', 'N/A')}."
                )

                st.rerun()

    st.divider()

    st.subheader("Uploaded documents")

    if documents:

        display_documents = [
            {
                "Filename": doc.get("original_filename"),
                "Document ID": doc.get("id"),
                "Pages": doc.get("total_pages"),
                "Chunks": doc.get("total_chunks")
            }
            for doc in documents
        ]

        st.dataframe(
            display_documents,
            hide_index=True,
            width="stretch"
        )

        st.divider()

        st.subheader("Delete a document")

        document_to_delete = st.selectbox(
            "Choose a document",
            options=list(document_lookup.keys()),
            format_func=lambda document_id: document_lookup[
                document_id
            ].get("original_filename", document_id),
            key="delete_document_selection"
        )

        if st.button(
            "Delete selected document",
            type="secondary"
        ):

            with st.spinner("Deleting document..."):

                result = api_request(
                    "DELETE",
                    f"/documents/{document_to_delete}",
                    timeout=30
                )

            if result:

                st.session_state.ui_notice = (
                    "Document deleted successfully."
                )

                st.rerun()

    else:

        st.info("No documents have been uploaded yet.")

# ==================================================
# TAB 3: SEMANTIC SEARCH
# ==================================================

with tab_search:

    st.subheader("Test semantic retrieval")

    st.write(
        "Search ChromaDB directly to inspect which document chunks "
        "are retrieved for a question."
    )

    with st.form("search_form"):

        search_question = st.text_input(
            "Enter a search query",
            placeholder="What does the document explain?"
        )

        search_submitted = st.form_submit_button(
            "Search documents"
        )

    if search_submitted:

        search_question = search_question.strip()

        if not search_question:

            st.warning("Please enter a search query.")

        else:

            request_data = {
                "question": search_question,
                "document_id": selected_document_id,
                "top_k": top_k
            }

            with st.spinner("Performing semantic search..."):

                search_result = api_request(
                    "POST",
                    "/chat/search",
                    json=request_data,
                    timeout=(10, 180)
                )

            if search_result is not None:

                st.metric(
                    "Retrieved chunks",
                    search_result.get("total_results", 0)
                )

                results = search_result.get("results", [])

                if not results:

                    st.info("No matching document chunks found.")

                for item in results:

                    title = (
                        f"Result {item['rank']} — "
                        f"{item['filename']} — "
                        f"Page {item['page_number']}"
                    )

                    with st.expander(title):

                        st.write(item["text"])

                        st.caption(
                            f"Distance: {item['distance']:.4f}"
                        )