import streamlit as st
import requests

# Set page config
st.set_page_config(
    page_title="Medical RAG System",
    page_icon="📄",
    layout="wide"
)

# API configuration
API_BASE_URL = "http://localhost:8001"  

def upload_pdf():
    st.sidebar.header("Upload PDF Document")
    uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        if st.sidebar.button("Upload"):
            response = requests.post(
                f"{API_BASE_URL}/upload_pdf/",
                files={"file": (uploaded_file.name, uploaded_file, "application/pdf")}
            )

            if response.status_code == 200:
                res_data = response.json()
                document_id = res_data["document_id"]
                st.sidebar.success(f"Document processing started! ID: `{document_id}`")
                st.sidebar.info("Processing may take several minutes. Use the ID later for queries.")
            else:
                st.sidebar.error(f"Upload failed: {response.text}")

def document_browser():
    st.sidebar.header("Processed Documents")
    
    try:
        response = requests.get(f"{API_BASE_URL}/documents")
        if response.status_code == 200:
            documents = response.json().get("documents", [])
            
            if not documents:
                st.sidebar.info("No processed documents found")
                return
            
            for doc in documents:
                with st.sidebar.expander(f"Document {doc['document_id']}"):
                    st.caption(f"Uploaded: {doc['uploaded_at']}")
                    st.code(f"{doc['document_id']}")
                    st.metric("Chunks", doc["chunk_count"])
        else:
            st.sidebar.error("Failed to fetch documents")
    except Exception as e:
        st.sidebar.error(f"Connection error: {str(e)}")

def chat_interface():
    st.header("Medical Document Assistant")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask me about medical documents!"}
        ]
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("Sources"):
                    for source in message["sources"]:
                        st.caption(f"Page {source['page']} | {source['section']}")
    
    # Document ID input
    col1, col2 = st.columns([0.3, 0.7])
    with col1:
        doc_id = st.text_input("Document ID (optional)", help="Restrict search to specific document")
    
    # Chat input
    if prompt := st.chat_input("Ask a medical question..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Query API
        with st.spinner("Searching documents..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/query/",
                    json={"query": prompt, "document_id": doc_id or None}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    # Add assistant response to history
                    assistant_msg = {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": [
                            {"section": s["section"], "page": s["page"], "document_id": s["document_id"]} 
                            for s in result["sources"]
                        ]
                    }
                    st.session_state.messages.append(assistant_msg)
                    
                    # Display assistant response
                    with st.chat_message("assistant"):
                        st.markdown(result["answer"])
                        with st.expander("Sources"):
                            for source in assistant_msg["sources"]:
                                st.caption(f"Page {source['page']} | {source['section']} | ID: {source['document_id']}")
                
                else:
                    raise Exception(response.text)
                    
            except Exception as e:
                error_msg = f"Query failed: {str(e)}"
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                with st.chat_message("assistant"):
                    st.error(error_msg)

def health_check():
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            return True
    except:
        return False

def main():
    st.title("Medical Document Query System")
    
    if not health_check():
        st.error("⚠️ API service unavailable. Start FastAPI server first!")
        st.stop()
    
    upload_pdf()
    document_browser()
    chat_interface()

if __name__ == "__main__":
    main()