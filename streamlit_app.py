import streamlit as st
from pypdf import PdfReader
import re
from transformers import pipeline
import torch
import os
from tempfile import NamedTemporaryFile
import base64

st.set_page_config(
    page_title="PDF Auto-Renamer",
    page_icon="📄",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the AI model - cached to prevent reloading"""
    return pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        device=0 if torch.cuda.is_available() else -1
    )

def extract_text_from_pdf(pdf_file):
    """Extract text from the first page of a PDF."""
    try:
        with NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_file_path = tmp_file.name

        reader = PdfReader(tmp_file_path)
        os.unlink(tmp_file_path)  # Clean up temp file

        if len(reader.pages) == 0:
            return None
        
        first_page = reader.pages[0]
        text = first_page.extract_text()
        return text if text.strip() else None
        
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None

def get_title_using_ai(text, summarizer):
    """Use AI to extract the most likely title."""
    try:
        summary = summarizer(
            text[:1024],
            max_length=30,
            min_length=5,
            do_sample=False
        )[0]['summary_text']
        
        title = summary.strip()
        title = re.sub(r'\s+', ' ', title)
        title = title.split('.')[0]
        return title
        
    except Exception as e:
        st.error(f"Error using AI for title extraction: {e}")
        return None

def get_binary_file_downloader_html(bin_file, file_label='File'):
    """Generate download link for file"""
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(file_label)}">Download {file_label}</a>'
    return href

def main():
    st.title("📄 PDF Auto-Renamer")
    st.write("Upload PDFs to automatically rename them based on their content using AI")
    
    # Initialize session state for file tracking
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = {}
    
    # Load AI model
    try:
        with st.spinner("Loading AI model... (this may take a minute on first run)"):
            summarizer = load_model()
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="You can upload multiple PDF files"
        )
        
        if uploaded_files:
            st.write("### Processing Files")
            
            for pdf_file in uploaded_files:
                # Check if file was already processed
                if pdf_file.name not in st.session_state.processed_files:
                    with st.spinner(f"Processing {pdf_file.name}..."):
                        # Extract text and generate title
                        text = extract_text_from_pdf(pdf_file)
                        if text:
                            title = get_title_using_ai(text, summarizer)
                            if title:
                                new_name = f"{title}.pdf"
                                st.session_state.processed_files[pdf_file.name] = {
                                    'content': pdf_file.getvalue(),
                                    'proposed_name': new_name,
                                    'original_name': pdf_file.name
                                }
                            else:
                                st.warning(f"Could not generate title for {pdf_file.name}")
                        else:
                            st.warning(f"Could not extract text from {pdf_file.name}")
            
            # Show results and allow editing
            if st.session_state.processed_files:
                st.write("### Review and Edit Names")
                st.write("You can edit the proposed names before downloading:")
                
                # Create columns for better layout
                cols = st.columns([3, 4, 2])
                with cols[0]:
                    st.write("**Original Name**")
                with cols[1]:
                    st.write("**Proposed Name**")
                
                # Display each file with editable proposed name
                for original_name, file_info in list(st.session_state.processed_files.items()):
                    cols = st.columns([3, 4, 2])
                    
                    with cols[0]:
                        st.write(original_name)
                    
                    with cols[1]:
                        new_name = st.text_input(
                            "Edit name",
                            file_info['proposed_name'],
                            key=f"input_{original_name}",
                            label_visibility="collapsed"
                        )
                        st.session_state.processed_files[original_name]['proposed_name'] = new_name
                    
                    with cols[2]:
                        # Create temporary file for download
                        with NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                            tmp_file.write(file_info['content'])
                            download_path = tmp_file.name
                        
                        st.download_button(
                            label="Download",
                            data=file_info['content'],
                            file_name=new_name,
                            mime="application/pdf"
                        )
                
                # Clear button
                if st.button("Clear All"):
                    st.session_state.processed_files = {}
                    st.experimental_rerun()
                
                # Instructions
                with st.expander("Instructions"):
                    st.write("""
                    1. Upload one or more PDF files
                    2. Wait for AI processing
                    3. Edit proposed names if needed
                    4. Download renamed files
                    5. Use 'Clear All' to start over
                    """)
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please refresh the page and try again")

if __name__ == "__main__":
    main()
