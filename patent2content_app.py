import streamlit as st
import base64
import fitz  # PyMuPDF
import google.generativeai as genai
import os
from datetime import date
from streamlit_pdf_viewer import pdf_viewer # Import the component
import time # For potential rerun delays if needed

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Patent Insights + AI Summary")

# --- Initialize Session State ---
# Initialize patent-specific keys and remove legal ones
default_session_state = {
    "patent_details": { # New section for patent metadata
        "patent_number": "",
        "title": "",
        "inventors": "",
        "assignee": "",
        "filing_date": None,
        "publication_date": None,
    },
    "user_notes": "", # Simple text area for user analysis
    "pdf_text": None,
    "summary": None,
    "gemini_configured": False,
    "gemini_error": None,
    # 'user_gemini_key' is no longer needed as we only use secrets
    "uploaded_file_id": None,
}
for key, default_value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# --- Sidebar ---
st.sidebar.title("Configuration & Upload")

# --- Configure Gemini API using Streamlit Secrets ONLY ---
st.sidebar.subheader("API Status")
gemini_model = None
api_key_to_use = None
# Reset config status at the start of each run, re-check secrets
st.session_state.gemini_configured = False
st.session_state.gemini_error = None

try:
    # Attempt to get the key ONLY from Streamlit secrets
    api_key_to_use = st.secrets["GEMINI_API_KEY"]

    # Configure if key found
    if api_key_to_use:
        try:
            # Configure genai with the key
            genai.configure(api_key=api_key_to_use)
            # Initialize the model instance
            gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest') # Or your chosen model
            st.session_state.gemini_configured = True
            st.session_state.gemini_error = None
            st.sidebar.success("Gemini API Configured via Secrets.", icon="✅")
        except Exception as e:
            # Handle errors during configuration (e.g., invalid key format)
            st.sidebar.error(f"Failed to configure Gemini: {e}", icon=" R⚠️")
            st.session_state.gemini_configured = False
            st.session_state.gemini_error = str(e)
            gemini_model = None # Ensure model is None on error
    else:
         # This case should ideally not happen if key is set in secrets, but good practice
         st.sidebar.warning("Gemini API Key is missing or empty in Streamlit Secrets.", icon=" R❓")
         st.session_state.gemini_configured = False

except KeyError:
    # Key 'GEMINI_API_KEY' not found in st.secrets
    st.sidebar.error("Required secret 'GEMINI_API_KEY' not found. Please add it in the Streamlit Cloud app settings.", icon=" R🔑")
    st.session_state.gemini_configured = False
except Exception as e:
     # Catch any other unexpected errors accessing secrets or configuring
     st.sidebar.error(f"An error occurred during API setup: {e}", icon=" R🔥")
     st.session_state.gemini_configured = False
     st.session_state.gemini_error = str(e)


# --- File Upload ---
st.sidebar.subheader("Upload Patent PDF")
uploaded_file = st.sidebar.file_uploader(
    "Choose a Patent PDF file", type="pdf", key="pdf_uploader_patent",
    on_change=lambda: ( # Reset relevant state when file changes/is removed
        st.session_state.update({
            "pdf_text": None, "summary": None, "uploaded_file_id": None,
            # Optionally reset manual fields too if desired
            # "patent_details": default_session_state["patent_details"],
            # "user_notes": default_session_state["user_notes"],
        })
    ) if st.session_state.pdf_uploader_patent is None else None
)

# --- Helper Functions ---
def extract_text_from_pdf(file_bytes):
    """Extracts text from PDF bytes using PyMuPDF."""
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            text = ""
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}", icon="📄")
        return None

def summarize_patent_with_gemini(text_to_summarize, model):
    """Generates a patent summary using the provided Gemini model instance."""
    # Check configuration status from session state
    if not st.session_state.gemini_configured or model is None:
        st.warning("Gemini not configured. Cannot summarize.", icon="⚠️")
        return None # Return None if not configured
    if not text_to_summarize or len(text_to_summarize) < 100:
         st.warning("Not enough text extracted from PDF to summarize effectively.", icon="⚠️")
         return None

    max_chars = 100000 # Example limit, adjust as needed
    truncated_text = text_to_summarize[:max_chars]
    if len(text_to_summarize) > max_chars:
        st.info(f"Summarizing the first {max_chars:,} characters due to length limit.", icon="ℹ️")

    prompt = f"""
    Please act as a patent analyst and provide a concise summary of the following patent document text. Focus on these key aspects:

    1.  **Problem Solved:** Briefly describe the technical problem or need the invention aims to address, as stated in the background or summary sections.
    2.  **Core Invention/Solution:** Explain the main technical concept, mechanism, or process disclosed. What is the essence of the invention described? Focus on the primary innovation outlined in the summary/detailed description.
    3.  **Key Features/Advantages:** Highlight 1-3 key distinguishing features, components, steps, or advantages mentioned in the description that characterize the invention.
    4.  **Field of Invention:** Briefly state the technical field this invention belongs to, if readily apparent.

    Keep the summary factual, objective, and focused on the technical disclosure. Avoid interpreting claim scope or providing legal opinions. Use clear language suitable for someone understanding the technology. Aim for 2-4 paragraphs.

    Patent Text:
    ---
    {truncated_text}
    ---
    Patent Summary:
    """
    try:
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        response = model.generate_content(prompt, safety_settings=safety_settings)

        if response.parts:
             return response.text
        else:
             block_reason = "Unknown"
             safety_ratings = "N/A"
             try: # Safely access feedback details
                 if response.prompt_feedback: block_reason = response.prompt_feedback.block_reason or "Not specified"
                 if response.candidates and response.candidates[0].safety_ratings: safety_ratings = response.candidates[0].safety_ratings
             except Exception: pass
             st.error(f"Summary generation failed or was blocked. Reason: {block_reason}. Safety Ratings: {safety_ratings}", icon="🚫")
             return None

    except Exception as e:
        st.error(f"Gemini API Error during generation: {e}", icon="🔥")
        # Add more specific error checks if needed
        if "API key not valid" in str(e): st.error("The API Key configured in Secrets appears invalid.", icon="🔑")
        elif "quota" in str(e).lower(): st.error("You may have exceeded your Gemini API quota.", icon="📊")
        return None

# --- Main App Area ---
st.title("📄 Patent Insights with AI Summary")
st.write("Upload a Patent PDF to view it and generate an AI-powered summary using the configured API key.")

if uploaded_file is not None:
    current_file_id = uploaded_file.file_id

    # Process PDF text only once per new file upload
    if st.session_state.pdf_text is None or current_file_id != st.session_state.uploaded_file_id:
        with st.spinner("Reading and extracting text from PDF..."):
            uploaded_file.seek(0) # Reset buffer before reading
            pdf_bytes_for_text = uploaded_file.read()
            st.session_state.pdf_text = extract_text_from_pdf(pdf_bytes_for_text)
            st.session_state.summary = None # Reset summary for new file
            st.session_state.uploaded_file_id = current_file_id
            if st.session_state.pdf_text:
                st.sidebar.success(f"PDF text extracted ({len(st.session_state.pdf_text):,} chars).", icon="📄")
            else:
                st.sidebar.error("Failed to extract text. Cannot generate summary.", icon="⚠️")

    # --- Layout: PDF Viewer and Data Entry/Summary Side-by-Side ---
    col1, col2 = st.columns([3, 2]) # Adjust ratio if needed

    with col1:
        st.header("📄 Patent PDF Viewer")
        try:
            uploaded_file.seek(0) # Ensure buffer is at the start for viewer
            pdf_bytes_for_viewer = uploaded_file.read()
            with st.container(): # Use container to control height better if needed
                 pdf_viewer(input=pdf_bytes_for_viewer, width=700, height=800)
        except Exception as e:
            st.error(f"Could not display PDF viewer: {e}", icon="🖼️")


    with col2:
        st.header("📊 Analysis & Details")

        # --- AI Summary Section ---
        st.subheader("🤖 AI Patent Summary")
        # Determine if the button should be disabled
        summarize_button_disabled = not st.session_state.gemini_configured or not st.session_state.pdf_text
        summarize_help_text = "Gemini API not configured or PDF text not extracted." if summarize_button_disabled else "Generates a summary using the configured Gemini API key."

        if st.button("Generate Patent Summary", disabled=summarize_button_disabled, help=summarize_help_text):
            # Check config and text again right before calling
            if st.session_state.gemini_configured and st.session_state.pdf_text and gemini_model:
                 with st.spinner("✨ Generating summary... Please wait."):
                     summary_result = summarize_patent_with_gemini(st.session_state.pdf_text, gemini_model)
                     if summary_result:
                          st.session_state.summary = summary_result
                     else:
                          st.warning("Summary generation failed or was blocked. Check errors above.", icon="⚠️")
            elif not st.session_state.pdf_text:
                 st.warning("Cannot generate summary - PDF text missing.", icon="📄")
            else: # Should only happen if config failed unexpectedly
                 st.error("Cannot generate summary - Gemini not configured.", icon="⚙️")

        # Display Summary or Informational Messages
        if st.session_state.summary:
            st.markdown("**Generated Summary:**")
            st.markdown(f"> {st.session_state.summary}", unsafe_allow_html=True)
            st.markdown("---")
        # Display specific messages based on why the button might be disabled
        elif summarize_button_disabled and not st.session_state.gemini_configured:
             st.warning("Cannot generate summary. Ensure 'GEMINI_API_KEY' is correctly set in app Secrets.", icon="🔑")
        elif summarize_button_disabled and not st.session_state.pdf_text and uploaded_file:
             st.warning("Cannot generate summary because text extraction failed.", icon="📄")
        # If button is enabled but no summary yet (initial state)
        elif not summarize_button_disabled and not st.session_state.summary:
             st.info("Click the button above to generate an AI patent summary.", icon="🤖")


        # --- Manual Entry Sections for Patent ---
        st.subheader("📝 Manual Entry / Analysis")
        with st.expander("Patent Details (Manual)", expanded=False):
            # Use .get for safer access to dictionary keys in session state
            st.session_state.patent_details['patent_number'] = st.text_input("Patent Number", value=st.session_state.patent_details.get('patent_number', ''), key="patent_num")
            st.session_state.patent_details['title'] = st.text_input("Patent Title", value=st.session_state.patent_details.get('title', ''), key="patent_title")
            st.session_state.patent_details['inventors'] = st.text_input("Inventor(s)", value=st.session_state.patent_details.get('inventors', ''), key="patent_inventors")
            st.session_state.patent_details['assignee'] = st.text_input("Assignee/Applicant", value=st.session_state.patent_details.get('assignee', ''), key="patent_assignee")
            st.session_state.patent_details['filing_date'] = st.date_input("Filing Date", value=st.session_state.patent_details.get('filing_date', None), key="patent_filing_date")
            st.session_state.patent_details['publication_date'] = st.date_input("Publication Date", value=st.session_state.patent_details.get('publication_date', None), key="patent_pub_date")

        with st.expander("User Notes / Analysis", expanded=False):
             st.session_state.user_notes = st.text_area(
                 "Your Notes (e.g., key claims, relevance, novelty assessment)",
                 value=st.session_state.get('user_notes', ''),
                 height=150,
                 key="user_patent_notes"
            )

else:
    st.info("☝️ Upload a Patent PDF file using the sidebar to get started.")
    # Clear specific state when no file is loaded, keeps API config status
    # st.session_state.pdf_text = None
    # st.session_state.summary = None
    # st.session_state.uploaded_file_id = None