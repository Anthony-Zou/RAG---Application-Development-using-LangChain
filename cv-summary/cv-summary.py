import os
import tempfile
from langchain.document_loaders import Docx2txtLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
import time
from dotenv import load_dotenv

# Try to load environment variables from .env file (for local development)
load_dotenv()

# Get API key from environment variable or Streamlit secrets


def get_api_key():
    # First, check for API key in Streamlit secrets (for deployment)
    if hasattr(st, 'secrets') and 'GOOGLE_API_KEY' in st.secrets:
        return st.secrets['GOOGLE_API_KEY']
    # Then, check environment variables (for local development)
    elif 'GOOGLE_API_KEY' in os.environ:
        return os.environ['GOOGLE_API_KEY']
    # If no key is found, return None
    else:
        return None


# Initialize LLM with API key
api_key = get_api_key()
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.0,
    google_api_key=api_key)


def process_docx(docx_file):
    # We need to modify this too to handle Streamlit uploads
    with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_file:
        tmp_file.write(docx_file.getvalue())
        tmp_path = tmp_file.name

    loader = Docx2txtLoader(tmp_path)
    text = loader.load_and_split()

    # Clean up the temporary file
    os.unlink(tmp_path)
    return text


def process_pdf(pdf_file):
    text = ""
    # Create a temporary file to save the uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    pages = loader.load()

    for page in pages:
        text += page.page_content
    text = text.replace('\t', ' ')

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=50,
    )

    texts = text_splitter.create_documents([text])

    # Clean up the temporary file
    os.unlink(tmp_path)

    print(len(texts))
    return texts


def main():
    # Set page configuration
    st.set_page_config(
        page_title="CareerCompass Pro",  # Updated app name
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Add custom CSS for better styling
    st.markdown("""
    <style>
    .main {
        padding: 2rem;
        background-color: #f8f9fa;
    }
    .title-container {
        background-color: #4f8bf9;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .upload-container {
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
    }
    .results-container {
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .section-header {
        color: #4f8bf9;
        font-weight: bold;
        margin-top: 1rem;
    }
    .expander-header {
        font-weight: bold !important;
        color: #2c3e50 !important;
    }
    div.stExpander > div:first-child {
        font-weight: bold;
        font-size: 1.1em;
        background-color: #f1f7fe;
        border-radius: 5px;
        padding: 0.5rem;
    }
    div.stExpander > div:nth-child(2) {
        max-height: 300px;
        overflow-y: auto;
        padding: 1rem;
        border-left: 3px solid #4CAF50;
        background-color: #fcfcfc;
    }
    .stProgress .st-c6 {
        background-color: #4f8bf9 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<div class="title-container"><h1>CareerCompass Pro</h1><p>Advanced career insights & personalized resume analysis</p></div>', unsafe_allow_html=True)

    # Sidebar for instructions and tips
    with st.sidebar:
        st.image("https://img.icons8.com/clouds/100/document.png", width=100)
        st.markdown("## CareerCompass Pro")
        st.info("""
        1. Upload your resume (PDF or DOCX)
        2. Our AI will analyze your professional profile
        3. Get personalized feedback and career advice
        """)

        st.markdown("## Tips for best results")
        st.success("""
        - Ensure your resume is up-to-date
        - Make sure the document is readable (no security restrictions)
        - Allow a minute for the AI to process your document
        """)

        st.markdown("---")
        st.caption("Powered by AI • Your data is processed securely")

    # Main content area with two columns
    col1, col2 = st.columns([1, 2])

    with col1:
        # st.markdown('<div class="upload-container">', unsafe_allow_html=True)
        st.markdown("### Upload Your Resume")
        uploaded_file = st.file_uploader(
            "", type=["pdf", "docx"], key="resume_uploader")

        if uploaded_file:
            st.success(f"Uploaded: {uploaded_file.name}")
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / 1024:.2f} KB",
                "File type": uploaded_file.type
            }
            for key, value in file_details.items():
                st.text(f"{key}: {value}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Main column for results
    with col2:
        if uploaded_file is not None:
            # st.markdown('<div class="results-container">',unsafe_allow_html=True)

            # Show progress during processing
            with st.spinner("Analyzing your resume..."):
                progress_bar = st.progress(0)

                # Process the file based on extension
                file_extension = uploaded_file.name.split(".")[-1]

                progress_bar.progress(25)
                time.sleep(0.5)  # Simulate processing time

                if file_extension == "docx":
                    text = process_docx(uploaded_file)
                elif file_extension == "pdf":
                    text = process_pdf(uploaded_file)
                else:
                    st.error(
                        "Unsupported file type. Please upload a PDF or DOCX file.")
                    st.markdown('</div>', unsafe_allow_html=True)
                    return

                progress_bar.progress(50)
                time.sleep(0.5)  # Simulate processing time

                # Setup prompts and chain
                prompt_template = """
                You are a professional CV summarizer.
                Write a verbose detail of the folllowing:
                {text}
                Details:
                """
                prompt = PromptTemplate.from_template(prompt_template)

                refine_template = (
                    "Your job is to produce a final outcome\n"
                    "We have provided an existing detail: {existing_answer}\n"
                    "We want a refined version of the existing detail based on initial details below\n"
                    "--------\n"
                    "Given the new context, refine the original summary in the following manner using proper markdown formatting:\n"
                    "## Name\n"
                    "## Email\n"
                    "## Career Assessment\n"
                    "Provide a detailed professional assessment of the candidate's profile, highlighting strengths and areas of concern. Format as bullet points.\n\n"
                    "## Resume Gaps & Improvement Areas\n"
                    "Identify specific missing elements, inconsistencies, or weak areas in the resume that could be improved. Be honest but constructive. Focus on:\n"
                    "- Missing skills relevant to their industry\n"
                    "- Unexplained career gaps\n"
                    "- Quantifiable achievements that should be added\n"
                    "- Areas where experience seems thin\n\n"
                    "## Industry Perspective\n"
                    "Analyze how the candidate's profile would be perceived by HR and hiring managers. Address:\n"
                    "- If the candidate has experience across too many different areas, explain how this appears to recruiters\n"
                    "- Whether the candidate's skills align with industry standards and expectations\n"
                    "- How the candidate compares to typical applicants in their target field\n\n"
                    "## Career Direction & Specialization\n"
                    "Based on their experience and background, recommend 1-2 specific career paths or specializations they should focus on to increase employability.\n\n"
                    "## 'Why You For This Role' Strategy\n"
                    "Provide concrete talking points the candidate should emphasize when answering the question 'Why should we hire you?' in interviews.\n\n"
                    "## Actionable Recommendations\n"
                    "Provide 3-5 specific, actionable steps the candidate should take to improve their career prospects, including:\n"
                    "- Skills they should develop to meet industry standards\n"
                    "- How to better position themselves for their target role/industry\n"
                    "- Resume and personal branding adjustments\n\n"
                    "## Last Company\n"
                    "## Experience Summary\n"
                    "\nEnsure your analysis is direct, honest, and provides genuine value for career development."
                )

                refine_prompt = PromptTemplate.from_template(refine_template)

                chain = load_summarize_chain(
                    llm=llm,
                    chain_type="refine",
                    question_prompt=prompt,
                    refine_prompt=refine_prompt,
                    return_intermediate_steps=True,
                    input_key="input_documents",
                    output_key="output_text",
                )

                progress_bar.progress(75)
                time.sleep(0.5)  # Simulate processing time

                result = chain({"input_documents": text},
                               return_only_outputs=True)

                progress_bar.progress(100)
                time.sleep(0.5)  # Simulate completion

                # Hide the progress elements
                progress_bar.empty()

                # Display results with better formatting
                st.markdown("## Your Career Analysis", unsafe_allow_html=True)

                # Parse the result into sections
                output_text = result['output_text']
                sections = output_text.split('##')

                # Display a success message
                st.success(
                    "Analysis complete! Review your personalized career guidance below.")

                # Process each section with better styling
                for section in sections:
                    if section.strip():  # Skip empty sections
                        lines = section.strip().split('\n', 1)
                        if len(lines) > 0:
                            section_title = lines[0].strip()
                            section_content = lines[1].strip() if len(
                                lines) > 1 else ""

                            # Special handling for Name and Email sections
                            if section_title.lower() == "name" or section_title.lower() == "email":
                                st.markdown(
                                    f"<h3 class='section-header'>{section_title}</h3>", unsafe_allow_html=True)
                                st.markdown(
                                    f"<p>{section_content}</p>", unsafe_allow_html=True)
                                if section_title.lower() == "email":
                                    st.markdown("<hr>", unsafe_allow_html=True)
                            else:
                                # Create an expander for other sections
                                with st.expander(f"{section_title}", expanded=False):
                                    st.markdown(section_content)

            st.markdown('</div>', unsafe_allow_html=True)

            # Add download options
            st.download_button(
                label="Download Analysis as Text",
                data=result['output_text'],
                file_name=f"career_analysis_{uploaded_file.name.split('.')[0]}.txt",
                mime="text/plain"
            )


if __name__ == "__main__":
    main()
