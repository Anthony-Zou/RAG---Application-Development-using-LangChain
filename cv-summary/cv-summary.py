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
import uuid
from datetime import datetime, timedelta

# Try to load environment variables from .env file (for local development)
load_dotenv()

# Get API key from environment variable or Streamlit secrets


def get_api_key():

    # Then, check environment variables (for local development)
    if 'GOOGLE_API_KEY' in os.environ:
        return os.environ['GOOGLE_API_KEY']
        # First, check for API key in Streamlit secrets (for deployment)
    elif hasattr(st, 'secrets') and 'GOOGLE_API_KEY' in st.secrets:
        return st.secrets['GOOGLE_API_KEY']
    # If no key is found, return None
    else:
        return None


# Initialize LLM with API key
api_key = get_api_key()
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.0,
    google_api_key=api_key)

# Initialize session state for user authentication and subscription
if 'user_authenticated' not in st.session_state:
    st.session_state['user_authenticated'] = False
if 'subscription_active' not in st.session_state:
    st.session_state['subscription_active'] = False
if 'subscription_end_date' not in st.session_state:
    st.session_state['subscription_end_date'] = None
if 'user_email' not in st.session_state:
    st.session_state['user_email'] = None
if 'user_id' not in st.session_state:
    st.session_state['user_id'] = None
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = "login"


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


# New function for resume generation
def generate_improved_resume(resume_text, target_job=None):
    prompt_template = """
    You are a professional resume writer with expertise in creating impactful, ATS-friendly resumes.
    
    Based on the following resume content:
    {resume_text}
    
    Create an improved, professionally formatted resume that:
    1. Emphasizes key achievements and quantifiable results
    2. Uses strong action verbs
    3. Optimizes for ATS systems
    4. Follows modern resume best practices
    5. Maintains all original information but presents it more effectively
    
    {job_target_info}
    
    Format the resume sections with proper markdown and ensure it's ready for professional use.
    """

    job_target_text = ""
    if target_job:
        job_target_text = f"Target the resume specifically for this job description or industry: {target_job}"

    prompt = PromptTemplate.from_template(prompt_template).format(
        resume_text=resume_text,
        job_target_info=job_target_text
    )

    response = llm.invoke(prompt)
    return response.content


# New function for cover letter generation
def generate_cover_letter(resume_text, job_description, company_info):
    prompt_template = """
    You are a professional cover letter writer with expertise in creating compelling, personalized cover letters.
    
    Based on:
    
    RESUME:
    {resume_text}
    
    JOB DESCRIPTION:
    {job_description}
    
    COMPANY INFORMATION:
    {company_info}
    
    Create a compelling cover letter that:
    1. Is personalized to the specific job and company
    2. Highlights relevant experience from the resume that matches the job description
    3. Demonstrates understanding of the company's values and goals
    4. Uses a professional but engaging tone
    5. Includes a strong opening and closing
    6. Is between 250-350 words
    
    Format the cover letter professionally with proper salutation and signature.
    """

    prompt = PromptTemplate.from_template(prompt_template).format(
        resume_text=resume_text,
        job_description=job_description,
        company_info=company_info
    )

    response = llm.invoke(prompt)
    return response.content


# Mock user database (replace with actual database in production)
MOCK_USERS = {
    "user@example.com": {"password": "password123", "id": "user123"}
}


# Mock subscription validation (replace with actual payment processor integration)
def validate_subscription(user_id, plan_type):
    # In a real app, call payment processor API to validate payment
    # For demo, always return success
    today = datetime.now()

    if plan_type == "2-week":
        end_date = today + timedelta(days=14)
    elif plan_type == "monthly":
        end_date = today + timedelta(days=30)
    elif plan_type == "annual":
        end_date = today + timedelta(days=365)
    else:
        return False, None

    return True, end_date


# Login page
def show_login_page():
    st.markdown('<div class="title-container"><h1>CareerCompass Pro</h1><p>Login to access premium career tools</p></div>', unsafe_allow_html=True)

    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

        if submit:
            if email in MOCK_USERS and MOCK_USERS[email]["password"] == password:
                st.session_state['user_authenticated'] = True
                st.session_state['user_email'] = email
                st.session_state['user_id'] = MOCK_USERS[email]["id"]
                st.session_state['current_page'] = "dashboard"
                st.rerun()
            else:
                st.error("Invalid email or password")

    st.markdown("---")
    if st.button("Create an account"):
        st.session_state['current_page'] = "signup"
        st.rerun()


# Signup page
def show_signup_page():
    st.markdown('<div class="title-container"><h1>CareerCompass Pro</h1><p>Create your account</p></div>',
                unsafe_allow_html=True)

    with st.form("signup_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submit = st.form_submit_button("Sign Up")

        if submit:
            if password != confirm_password:
                st.error("Passwords do not match")
            elif email in MOCK_USERS:
                st.error("Email already registered")
            else:
                # In a real app, securely hash the password before storing
                MOCK_USERS[email] = {
                    "password": password, "id": str(uuid.uuid4())}
                st.success("Account created! You can now log in.")
                st.session_state['current_page'] = "login"
                st.rerun()

    st.markdown("---")
    if st.button("Already have an account? Login"):
        st.session_state['current_page'] = "login"
        st.rerun()


# Subscription page
def show_subscription_page():
    st.markdown('<div class="title-container"><h1>Choose Your Plan</h1><p>Select a subscription that fits your job search timeline</p></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="border:1px solid #ddd; padding: 20px; border-radius: 10px; height: 360px;">
            <h3>2-Week Access</h3>
            <h2>$14.99</h2>
            <p>Perfect for quick job applications</p>
            <ul>
                <li>Resume Analysis & Optimization</li>
                <li>Cover Letter Generation</li>
                <li>Interview Preparation</li>
                <li>2-Week Access</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Select 2-Week Plan"):
            success, end_date = validate_subscription(
                st.session_state['user_id'], "2-week")
            if success:
                st.session_state['subscription_active'] = True
                st.session_state['subscription_end_date'] = end_date
                st.session_state['current_page'] = "dashboard"
                st.rerun()

    with col2:
        st.markdown("""
        <div style="border:1px solid #4f8bf9; padding: 20px; border-radius: 10px; background-color: #f8f9fe; height: 360px;">
            <h3>Monthly Access</h3>
            <h2>$29.99</h2>
            <p><strong>Most Popular</strong></p>
            <ul>
                <li>Resume Analysis & Optimization</li>
                <li>Cover Letter Generation</li>
                <li>Interview Preparation</li>
                <li>30-Day Access</li>
                <li>Priority Support</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Select Monthly Plan"):
            success, end_date = validate_subscription(
                st.session_state['user_id'], "monthly")
            if success:
                st.session_state['subscription_active'] = True
                st.session_state['subscription_end_date'] = end_date
                st.session_state['current_page'] = "dashboard"
                st.rerun()

    with col3:
        st.markdown("""
        <div style="border:1px solid #ddd; padding: 20px; border-radius: 10px; height: 360px;">
            <h3>Annual Access</h3>
            <h2>$199.99</h2>
            <p>Best Value</p>
            <ul>
                <li>Resume Analysis & Optimization</li>
                <li>Cover Letter Generation</li>
                <li>Interview Preparation</li>
                <li>365-Day Access</li>
                <li>Priority Support</li>
                <li>Job Search Tracking</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Select Annual Plan"):
            success, end_date = validate_subscription(
                st.session_state['user_id'], "annual")
            if success:
                st.session_state['subscription_active'] = True
                st.session_state['subscription_end_date'] = end_date
                st.session_state['current_page'] = "dashboard"
                st.rerun()


# Dashboard
def show_dashboard():
    st.sidebar.success(f"Logged in as: {st.session_state['user_email']}")
    if st.session_state['subscription_end_date']:
        days_left = (
            st.session_state['subscription_end_date'] - datetime.now()).days
        st.sidebar.info(f"Subscription active: {days_left} days remaining")

    st.sidebar.markdown("## Navigation")
    if st.sidebar.button("Resume Analysis"):
        st.session_state['current_page'] = "resume_analysis"
        st.rerun()

    if st.sidebar.button("Resume Generator"):
        st.session_state['current_page'] = "resume_generator"
        st.rerun()

    if st.sidebar.button("Cover Letter Generator"):
        st.session_state['current_page'] = "cover_letter_generator"
        st.rerun()

    if st.sidebar.button("Log Out"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()

    st.markdown('<div class="title-container"><h1>Welcome to CareerCompass Pro</h1><p>Your all-in-one career toolkit</p></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="border:1px solid #ddd; padding: 20px; border-radius: 10px; text-align: center; cursor: pointer;">
            <h3>Resume Analysis</h3>
            <p>Get detailed feedback on your current resume</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="border:1px solid #ddd; padding: 20px; border-radius: 10px; text-align: center; cursor: pointer;">
            <h3>Resume Generator</h3>
            <p>Create an optimized version of your resume</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="border:1px solid #ddd; padding: 20px; border-radius: 10px; text-align: center; cursor: pointer;">
            <h3>Cover Letter Generator</h3>
            <p>Generate targeted cover letters</p>
        </div>
        """, unsafe_allow_html=True)


# Resume Generator Page
def show_resume_generator():
    st.markdown('<div class="title-container"><h1>Resume Generator</h1><p>Create an optimized version of your resume</p></div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload your current resume", type=["pdf", "docx"])
    target_job = st.text_area(
        "Enter the job title or description you're targeting (optional)", height=100)

    if uploaded_file and st.button("Generate Improved Resume"):
        with st.spinner("Analyzing and improving your resume..."):
            if uploaded_file.name.endswith('.docx'):
                text = process_docx(uploaded_file)
                resume_text = "\n".join([doc.page_content for doc in text])
            else:
                text = process_pdf(uploaded_file)
                resume_text = "\n".join([doc.page_content for doc in text])

            improved_resume = generate_improved_resume(resume_text, target_job)

            st.success("Resume successfully improved!")
            with st.expander("View Improved Resume", expanded=True):
                st.markdown(improved_resume)

            # Download options
            st.download_button(
                label="Download Improved Resume as Markdown",
                data=improved_resume,
                file_name="improved_resume.md",
                mime="text/markdown"
            )


# Cover Letter Generator Page
def show_cover_letter_generator():
    st.markdown('<div class="title-container"><h1>Cover Letter Generator</h1><p>Create targeted cover letters for specific job applications</p></div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload your resume", type=["pdf", "docx"])
    job_description = st.text_area("Paste the job description", height=150)
    company_info = st.text_area(
        "Enter information about the company (culture, values, mission, etc.)", height=100)

    if uploaded_file and job_description and st.button("Generate Cover Letter"):
        with st.spinner("Creating your personalized cover letter..."):
            if uploaded_file.name.endswith('.docx'):
                text = process_docx(uploaded_file)
                resume_text = "\n".join([doc.page_content for doc in text])
            else:
                text = process_pdf(uploaded_file)
                resume_text = "\n".join([doc.page_content for doc in text])

            cover_letter = generate_cover_letter(
                resume_text, job_description, company_info)

            st.success("Cover letter successfully generated!")
            with st.expander("View Cover Letter", expanded=True):
                st.markdown(cover_letter)

            # Download options
            st.download_button(
                label="Download Cover Letter",
                data=cover_letter,
                file_name="cover_letter.md",
                mime="text/markdown"
            )


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

    # Check if API key is available
    if not api_key:
        st.error(
            "Google API Key not found. Please set up your API key in Streamlit secrets or environment variables.")
        st.info("For local development: Add GOOGLE_API_KEY to your .env file\nFor Streamlit deployment: Add GOOGLE_API_KEY to your app secrets")
        return

    # Route to the appropriate page based on auth status and current page
    if not st.session_state['user_authenticated']:
        if st.session_state['current_page'] == "signup":
            show_signup_page()
        else:
            show_login_page()
    else:
        if not st.session_state['subscription_active']:
            show_subscription_page()
        else:
            if st.session_state['current_page'] == "dashboard":
                show_dashboard()
            elif st.session_state['current_page'] == "resume_analysis":
                # Original functionality
                show_resume_analysis()
            elif st.session_state['current_page'] == "resume_generator":
                show_resume_generator()
            elif st.session_state['current_page'] == "cover_letter_generator":
                show_cover_letter_generator()


# Rename the original functionality to show_resume_analysis
def show_resume_analysis():
    # Header
    st.markdown('<div class="title-container"><h1>Resume Analysis</h1><p>Get detailed feedback on your current resume</p></div>', unsafe_allow_html=True)

    # Main content area with two columns
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('<div class="upload-container">', unsafe_allow_html=True)
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
