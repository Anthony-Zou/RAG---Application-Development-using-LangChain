# AI Career Assistant

An advanced resume analysis tool that provides professional career guidance using AI.

![AI Career Assistant Screenshot](https://img.shields.io/badge/AI%20Career-Assistant-blue)

## Overview

AI Career Assistant is a Streamlit web application that analyzes resumes (CV) and provides comprehensive feedback, career guidance, and actionable recommendations. The application leverages Google's Gemini AI model through LangChain to extract meaningful insights from resume documents.

## Features

- **Resume Analysis**: Upload PDF or DOCX resume documents for AI analysis
- **Comprehensive Career Assessment**: Get detailed professional evaluation of your profile
- **Gap Analysis**: Identify weaknesses and missing elements in your resume
- **Industry Insights**: Understand how recruiters perceive your profile
- **Career Direction**: Receive guidance on optimal career paths based on your experience
- **Interview Preparation**: Get talking points for the "Why should we hire you?" question
- **Actionable Recommendations**: Specific steps to improve your career prospects
- **Professional UI**: User-friendly interface with expandable sections and download options

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd cv-summary
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Set up your Google API key:

```bash
export GOOGLE_API_KEY="your_google_api_key"
```

## Usage

1. Run the Streamlit application:

```bash
streamlit run cv-summary.py
```

2. Open your web browser and navigate to the displayed URL (typically `http://localhost:8501`)

3. Upload your resume (PDF or DOCX format) using the file uploader

4. Wait for the analysis to complete

5. Review the detailed career guidance provided in the expandable sections

6. Optionally download the analysis as a text file for future reference

## Dependencies

- Python 3.9+
- Streamlit
- LangChain
- Google Gemini API
- PyPDF Loader
- Docx2txt

## Project Structure

```
cv-summary/
├── cv-summary.py     # Main application code
├── README.md         # Project documentation
└── requirements.txt  # Project dependencies
```

## How It Works

1. **Document Processing**: The application accepts PDF and DOCX files, which are temporarily saved and processed using PyPDFLoader or Docx2txtLoader.

2. **Text Analysis**: The document content is split into manageable chunks and processed by the Gemini AI model.

3. **Structured Output**: The AI generates a comprehensive analysis with multiple sections:

   - Name and Email identification
   - Career Assessment
   - Resume Gaps & Improvement Areas
   - Industry Perspective
   - Career Direction & Specialization
   - Interview Strategy
   - Actionable Recommendations
   - Experience Summary

4. **User Interface**: Results are displayed in an organized, expandable format for easy navigation.

## Privacy and Security

- No resume data is stored permanently; files are processed in memory and temporary files are deleted after analysis
- Your data is processed securely through the Google Gemini API
- No personal information is shared with third parties

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
