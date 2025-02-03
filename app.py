import streamlit as st
import os
from dotenv import load_dotenv
from rag_pipeline import PDFQueryEngine

# Load environment variables
load_dotenv()

def main():
    # Page config
    st.set_page_config(page_title="Resume Analyzer", layout="wide")
    
    # Title
    st.title("Resume Analyzer")
    
    # Initialize the engine
    pdf_engine = PDFQueryEngine()
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        # Job Description Input
        st.subheader("Job Description")
        job_description = st.text_area(
            "Enter the job description",
            height=300
        )

    with col2:
        # Resume Upload
        st.subheader("Upload Resume")
        resume_file = st.file_uploader("Choose a PDF file", type=['pdf'])

    # Analysis section
    if job_description and resume_file:
        if st.button("Analyze Resume"):
            with st.spinner("Analyzing..."):
                try:
                    # Save uploaded file temporarily
                    with open("temp_resume.pdf", "wb") as f:
                        f.write(resume_file.getvalue())
                    
                    # Set job description
                    pdf_engine.set_job_description(job_description)
                    
                    # Load resume
                    pdf_engine.load_pdf("temp_resume.pdf")
                    
                    # Analysis question
                    question = """Based on the job description and resume content, provide summary of the resume based on the job description.
                    Also create a table with skill sets that are asked in the job description. If the candidate has the skill, add a tick mark in the same row in the table, if not add a cross mark. 
                    Is the candidate suitable for the job based on the resume? Score the candidate out of 10 based on the resume."""
                #     question = """Based on the job description and resume content, Follow these steps:

                #     1. Skills Analysis:
                #        - First, extract ALL skills mentioned in the full content. You should be aware of what skills are. Skills means technical skills, soft skills, etc. Technical like python, java, etc. Soft skills like communication, leadership, etc.
                #        - Compare these with the required skills from the job description
                #        - Create a table with two columns:
                #          * Required Skill (from job description)
                #          * Status (Yes if found in resume, No  if missing)
                  


                #     2. Summary:
                #        - Provide a concise summary of the candidate's suitability
                #        - List top 3 strengths and areas for improvement
                #        - Make a clear hire/no hire recommendation

                #   ."""
                    
                    # Get analysis
                    analysis = pdf_engine.ask_question(question)
                    
                    # Display results
                    st.subheader("Analysis Results")
                    st.write(analysis)
                    
                    # Cleanup
                    os.remove("temp_resume.pdf")
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Please provide both job description and resume to start analysis.")

    # Add some styling
    st.markdown("""
        <style>
        .stTextArea textarea {
            background-color: #f0f2f6;
            color: black;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 24px;
            border-radius: 5px;
            border: none;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        </style>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 