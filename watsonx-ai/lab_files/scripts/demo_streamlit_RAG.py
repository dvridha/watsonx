"""
author: Elena Lowery

This code sample shows how to invoke Large Language Models (LLMs) deployed in watsonx.ai.
Documentation: # https://ibm.github.io/watson-machine-learning-sdk/foundation_models.html#
You will need to provide your IBM Cloud API key and a watonx.ai project id (any project)
for accessing watsonx.ai
This example shows a Question and Answer use case for a provided document

"""

# You may need to install these packages in your Python env
# pip install ibm-watsonx-ai
# pip install streamlit

# In non-Anaconda Python environments, you may also need to install dotenv
# pip install python-dotenv

# For reading credentials from the .env file
import os
from dotenv import load_dotenv

import streamlit as st
import fitz  # in pymupdf package
import use_case_RAG

# These global variables will be updated in get_credentials() functions
watsonx_project_id = ""
api_key = ""
url = ""

def get_credentials():

    load_dotenv()

    # Update the global variables that will be used for authentication in another function
    globals()["api_key"] = os.getenv("api_key", None)
    globals()["watsonx_project_id"] = os.getenv("project_id", None)
    globals()["url"] = os.getenv("url", None)



def main():

    # Get the API key and project id and update global variables
    get_credentials()

    # Declare variables
    file_path = ""
    collection_name = ""
    file_type = ""

    # Use the full page instead of a narrow central column
    st.set_page_config(layout="wide")

    # Streamlit app titled
    st.title("📄Chat with Documents - RAG pattern")

    # Write bold text
    st.markdown('<font color="blue"><b><i>Please upload one file type at a time and delete the file if switching types of files.</i></b></font>', unsafe_allow_html=True)

    # UI component for uploading a PDF file
    pdf_file = st.file_uploader("Upload a PDF File", type=["pdf"])

    # UI component for uploading a TXT file
    txt_file = st.file_uploader("Upload a TXT File", type=["txt"])


    # Check if a PDF file is uploaded
    if pdf_file:

        # Load the PDF content
        pdf_data = pdf_file.read()
        pdf_doc = fitz.open(stream=pdf_data, filetype="pdf")

        # Used for debugging
        print("Name of the uploaded pdf_file:" + pdf_file.name)

        # For writing file to disk
        full_file_name = pdf_file.name

        # Generate a unique collection name that follows chhoma's standards for colleciton names
        collection_name = pdf_file.name.lower()
        # Remove the .pdf
        collection_name= "pdf_" + collection_name[:-4]
        # For debugging
        print("collection_name: " + collection_name)

        # Save the pdf file in the current directory, it will be used by the module that loads data into chromadb
        pdf_doc.save(full_file_name)

        # Parameters to invoke the RAG module
        file_path = full_file_name
        file_type = use_case_RAG.FILE_TYPE_PDF

        # Close the PDF document
        pdf_doc.close()

    elif txt_file:

        # Read and display the content of the uploaded text file
        file_content = txt_file.read()
        # Convert bytes to string
        file_content = file_content.decode('latin-1')

        # Generate a unique collection name that follows chhoma's standards for colleciton names
        collection_name = txt_file.name.lower()
        # Remove the .txt
        collection_name = "txt_" + collection_name[:-4]
        # For debugging
        print("collection_name: " + collection_name)

        # Parameters to invoke the RAG module
        full_file_name = txt_file.name
        file_path = full_file_name
        file_type = use_case_RAG.FILE_TYPE_TXT

        # Define the path to the output text file
        # Write the content to the output text file
        with open(full_file_name, 'w', encoding='utf-8') as output_file:
            output_file.write(file_content)


    # UI component to enter the question
    question = st.text_area('Question',height=100)
    button_clicked = st.button("Answer the question")

    st.subheader("Response")

    # Invoke the LLM when the button is clicked
    if button_clicked:
        response = use_case_RAG.answer_questions_from_doc(api_key,watsonx_project_id,file_path,file_type,question,collection_name)
        print("Response from the LLM:" + response)
        st.write(response)

if __name__ == "__main__":
    main()


