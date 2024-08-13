"""
authors: Elena Lowery and Catherine Cao

This code sample shows how to implement a simple UI for an AI Assistant application that's running in watsonx.ai
"""

# In non-Anaconda Python environments, you may also need to install dotenv
# pip install python-dotenv

# For reading credentials from the .env file
import os
from dotenv import load_dotenv

# All UI capabilities are provided by Streamlit
import streamlit as st

import chat_session
# A Python module that implements calls to LLMs
from watsonx_engine import *
from chat_session import *

TASK_GENERIC = "generic"
TASK_PROGRAMMING = "programming"

# These global variables will be updated in get_credentials() functions
watsonx_project_id = ""
api_key = ""
url = ""
space_id = ""
# Variables to hold prompt template deployment ids
classification_deployment_id=""
question_deployment_id = ""
programming_deployment_id = ""

def get_credentials():

    load_dotenv()

    # Update the global variables that will be used for authentication in another function
    globals()["api_key"] = os.getenv("api_key", None)
    globals()["watsonx_project_id"] = os.getenv("project_id", None)
    globals()["url"] = os.getenv("url", None)
    globals()["space_id"] = os.getenv("space_id", None)
    globals()["classification_deployment_id"] = os.getenv("classification_deployment_id", None)
    globals()["question_deployment_id"] = os.getenv("question_deployment_id", None)
    globals()["programming_deployment_id"] = os.getenv("programming_deployment_id", None)

def main():

    # Retrieve values required for invocation of LLMs from the .env file
    get_credentials()

    # Use the full page instead of a narrow central column
    st.set_page_config(layout="wide")

    # Streamlit UI
    st.title("AI Assistant")
    st.caption("AI Assistant powered by watsonx")

    # Streamlit saves previous messages in a list "messages".
    # Note that this is NOT memory management for LLMs, which needs to be implemented separately
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant",
                                         "content": "I am a technical AI assistant powered by watsonx."}]

    # Display previous messages in the UI
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Get the prompt from the input box in the UI
    if prompt := st.chat_input():

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # ***********************  Part 1 of the lab ****************************
        # Echo input - comment out this line after implementing the call to the LLM
        msg = "Testing UI: " + prompt

        # ***********************  Part 2 of the lab ****************************
        # This code is used in the 2nd part of the lab as we're deploying and testing prompts
        # Uncomment the next two lines after deploying the question prompt
        # current_deployment_id = question_deployment_id
        # Invoke the LLM
        # msg = invoke_prompt_template(url, api_key, space_id, current_deployment_id, prompt)

        # ***********************  Part 3 of the lab ****************************
        # Invoke the sequence of prompts - first, classification, then task
        # This code should be commented out until you deploy classification, question, and programming prompts
        # msg = generate_response(prompt)

        # ***********************  Part 4 of the lab ****************************
        # Add the prompt to chat history
        # chat_session.add_message(prompt)
        # prompt_with_history = chat_session.convert_to_prompt()
        # Invoke the LLM
        # msg = generate_response_with_history(prompt,prompt_with_history)
        # Add response to chat history
        # chat_session.add_message(msg)

        # Save the response in the Streamlit session state (for UI)
        st.session_state.messages.append({"role": "assistant", "content": msg})
        # Display the message in the UI
        st.chat_message("assistant").write(msg)

def get_deployment_id():

    # Check if the classification template has been deployed, if not, return the generic question template
    current_deployment_id = ""

    if classification_deployment_id:
        print("Classification id not null: " + classification_deployment_id)
        current_deployment_id = classification_deployment_id
    else:
        print("Classification id has not been set is null or empty")
        current_deployment_id = question_deployment_id

    return current_deployment_id

def generate_response(prompt):

    # Since we're using the same function in watsonx_engine to invoke prompt templates,
    # we need to determine which deployment ID to use

    # If the classification template has been deployed, we will use it, if not, we will use the
    # question template
    current_deployment_id = get_deployment_id()

    # Invoke the prompt to determine task type - question or programming
    task = invoke_prompt_template(url, api_key, space_id, current_deployment_id, prompt)
    # The classification prompt returns response in double quotes. We're removing them
    task_formatted = task.replace('"', '')
    print("Task type: " + task_formatted)

    # Determine which prompt to use based on task classification
    if task_formatted == TASK_GENERIC:
        current_deployment_id = question_deployment_id
        print("Assigned question deployment id")
    elif task_formatted == TASK_PROGRAMMING:
        current_deployment_id = programming_deployment_id
        print("Assigned programming deployment id")
    else:
        print("Task was not determined - missing classification prompt deployment. Using the question prompt.")
        current_deployment_id = question_deployment_id

    # Invoke the assigned deployment id (question OR programming prompt)
    response = invoke_prompt_template(url, api_key, space_id, current_deployment_id, prompt)

    return response

def generate_response_with_history(prompt,prompt_with_history):

    # Since we're using the same function in watsonx_engine to invoke prompt templates,
    # we need to determine which deployment ID to use

    # If the classification template has been deployed, we will use it, if not, we will use the
    # question template
    current_deployment_id = get_deployment_id()

    # Invoke the prompt to determine task type - question or programming
    # We do not need chat history to determine the quesiton type
    task = invoke_prompt_template(url, api_key, space_id, current_deployment_id, prompt)
    # The classification prompt returns response in double quotes. We're removing them
    task_formatted = task.replace('"', '')
    print("Task type: " + task_formatted)

    # Determine which prompt to use based on task classification
    if task_formatted == TASK_GENERIC:
        current_deployment_id = question_deployment_id
        print("Assigned question deployment id")
    elif task_formatted == TASK_PROGRAMMING:
        current_deployment_id = programming_deployment_id
        print("Assigned programming deployment id")
    else:
        print("Task was not determined - missing classification prompt deployment. Using the question prompt.")
        current_deployment_id = question_deployment_id

    # Invoke the assigned deployment id (question OR programming prompt)
    # Our prompt contains previous prompts and responses
    response = invoke_prompt_template(url, api_key, space_id, current_deployment_id, prompt_with_history)

    return response

if __name__ == "__main__":
    main()