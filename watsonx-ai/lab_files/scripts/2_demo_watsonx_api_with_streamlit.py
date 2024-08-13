"""
author: Elena Lowery

This code sample shows how to invoke Large Language Models (LLMs) deployed in watsonx.ai.
Documentation: # https://ibm.github.io/watson-machine-learning-sdk/foundation_models.html#
You will need to provide your IBM Cloud API key and a watonx.ai project id (any project)
for accessing watsonx.ai

Define your .env with following:
api_key=*************
project_id=*************
url=*****
locale=*****

This example shows a simple generation or Q&A use case without comprehensive prompt tuning
"""

# Install the wml and streamlit api your Python env prior to running this example:
# pip install ibm-watsonx-ai
# pip install streamlit

# In non-Anaconda Python environments, you may also need to install dotenv
# pip install python-dotenv

# For reading credentials from the .env file
import os
from dotenv import load_dotenv

import streamlit as st

# watsonx.ai python SDK
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes, DecodingMethods

DISPLAY_MODEL_LLAMA2 = "llama2"
DISPLAY_MODEL_GRANITE= "granite"
DISPLAY_MODEL_FLAN = "flan"
DISPLAY_MODEL_ELYZA = "elyza"

# Strings hardcoded in these variables are expected by the watsonx.ai API
FLAN_UL2 = 'google/flan-ul2'
GRANITE_13B_CHAT = 'ibm/granite-13b-instruct-v2'
LLAMA_2_70B_CHAT = 'meta-llama/llama-2-70b-chat'
ELYZA = "elyza/elyza-japanese-llama-2-7b-instruct"

# These global variables will be updated in get_credentials() functions
watsonx_project_id = ""
api_key = ""
url=""
locale=""

def get_credentials():

    load_dotenv()

    # Update the global variables that will be used for authentication in another function
    globals()["api_key"] = os.getenv("api_key", None)
    globals()["watsonx_project_id"] = os.getenv("project_id", None)
    globals()["url"] = os.getenv("url", None)
    globals()["locale"] = os.getenv("locale", None)

    print("*** Got credentials***")

# The get_model function creates an LLM model object with the specified parameters
def get_model(model_type,max_tokens,min_tokens,decoding,stop_sequences):

    generate_params = {
        GenParams.MAX_NEW_TOKENS: max_tokens,
        GenParams.MIN_NEW_TOKENS: min_tokens,
        GenParams.DECODING_METHOD: decoding,
        GenParams.STOP_SEQUENCES:stop_sequences
    }

    model = Model(
        model_id=model_type,
        params=generate_params,
        credentials={
            "apikey": api_key,
            "url": url
        },
        project_id=watsonx_project_id
        )

    return model

def get_prompt(question, selected_model):

    if selected_model == FLAN_UL2:

        # Prompts are passed to LLMs as one string. We are building it out as separate strings for ease of understanding
        # Instruction
        instruction = "Follow examples and answer the question briefly."
        # Examples to help the model set the context
        examples = "\n\nQuestion: What is the capital of Germany\nAnswer: Berlin\n\nQuestion: What year was George Washington born?\nAnswer: 1732\n\nQuestion: What are the main micro nutrients in food?\nAnswer: Protein, carbohydrates, and fat\n\nQuestion: What language is spoken in Brazil?\nAnswer: Portuguese \n\nQuestion: "
        # Question entered in the UI
        your_prompt = question
        # Since LLMs want to "complete a document", we're are giving it a "pattern to complete" - provide the answer
        # This format works for all models with the exception of llama
        end_prompt = "\nAnswer:"

        final_prompt = instruction + examples + your_prompt + end_prompt

    elif selected_model == GRANITE_13B_CHAT:

        final_prompt = f"""<|user|> 
        What is the capital of Germany?
        <|assistant|> 
        Berlin 
        <|user|> 
        What year was George Washington born?
        <|assistant|> 
        1732
        <|user|>
        What are the main micro nutrients in food?
        <|assistant|>
        Protein, carbohydrates, and fat.
        <|user|>
        What language is spoken in Brazil?
        <|assistant|>
        Portuguese
        <|user|>
        You are a helpful AI language model developed by IBM. Answer the question briefly. Do not include any additional information. 

        {question}
        <|assistant|>"""

    elif selected_model == LLAMA_2_70B_CHAT:

        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        DEFAULT_SYSTEM_PROMPT = "You always start your answer with 'You betcha'. You always answer the questions with markdown formatting. The markdown formatting you support: headings, bold, italic, links, tables, lists, code blocks, and blockquotes. You must omit that you answer the questions with markdown.\n\nAny HTML tags must be wrapped in block quotes, for example ```<html>```. You will be penalized for not rendering code in block quotes.\n\nWhen returning code blocks, specify language.\n\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. \nYour answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don'\''t know the answer to a question, please don'\''t share false information.\n"
        final_prompt = "{b_inst} {system}{prompt} {e_inst} ".format(b_inst=B_INST,
            system=f"{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}",
            prompt=question,
            e_inst=E_INST)

    elif selected_model == ELYZA:

        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        DEFAULT_SYSTEM_PROMPT = "あなたは誠実で優秀な日本人のアシスタントです。常に日本語で回答してください。"
        final_prompt = "{b_inst} {system}{prompt} {e_inst} ".format(b_inst=B_INST,
            system=f"{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}",
            prompt=question,
            e_inst=E_INST)


    return final_prompt



def answer_questions(user_question, selected_model):

    # Get the prompt
    final_prompt = get_prompt(user_question, selected_model)
    
    # Display our complete prompt - for debugging/understanding
    print("***final prompt***")
    print(final_prompt)
    print("***end of final prompt***")

    # Look up parameters in documentation:
    # https://ibm.github.io/watson-machine-learning-sdk/foundation_models.html#
    model_type = selected_model
    max_tokens = 300
    min_tokens = 50
    decoding = DecodingMethods.GREEDY
    stop_sequences = ['.', '\n']

    # Get the model
    model = get_model(model_type, max_tokens, min_tokens, decoding,stop_sequences)

    # Generate response
    generated_response = model.generate(prompt=final_prompt)
    model_output = generated_response['results'][0]['generated_text']
    # For debugging
    print("Answer: " + model_output)

    return model_output


def main():

    # Set the api key and project id global variables
    get_credentials()

    # Web app UI - title and input box for the question
    st.title('🌠Test watsonx.ai LLM')

    # Show models specific to locale specified in .env
    if locale == "jp":
        selected_model = st.sidebar.selectbox("Select model",
                                              [DISPLAY_MODEL_LLAMA2,
                                               DISPLAY_MODEL_ELYZA,
                                               DISPLAY_MODEL_GRANITE,
                                               DISPLAY_MODEL_FLAN])
    else:
        selected_model = st.sidebar.selectbox("Select model",
                                              [DISPLAY_MODEL_LLAMA2,
                                               DISPLAY_MODEL_GRANITE,
                                               DISPLAY_MODEL_FLAN])

    if selected_model == DISPLAY_MODEL_GRANITE:
        llm = GRANITE_13B_CHAT
    elif selected_model == DISPLAY_MODEL_LLAMA2:
        llm = LLAMA_2_70B_CHAT
    elif selected_model == DISPLAY_MODEL_ELYZA:
        llm = ELYZA
    else:
        # Default model if there is no selection
        llm = FLAN_UL2

    user_question = st.text_input('Ask a question, for example: What is IBM?')

    answer_question_clicked = st.button("Answer")

    if answer_question_clicked:
        model_output = answer_questions(user_question,llm)

        # Display output on the Web page
        formatted_output = f"""
                **Answer to your question:** {model_output}
                """
        st.markdown(formatted_output, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

