"""
author: Elena Lowery and Catherine Cao

This code sample shows how to invoke Large Language Models (LLMs) deployed in watsonx.ai.
Documentation: # https://ibm.github.io/watson-machine-learning-sdk/foundation_models.html#
You will need to provide your IBM Cloud API key and a watonx.ai project id (any project)
for accessing watsonx.ai
This example shows a Question and Answer use case for a provided web site


# Install the wml api your Python env prior to running this example:
# pip install ibm-watsonx-ai

# Install chroma
# pip install chromadb

IMPORTANT: Be aware of the disk space that will be taken up by documents when they're loaded into
chromadb on your laptop. The size in chroma will likely be the same as .txt file size
"""

# For reading credentials from the .env file
import os
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from chromadb.api.types import EmbeddingFunction

# WML python SDK
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes, DecodingMethods

import requests
from bs4 import BeautifulSoup
import spacy
import chromadb
import en_core_web_md

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

# The get_model function creates an LLM model object with the specified parameters

def get_model(model_type, max_tokens, min_tokens, decoding, temperature, top_k, top_p):
    generate_params = {
        GenParams.MAX_NEW_TOKENS: max_tokens,
        GenParams.MIN_NEW_TOKENS: min_tokens,
        GenParams.DECODING_METHOD: decoding,
        GenParams.TEMPERATURE: temperature,
        GenParams.TOP_K: top_k,
        GenParams.TOP_P: top_p,
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


def get_model_test(model_type, max_tokens, min_tokens, decoding, temperature):
    generate_params = {
        GenParams.MAX_NEW_TOKENS: max_tokens,
        GenParams.MIN_NEW_TOKENS: min_tokens,
        GenParams.DECODING_METHOD: decoding,
        GenParams.TEMPERATURE: temperature
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


# Embedding function
class MiniLML6V2EmbeddingFunction(EmbeddingFunction):
    MODEL = SentenceTransformer('all-MiniLM-L6-v2')

    def __call__(self, texts):
        return MiniLML6V2EmbeddingFunction.MODEL.encode(texts).tolist()


def extract_text(url):
    try:
        # Send an HTTP GET request to the URL
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the HTML content of the page using BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract contents of <p> elements
            p_contents = [p.get_text() for p in soup.find_all('p')]

            # Print the contents of <p> elements
            print("\nContents of <p> elements: \n")
            for content in p_contents:
                print(content)
            raw_web_text = " ".join(p_contents)
            # remove \xa0 which is used in html to avoid words break acorss lines.
            cleaned_text = raw_web_text.replace("\xa0", " ")
            return cleaned_text

        else:
            print(f"Failed to retrieve the page. Status code: {response.status_code}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


def split_text_into_sentences(text):
    nlp = spacy.load("en_core_web_md")
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    cleaned_sentences = [s.strip() for s in sentences]
    return cleaned_sentences


def create_embedding(url, collection_name):
    cleaned_text = extract_text(url)
    cleaned_sentences = split_text_into_sentences(cleaned_text)

    client = chromadb.Client()

    collection = client.get_or_create_collection(collection_name)

    # Upload text to chroma
    collection.upsert(
        documents=cleaned_sentences,
        metadatas=[{"source": str(i)} for i in range(len(cleaned_sentences))],
        ids=[str(i) for i in range(len(cleaned_sentences))],
    )

    return collection


def create_prompt(url, question, collection_name):
    # Create embeddings for the text file
    collection = create_embedding(url, collection_name)

    # query relevant information
    relevant_chunks = collection.query(
        query_texts=[question],
        n_results=5,
    )
    context = "\n\n\n".join(relevant_chunks["documents"][0])
    # Please note that this is a generic format. You can change this format to be specific to llama
    prompt = (f"{context}\n\nPlease answer the following question in one sentence using this "
              + f"text. "
              + f"If the question is unanswerable, say \"unanswerable\". Do not include information that's not relevant to the question."
              + f"Question: {question}")

    return prompt


def main():

    # Get the API key and project id and update global variables
    get_credentials()

    # Try diffrent URLs and questions
    url = "https://www.usbank.com/financialiq/manage-your-household/buy-a-car/own-electric-vehicles-learned-buying-driving-EVs.html"

    question = "What are the incentives for purchasing EVs?"
    # question = "What is the percentage of driving powered by hybrid cars?"
    # question = "Can an EV be plugged in to a household outlet?"
    collection_name = "test_web_RAG"

    answer_questions_from_web(api_key, watsonx_project_id, url, question, collection_name)


def answer_questions_from_web(request_api_key, request_project_id, url, question, collection_name):

    # Retrieve variables for invoking llms
    get_credentials()

    # Update the global variable
    globals()["api_key"] = request_api_key
    globals()["watsonx_project_id"] = request_project_id

    # Specify model parameters
    model_type = "meta-llama/llama-2-70b-chat"
    max_tokens = 100
    min_tokens = 50
    top_k = 50
    top_p = 1
    decoding = DecodingMethods.GREEDY
    temperature = 0.7

    # Get the watsonx model = try both options
    model = get_model(model_type, max_tokens, min_tokens, decoding, temperature, top_k, top_p)

    # Get the prompt
    complete_prompt = create_prompt(url, question, collection_name)

    # Let's review the prompt
    print("----------------------------------------------------------------------------------------------------")
    print("*** Prompt:" + complete_prompt + "***")
    print("----------------------------------------------------------------------------------------------------")

    generated_response = model.generate(prompt=complete_prompt)
    response_text = generated_response['results'][0]['generated_text']

    # Remove trailing white spaces
    response_text = response_text.strip()

    # print model response
    print("--------------------------------- Generated response -----------------------------------")
    print(response_text)
    print("*********************************************************************************************")

    return response_text


# Invoke the main function
if __name__ == "__main__":
    main()
