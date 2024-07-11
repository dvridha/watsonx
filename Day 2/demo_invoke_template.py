"""
author: Elena Lowery

This code sample shows how to invoke prompt templates deployed in watsonx

"""

import requests, json

# Replace with your IBM Cloud API key
cloud_api_key = ''
# In most cases the URL for authentication should be this value.
# If you get an authentication error, check the URL in IBM Cloud
auth_url = 'https://iam.cloud.ibm.com/identity/token'
# Make sure to provide public, text URL (not private and not streaming)
prompt_url = ''

def get_credentials():

    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json',
    }
    data = {
        'grant_type': 'urn:ibm:params:oauth:grant-type:apikey',
        'apikey': cloud_api_key,
    }
    # Use the requests library to make the HTTP request
    response = requests.post(auth_url, headers=headers, data=data, verify=False)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the JSON response
        json_data = response.json()

        # Access the token from the JSON data
        access_token = json_data.get('access_token', None)

        if access_token:
            print(f'The access token is: {access_token}')
        else:
            print('Access token not found in the JSON response.')
    else:
        print(f'Request failed with status code: {response.status_code}')
        print(f'Response content: {response.text}')

    return access_token

def invoke_prompt(access_token):

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {access_token}",
    }

    data = {
        "parameters": {
            "prompt_variables": {
                "claim_desc": "The insured vehicle, a Tesla model X, was vandalized on March 23rd while parked in front of the insured residence on Magador Street. The vandalism included scratched paint, broken windows, and damage to the side mirrors. The insured promptly reported the incident to the police and obtained a police report. The insured is filing a claim for the repairs and any necessary replacement parts. The estimated cost of repairs has been assessed by a reputable auto repair shop."
            }
        }
    }

    # Use the requests library to make the HTTP request
    response = requests.post(prompt_url, headers=headers, json=data)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the JSON response
        generated_text = response.json()['results'][0]['generated_text']

        if generated_text:
            print(f'The generated text is: {generated_text}')
        else:
            print('Generated text not found in the JSON response.')
    else:
        print(f'Request failed with status code: {response.status_code}')
        print(f'Response content: {response.text}')

def demo_prompt_invocation():

    # Load the api key and project id
    access_token = get_credentials()

    # Show examples of 2 use cases/prompts
    invoke_prompt(access_token)

demo_prompt_invocation()
