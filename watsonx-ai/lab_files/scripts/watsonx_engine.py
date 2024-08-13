"""
authors: Elena Lowery and Catherine Cao

This code sample shows how to implement a simple business logic layer for an
AI Assistant application that's running in watsonx.ai
"""

from ibm_watsonx_ai import APIClient

def invoke_prompt_template(url,api_key,space_id, deployment_id,task):

    credentials = {
        "url": url,
        "apikey": api_key
    }

    client = APIClient(credentials)
    client.set.default_space(space_id)

    generated_response = client.deployments.generate_text(deployment_id,params={"prompt_variables": {"task": task}})

    print("--------------------------Invocation of a prompt template -------------------------------------------")
    print("Task: " + task)
    print("Response: " + generated_response)
    print("------------------------------------------------------------------------------------------------------")

    return generated_response
