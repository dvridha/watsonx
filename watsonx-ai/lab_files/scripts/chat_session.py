"""
authors: Elena Lowery

This code sample shows how to implement simple memory management for an
AI Assistant application that's running in watsonx.ai
"""

# Initialize the list of messages: LLM inference requests/responses
messages = []

def add_message(message):

   # Check how many messages we saved - we don't want to exceed the LLM context window (supported tokens)
   # Since we know which LLM we are using and the limit for each prompt, this approach will work for a quick prototype
   # In a production implementation, a better approach is to do token count
    if len(messages) > 10:
        drop_message()

    messages.append(message)

def convert_to_prompt():

    return ' '.join(messages)

def drop_message():

    # We are using the sliding window approach for maintaining
    # history - first in, first out (first added prompt/response is dropped)
    if messages:
        messages.pop(0)

    print("--------------------------------------------")
    print("*** Dropping messages from chat history ***")
    print("--------------------------------------------")

def print_messages():

    for message in messages:
        print(message)

