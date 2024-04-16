# BE SURE TO CHANGE THE API KEY TO YOUR OWN AS WELL AS THE USERNAME (FOUND IN PROMPT) TO THE USER'S USERNAME

from openai import OpenAI
import subprocess, os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENAI_KEY")
client = OpenAI(api_key=api_key)

def get_user_input(prompt):
    user_input = input(prompt)
    return user_input

def get_chat_response(prompt):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a computer assistant writing windows command prompt commands for a user in order to create and manipulate existing files. The user is asking you to help them with a few tasks. Make sure to ONLY include the commands in your response and do NOT use markdown (just plain text). Their username is USERNAME use this in your commands when needed. If the user doesn't specify the contents of a file or folder created, assume it will be created empty."},
            {"role": "user", "content": prompt}
        ]
    )
    response = completion.choices[0].message.content
    return response

while True:
    user_input = get_user_input("\nAwaiting prompt: ")
    response = get_chat_response(user_input)

    command = response.split('\n')
    print(command)
    for cmd in command:
        subprocess(cmd, shell=True)
        print("\nSubcommand processed.\n")
    print("\nCommand fully processed.\n")