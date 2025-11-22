import boto3
import json

# ---------------------------------------------------------------------
# Initialize Amazon Bedrock Runtime client
# ---------------------------------------------------------------------
# - This client is used to invoke foundation models (Titan, Claude, Llama, etc.)
# - Ensure AWS credentials & permissions are correctly configured.
client = boto3.client(
    service_name='bedrock-runtime',
    region_name="us-east-1"   # Modify if needed based on model availability
)


def get_configuration(prompt: str) -> str:
    """
    Create the JSON configuration required for Amazon Titan Text Express.

    Parameters
    ----------
    prompt : str
        The text input from the user that will be sent to the model.

    Returns
    -------
    str
        A JSON-formatted string that includes the input text and the
        text-generation parameters expected by Titan.

    Notes
    -----
    - Temperature = 0 → deterministic output (no randomness)
    - topP = 1 → standard sampling
    - maxTokenCount controls output token size
    """
    return json.dumps({
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 4096,  # Maximum tokens allowed in generated output
            "stopSequences": [],    # Titan will stop only when it finishes
            "temperature": 0,       # Ensures consistent output
            "topP": 1               # Keep sampling simple
        }
    })


# ---------------------------------------------------------------------
# Chatbot Introduction
# ---------------------------------------------------------------------
print("Bot: Hello! I am a chatbot. I can help you with anything you want to talk about.")
print("Type 'exit' to stop the conversation.\n")


# ---------------------------------------------------------------------
# Chat Loop
# ---------------------------------------------------------------------
while True:
    # Take user input
    user_input = input("User: ")

    # Exit condition
    if user_input.lower() == "exit":
        print("Bot: Goodbye! Have a great day!")
        break

    # -----------------------------------------------------------------
    # Invoke Titan model via Amazon Bedrock
    # -----------------------------------------------------------------
    response = client.invoke_model(
        body=get_configuration(user_input),
        modelId="amazon.titan-text-express-v1",
        accept="application/json",
        contentType="application/json"
    )

    # Titan sends response body as a stream → must read() then JSON parse
    response_body = json.loads(response.get("body").read())

    # Extract generated text from Titan output structure
    bot_reply = response_body.get("results")[0].get("outputText")

    print(f"Bot: {bot_reply}\n")
