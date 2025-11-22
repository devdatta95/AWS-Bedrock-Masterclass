import boto3
import json

# ---------------------------------------------------------------------
# Initialize Amazon Bedrock Runtime client
# ---------------------------------------------------------------------
client = boto3.client(
    service_name='bedrock-runtime',
    region_name="us-east-1"
)

# ---------------------------------------------------------------------
# Helper: Build Titan input configuration with conversation history
# ---------------------------------------------------------------------
def build_prompt(conversation_history, user_input):
    """
    Construct the prompt by combining conversation history with
    the latest user message.

    Titan does NOT use chat format like Claude; instead we manually
    build a context prompt.

    Parameters
    ----------
    conversation_history : list[str]
        List storing all previous messages (bot + user).
    user_input : str
        Latest user message.

    Returns
    -------
    str
        A combined text prompt with history + new question.
    """
    # Format history as readable dialogue
    history_text = "\n".join(conversation_history)

    # Add latest question
    final_prompt = f"""
You are a helpful conversation assistant. Continue the conversation naturally.

Conversation so far:
{history_text}

User: {user_input}
Bot:
"""
    return final_prompt


def get_configuration(prompt: str) -> str:
    """
    Prepare Titan model configuration.
    """
    return json.dumps({
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 4096,
            "stopSequences": ["User:"],  # Prevent overshooting
            "temperature": 0.2,          # Slight randomness for natural conversation
            "topP": 1
        }
    })


# ---------------------------------------------------------------------
# Chatbot Introduction
# ---------------------------------------------------------------------
print("Bot: Hello! I am a chatbot. I can help you with anything you want to talk about.")
print("Type 'exit' to stop the conversation.\n")

# Conversation memory
conversation_history = ["Bot: Hello! I am a chatbot."]


# ---------------------------------------------------------------------
# Chat Loop
# ---------------------------------------------------------------------
while True:
    user_input = input("User: ")

    if user_input.lower() == "exit":
        print("Bot: Goodbye! Have a great day!")
        break

    # Build context-aware prompt
    prompt = build_prompt(conversation_history, user_input)

    # Invoke Titan model
    response = client.invoke_model(
        body=get_configuration(prompt),
        modelId="amazon.titan-text-express-v1",
        accept="application/json",
        contentType="application/json"
    )

    # Parse Titan response
    response_body = json.loads(response.get("body").read())
    bot_reply = response_body.get("results")[0].get("outputText").strip()

    # Print bot's reply
    print(f"Bot: {bot_reply}\n")

    # Update conversation memory
    conversation_history.append(f"User: {user_input}")
    conversation_history.append(f"Bot: {bot_reply}")

    # Optional: Limit history to avoid token overflow
    if len(conversation_history) > 20:
        conversation_history = conversation_history[-20:]
