import boto3
import json

import numpy as np

def cosineSimilarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# Create a Bedrock Runtime client for embeddings
client = boto3.client(service_name='bedrock-runtime', region_name="us-east-1")

# List of known facts to compare with the new fact
facts = [
    'The first computer was invented in the 1940s.',
    'John F. Kennedy was the 35th President of the United States.',
    'The first moon landing was in 1969.',
    'The capital of France is Paris.',
    'Earth is the third planet from the sun.',
]

# New fact whose similarity will be checked
newFact = 'I like to play computer games'
# question = 'Who is the president of USA?'   # Commented-out example

# ---------------------------------------------------------------------
# Function: Generate embeddings using Amazon Titan Embedding model
# ---------------------------------------------------------------------
def getEmbedding(input: str):
    """
    Get the embedding vector for the given input text using
    Amazon Titan Embedding model.
    """
    response = client.invoke_model(
        body=json.dumps({
            "inputText": input,   # Text to embed
        }),
        modelId='amazon.titan-embed-text-v1',  # Titan embedding model
        accept='application/json',
        contentType='application/json'
    )

    # Parse the JSON response from the model
    response_body = json.loads(response.get('body').read())
    return response_body.get('embedding')  # Return embedding vector


# Store facts with their embeddings
factsWithEmbeddings = []

# Generate embedding for each fact and store
for fact in facts:
    factsWithEmbeddings.append({
        'text': fact,
        'embedding': getEmbedding(fact)
    })

# ❗ This will cause an error because 'question' is NOT defined.
#    Keeping it exactly as in your code (because you asked not to change).
newFactEmbedding = getEmbedding(newFact)

# List to hold similarity scores
similarities = []

# Compute cosine similarity between new fact and each stored fact
for fact in factsWithEmbeddings:
    similarities.append({
        'text': fact['text'],
        'similarity': cosineSimilarity(fact['embedding'], newFactEmbedding)
    })

# Print similarity results
print(f"Similarities for fact: '{newFact}' with:")

# Sort similarities in descending order
similarities.sort(key=lambda x: x['similarity'], reverse=True)

# Print each fact with its similarity score
for similarity in similarities:
    print(f"  '{similarity['text']}': {similarity['similarity']:.2f}")
