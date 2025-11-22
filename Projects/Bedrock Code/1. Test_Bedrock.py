import boto3
import pprint

# ---------------------------------------------------------------------
# Initialize Bedrock client
# ---------------------------------------------------------------------
# Creating a client object to interact with Amazon Bedrock.
# Make sure your AWS credentials and region are correctly configured.
bedrock = boto3.client(
    service_name='bedrock',
    region_name='us-east-1'  # Change region as per your Bedrock setup
)

# PrettyPrinter for clean console output
pp = pprint.PrettyPrinter(depth=4)


def list_foundation_models():
    """
    Retrieve and print all available foundation models in the current AWS region.

    This function uses the Bedrock client's `list_foundation_models()` method
    to fetch a list of all models that are accessible to the account in the
    selected region. Each model's metadata is printed in a structured format.

    Useful when:
    - You want to explore which models are available.
    - You need to identify the correct model identifier to use.
    """
    models = bedrock.list_foundation_models()

    # Iterate over each model summary and print details
    for model in models.get("modelSummaries", []):
        pp.pprint(model)            # Print complete model information
        pp.pprint("-------------------")  # Separator for readability


def get_foundation_model(modelIdentifier):
    """
    Fetch details of a specific foundation model from Amazon Bedrock.

    Parameters
    ----------
    modelIdentifier : str
        The exact Bedrock model ID (e.g., 'amazon.nova-lite-v1:0',
        'anthropic.claude-3-sonnet-20240229-v1:0', etc.)

    This function calls the `get_foundation_model()` Bedrock API and prints
    the model details. If an invalid identifier is passed, AWS will raise a
    ValidationException.

    Example:
        get_foundation_model('amazon.nova-lite-v1:0')
    """
    # Call Bedrock to retrieve model information
    model = bedrock.get_foundation_model(modelIdentifier=modelIdentifier)

    # Print the response in a readable format
    pp.pprint(model)


# ---------------------------------------------------------------------
# Example Usage
# ---------------------------------------------------------------------

# Uncomment the line below to list all available models
# list_foundation_models()

# Fetch details for a specific Foundation Model
get_foundation_model('amazon.nova-lite-v1:0')
