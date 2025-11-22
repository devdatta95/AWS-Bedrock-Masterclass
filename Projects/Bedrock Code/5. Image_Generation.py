import boto3
import json
import base64

# Initialize Bedrock Runtime client
# This client is used to call models like Titan Image Generator
client = boto3.client(service_name='bedrock-runtime', region_name="us-east-1")

# Configuration for Titan Image Generator model
# - taskType: TEXT_IMAGE (generate image from text)
# - textToImageParams: contains the prompt
# - imageGenerationConfig: defines size, number of images, cfgScale, etc.
stability_image_config = json.dumps({
    "taskType": "TEXT_IMAGE",
    "textToImageParams": {
        "text": "cat on a mat on a country hillside",  # Image description
    },
    "imageGenerationConfig": {
        "numberOfImages": 1,   # Number of images to generate
        "height": 512,         # Image height
        "width": 512,          # Image width
        "cfgScale": 8.0,       # Controls how strongly the image follows the prompt
    }
})

# Invoke the Titan Image Generator model on Bedrock
response = client.invoke_model(
    body=stability_image_config,
    modelId="amazon.titan-image-generator-v1",
    accept="application/json",
    contentType="application/json"
)

# Convert model response (JSON body) to Python dictionary
response_body = json.loads(response.get("body").read())

# Extract Base64 image string from response
base64_image = response_body.get("images")[0]

# Decode Base64 → raw image bytes
base_64_image = base64.b64decode(base64_image)

# Save the generated image to a file
file_path = "cat.png"
with open(file_path, "wb") as f:
    f.write(base_64_image)
