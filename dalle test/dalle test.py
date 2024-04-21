import requests
import shutil
from openai import OpenAI
from PIL import Image
import cv2
import os

client = OpenAI()

response = client.images.generate(
  model="dall-e-3",
  prompt=input("Prompt: "),
  size="1024x1024",
  quality="standard",
  n=1,
)

image_url = response.data[0].url
print(image_url)

# Download the image
response = requests.get(image_url, stream=True)

# Check if the request was successful
if response.status_code == 200:
    # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
    response.raw.decode_content = True
    
    # Open a local file with wb ( write binary ) permission.
    with open('test.jpg', 'wb') as f:
        shutil.copyfileobj(response.raw, f)
        
    print('Image successfully downloaded: test.jpg')
else:
    print('Image couldn\'t be retrieved')

# Check if the file exists
if os.path.exists('./test.jpg'):
    # Resize the image
    img = cv2.resize(cv2.imread('./test.jpg'), (32, 32))
    # Save the image
    cv2.imwrite('test_resized.jpg', img)

    print('Image successfully resized to 32x32: test_resized.jpg')
else:
    print('File test.jpg does not exist')