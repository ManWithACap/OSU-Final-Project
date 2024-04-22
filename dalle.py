import requests
import shutil
from openai import OpenAI
import cv2
import os
from dotenv import load_dotenv

load_dotenv()

def generate_and_resize_image(prompt):
    client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
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
        return 'test_resized.jpg'
    else:
        print('File test.jpg does not exist')
        return None

if __name__ == "__main__":
    prompt = input("Prompt: ")
    resized_image_filename = generate_and_resize_image(prompt)
    if resized_image_filename:
        print("Resized image saved as:", resized_image_filename)
    else:
        print("Failed to generate or resize image.")
