import requests
import os
import gradio as gr
import matplotlib.colors as mcolors
import numpy as np
import spaces
import torch
from gradio.themes.utils import sizes
from PIL import Image
from torchvision import transforms

# Function to download images from Pixabay
def download_pixabay_images(search_term, api_key, num_images=10, download_path='images'):
    """
    Downloads images from Pixabay based on a search term.

    Args:
        search_term (str): Keyword to search for images.
        api_key (str): Your Pixabay API key.
        num_images (int): Number of images to download (default is 10).
        download_path (str): Path to the folder where images will be saved (default is 'images').
    """
    # Create the directory if it doesn't exist
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    # Pixabay API endpoint
    url = f"https://pixabay.com/api/?key={api_key}&q={search_term}&image_type=photo&per_page={num_images}"

    # Send a GET request to the Pixabay API
    response = requests.get(url)

    # Check for a successful response
    if response.status_code == 200:
        data = response.json()
        images = data.get('hits', [])

        # Loop through the images and download them
        for i, image in enumerate(images):
            image_url = image['largeImageURL']
            image_data = requests.get(image_url).content

            # Save the image locally
            file_path = os.path.join(download_path, f"{search_term}_{i+1}.jpg")
            with open(file_path, 'wb') as f:
                f.write(image_data)

            print(f"Downloaded: {file_path}")
    else:
        print(f"Error: {response.status_code}, {response.text}")

# Example usage
if __name__ == "__main__":
    search_query = input("Enter the search term: ")
    api_key = "47641863-94e5b19f029ba409fdebd4bdf"  # Replace with your Pixabay API key
    num_images_to_download = int(input("Enter the number of images to download: "))

    download_pixabay_images(search_query, api_key, num_images=num_images_to_download)
