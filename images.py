from PIL import Image
import requests
from io import BytesIO
image_urls = {     
        "fruit.png": "https://drive.google.com/uc?export=view&id=14apFL43k3QnTRoCPD8oEMcsHB1y5ELk2"
}

image_cache = {}


#Function to download image from the internet and store it in a cache directory and return it
def download_image(image_name):
        if image_name in image_cache:
                return image_cache[image_name]
        else:                
                #check cache directory for image
                image_path = "asset/cache/" + image_name
                try:
                        image = Image.open(image_path)
                        print("Image found in cache")
                        image_cache[image_name] = image
                        return image
                except:
                        print("Image not found in cache")
                        #download image from the internet
                        print("downloading...")
                        response = requests.get(image_urls[image_name])
                        buffer = BytesIO(response.content)
                        image = Image.open(buffer)
                        #set image to image cache
                        image_cache[image_name] = image
                        #save image to cache directory
                        image.save(image_path)
                        return image        



