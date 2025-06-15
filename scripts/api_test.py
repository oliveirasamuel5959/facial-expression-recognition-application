import numpy as np
import base64
import json
import requests

from PIL import Image
from io import BytesIO

image_path = "images/me.jpg"


URL = "http://192.168.0.16:5000/v1/predictions"
headers = {'Content-type': 'application/json'}

def image64_encode(filename):
    try:
        with open(filename, "rb") as image_file:
            base64_bytes = base64.b64encode(image_file.read())
            base64_encoded = base64_bytes.decode()
            data = {
                'images':
                    [
                        {
                            'author': 'Samuel',
                            'timestamp': 1234665,
                            'name': filename,
                            'content': base64_encoded
                        }
                    ],
                'collection_name': 'Image base64 for analysis'
            }
            
            with open('scripts/embedded.json', 'w') as f:    
                json.dump(data, f)
        
    except FileNotFoundError:
        raise FileNotFoundError
    
    
def image64_decode(jsonfile):
    with open(jsonfile, 'r') as f:
        image_data = json.load(f)
    image = Image.open(BytesIO(base64.b64decode(image_data['images'][0]['content'])))
    image.save('scripts/base64_image_out.jpg', 'JPEG')
    
    
def send_image_post(jsonfile):
    with open(jsonfile, 'r') as f:
        image_data = json.load(f)
    print(image_data['images'][0]['name'])
    
    r = requests.post(url=URL, data=json.dumps(image_data), headers=headers)
    print(r.status_code)

base64_image_data = image64_encode(filename=image_path)
image64_decode("scripts/embedded.json")

send_image_post('scripts/embedded.json')
