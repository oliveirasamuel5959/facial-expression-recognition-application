import numpy as np
import base64
import json
import requests

from PIL import Image
from io import BytesIO

image_path = "scripts/pretty_family.jpg"

URL = "http://192.168.0.16:8080/v1/predictions"
headers = {'Content-type': 'application/json'}

def image64_encode(filename):
    try:
        with open(filename, "rb") as image_file:
            base64_bytes = base64.b64encode(image_file.read())
            base64_encoded = base64_bytes.decode()
            data = {
                'image':
                    {
                        'author': 'Samuel',
                        'timestamp': 1234665,
                        'content': base64_encoded
                    },
                'collection_name': 'Image base64 for analysis'
            }
            
            print(data['image']['content'][0:20])
        
        return data
            
            # with open('scripts/embedded.json', 'w') as f:    
            #   json.dump(data, f)
        
    except FileNotFoundError:
        print(f"File not found {filename}")
    
    
def image64_decode(jsonfile):
    with open(jsonfile, 'r') as f:
        image_data = json.load(f)
    image = Image.open(BytesIO(base64.b64decode(image_data['images'][0]['content'])))
    image.save('scripts/base64_image_out.jpg', 'JPEG')
    
    
def send_image_post(jsonfile):
    r = requests.post(url=URL, data=json.dumps(jsonfile), headers=headers)
    print(r.status_code)

data = image64_encode(filename=image_path)
print(data)
print(data['image']['content'][0:20])
# image64_decode("scripts/embedded.json")
send_image_post(data)
