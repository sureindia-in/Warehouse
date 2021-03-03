from fastapi import FastAPI
from pydantic import BaseModel
from palletsnboxes.predictor_yolo_detector.detector_test import Detector
from palletsnboxes.com_ineuron_utils.utils import decodeImage
import uvicorn
import os

import sys
sys.path.insert(0, './palletsnboxes')

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = FastAPI()

class Item(BaseModel):
    image:str


class ClientApp():
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.objectDetection = Detector(self.filename)


@app.get("/")
async def home():
    return "Landing page"
    #return render_template("index.html")

@app.post("/predict")
async def predictRoute(item:Item):
    try:
        image = item.image
        decodeImage(image, clApp.filename)
        result = clApp.objectDetection.detect_action()

    except ValueError as val:
        print(val)
        return "Value not found inside JSON data"
    except KeyError:
        return "Key value error , incorrect key passed"
    except Exception as e:
        print(e)
        result = "Invalid input"

    return result

#clApp = ClientApp()
if __name__ == "__main__":
    clApp = ClientApp()
    uvicorn.run(app,port=5000)