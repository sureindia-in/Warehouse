import pymongo
from PIL import Image
from bson.binary import Binary
#import matplotlib.pyplot as plt
import io
#from datetime import datetime

class CreateConnection:
    def __init__(self):
        self.url = "mongodb+srv://root:root@cluster0.kcya4.mongodb.net/Prediction?retryWrites=true&w=majority"
        self.client = pymongo.MongoClient(self.url)
        self.db_name = "Prediction"
        self.database = self.client[self.db_name]


    def insert_bad(self,image):
        self.db_name = "Bad_Prediction"
        self.image = image
        self.collection = self.database["Bad_prediction_images"]
        record = {"image": self.image}
        self.collection.insert_one(record)

    def insert_good(self,image,s):
        self.db_name = "Good_Prediction"
        self.image = image
        self.collection = self.database["Good_prediction_images"]
        record = {"image": self.image,"Predictions" : s}
        self.collection.insert_one(record)

#convert string back to image

# client = MongoClient()
# db = client.testdb
# images = db.images
# image = images.find_one()
#
# pil_img = Image.open(io.BytesIO(image['data']))
# plt.imshow(pil_img)
# plt.show()