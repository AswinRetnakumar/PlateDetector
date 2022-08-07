from flask import Flask, request, jsonify
from flask_cors import CORS
from plate_detector import PlateDetector
import cv2 
import numpy as np
import requests

app = Flask(__name__)
cors = CORS(app)
plt = PlateDetector()

content_type = 'image/jpeg'
headers = {'content-type': content_type}
test_url = "http://192.168.1.39:5000/recog_plate"

@app.route('/read_plate', methods=['POST', 'GET'])
def read_plate():

    if request.method == 'GET':
        return 'Plate detector is running'

    else:
        results = {}
        print(request.data)
        nparr = np.fromstring(request.data, np.uint8)
        # decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        plates = plt.detect_plate(img)

        for i, plate in enumerate(plates):
            plate = cv2.cvtColor(plate, cv2.COLOR_BGR2RGB)
            _, img_encoded = cv2.imencode('.jpg', plate)
            response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)
            
            results[str(i)] = response.json()['plate_number']

        return {'plates':results}

@app.route('/test', methods = ['POST'])
def test_plate_reader():
    nparr = np.fromstring(request.data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    plate = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    _, img_encoded = cv2.imencode('.jpg', plate)
    response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)   
    result = response.json()['plate_number']
    return result


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)