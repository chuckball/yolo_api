from flask import Flask, request, Response, jsonify
import io
from PIL import Image
from darkflow.net.build import TFNet
import cv2
import numpy as np
from urllib.request import urlopen
# For Eureka Functionality (Register, Renew, and Quit)
import requests
import json
import schedule
import threading
# Scheduler to repeatedly send renew
from apscheduler.schedulers.background import BackgroundScheduler
import atexit

# Initialize the Flask application
app = Flask(__name__)

SERVICE_NAME = "3102YOLOSERVICE"
REGISTRY_URL = "http://localhost:8761/eureka/apps/" + SERVICE_NAME +"/"
INSTANCEID = "YOLO"

def image_to_byte_array(image: Image):
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format='PNG')
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr

# Function to register the YOLO service with Spring Registry
def registerService(port):
    # Request body for Eureka Registration
    data = {
        "instance": {
            "instanceId": INSTANCEID + str(port),
            "hostName": "127.0.0.1",
            "app": SERVICE_NAME,
            "ipAddr": "127.0.0.1",
            "status": "UP",
            "port": {
                        "$": port,
                        "@enabled": True
            },
            "dataCenterInfo": {
                "@class": "com.netflix.appinfo.InstanceInfo$DefaultDataCenterInfo",
                "name": "MyOwn"
            }
        }
    }
    header = {"Content-Type": "application/json"}
    response = requests.post(REGISTRY_URL, data=json.dumps(data), headers=header)
    if(response.status_code == 204):
        print("Successfully registered service")
    else:
        print("Error registering | Status Code: " + str(response.status_code))

# Function to renew YOLO service leaase with Spring Registry
def sendHeartbeat(port):
    response = requests.put(REGISTRY_URL + INSTANCEID + str(port) + "?status=UP")
    if(response.status_code == 200):
        print("Heartbeat Sent")
    elif (response.status_code == 404):
        print("Failed to send Heartbeat, 404 received, registering...")
        registerService(port)
    else:
        print("Failed to send Heartbeat | Status Code: " + str(response.status_code))

# Initialises and starts a scheduler to repeatedly call renewLease
def initHeartbeatScheduler(port):
    scheduler = BackgroundScheduler()
    scheduler.add_job(func=(lambda: sendHeartbeat(port)),
                      trigger="interval", seconds=20)
    scheduler.start()
    if(scheduler.running):
        print("Scheduler started")
    else:
        print("Scheduler failed to start")

# Unregister YOLO service from Spring Registry
def unregisterService(port):
    response = requests.delete(REGISTRY_URL + INSTANCEID + str(port))
    if(response.status_code == 200):
        print("Successfully unregistered")
    else:
        print("Failed to unregister | Status Code: " + str(response.status_code))
    

# route http posts to this method
@app.route('/api/upload', methods=['POST'])
def upload():
    # load our input image and grab its spatial dimensions
    img = request.files["image"].read()
    img = Image.open(io.BytesIO(img))

    options = {"model": "./cfg/yolo.cfg",
               "load": "./cfg/yolo.weights", "threshold": 0.1, "gpu": 1.0}

    tfnet = TFNet(options)
    #url = request.form["image"]
    #url = "https://c8.alamy.com/comp/C8XXYP/airport-vehicles-dubai-international-airport-united-arab-emirates-C8XXYP.jpg"

    # Read image to classify
    #resp = urlopen(url)
    #img = np.asarray(bytearray(resp.read()), dtype="uint8")
    img = np.asarray(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    font = cv2.FONT_HERSHEY_SIMPLEX

    result = tfnet.return_predict(img)

    for row in result:
        cv2.rectangle(img, tuple(row['topleft'].values()), tuple(
            row['bottomright'].values()), (255, 0, 0), 1)
        cv2.putText(img, row['label'], tuple(
            row['topleft'].values()), font, 2, 255)

    print(type(img))
    np_img = Image.fromarray(img)
    img_encoded = image_to_byte_array(np_img)
    print(type(img_encoded))

    return Response(response=img_encoded, status=200, mimetype="image/jpeg")


# Test endpoint to check status of server
@app.route('/test', methods=['GET'])
def test():
    print("Test works")
    return "WORKS"
# In case you wanna use this as backend too
# @app.route('/')
# def index():
#    return render_template('update.html')
#
# @app.route('/success/<name>')
# def success(name):
#    return 'welcome %s' % name
#
# @app.route('/update',methods = ['POST', 'GET'])
# def update():
#    if request.method == 'POST':
#       user = request.form['nm']
#       return redirect(url_for('success',name = user))
#    else:
#       user = request.args.get('nm')
#       return redirect(url_for('success',name = user))


if __name__ == '__main__':
    print("if __name__ == '__main__'")
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-port")
    args = parser.parse_args()
    portArgs = args.port
    registerService(portArgs)
    initHeartbeatScheduler(portArgs)
    atexit.register(lambda: unregisterService(portArgs))
    app.run(debug=False, host="0.0.0.0", port=portArgs, use_reloader=False)

# git clone https://github.com/thtrieu/darkflow.git
# pip install Cython
# pip install ./darkflow/
# pip install opencv-python
# pip install tensorflow==1.14
