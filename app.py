from flask import Flask, request, Response, jsonify
import io
from PIL import Image
from darkflow.net.build import TFNet
import cv2
import numpy as np
from urllib.request import urlopen
import py_eureka_client.eureka_client as eureka_client

REST_SERVER_PORT = 5000

eureka_client.init_registry_client(eureka_server="http://localhost:8761/eureka",
                                   app_name="3102-YOLO-SERVICE",
                                   instance_port=REST_SERVER_PORT)
print("Registered")

options = {"model": "./cfg/yolo.cfg",
           "load": "./cfg/yolo.weights", "threshold": 0.1, "gpu": 1.0}

tfnet = TFNet(options)

# Initialize the Flask application
app = Flask(__name__)


def image_to_byte_array(image: Image):
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format='PNG')
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr


# route http posts to this method
@app.route('/api/upload', methods=['POST'])
def yoloImage():
    # load our input image and grab its spatial dimensions
    img = request.files["image"].read()
    img = Image.open(io.BytesIO(img))

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
        print("Row: " + str(row))
        cv2.rectangle(img, tuple(row['topleft'].values()), tuple(
            row['bottomright'].values()), (255, 0, 0), 1)
        cv2.putText(img, row['label'], tuple(
            row['topleft'].values()), font, 2, 255)

    print(type(img))
    np_img = Image.fromarray(img)
    img_encoded = image_to_byte_array(np_img)
    print(type(img_encoded))

    return Response(response=img_encoded, status=200, mimetype="image/jpeg")

# Converts results from prediction into JSON string
def resultToJson(result):
    jsonResult = "{ \"results\" : "
    resultsStr = str(result)
    jsonResult += resultsStr + "}"
    print("jsonResult:\n" + jsonResult)
    return jsonResult

# route http posts to this method
@app.route('/api/uploadTest', methods=['POST'])
def yoloJson():
    # load our input image and grab its spatial dimensions
    img = request.files["image"].read()
    img = Image.open(io.BytesIO(img))

    #url = request.form["image"]
    #url = "https://c8.alamy.com/comp/C8XXYP/airport-vehicles-dubai-international-airport-united-arab-emirates-C8XXYP.jpg"

    # Read image to classify
    #resp = urlopen(url)
    #img = np.asarray(bytearray(resp.read()), dtype="uint8")
    img = np.asarray(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    font = cv2.FONT_HERSHEY_SIMPLEX

    result = tfnet.return_predict(img)

    response = resultToJson(result)
    print("Response\n" + response)

    return Response(response=response, status=200, mimetype="application/json")

@app.route('/test', methods=['GET'])
def testRoute():
    print("/test reached")
    return "FLASK TEST PASSED"
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
    app.run(debug=False, host='0.0.0.0', port=REST_SERVER_PORT)

# git clone https://github.com/thtrieu/darkflow.git
# pip install Cython
# pip install ./darkflow/
# pip install opencv-python
# pip install tensorflow==1.14