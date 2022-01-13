from flask import Flask, jsonify, request
import cv2
import pytesseract
import urllib.request
from flask_cors import CORS

application = Flask(__name__)
CORS(application)


# Grayscale, Gaussian blur, Otsu's threshold
# image = cv2.imread("image/local.jpg")
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(gray, (3, 3), 0)
# thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
#
# # Morph open to remove noise and invert image
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
# invert = 255 - opening
#
# # Perform text extraction
# data = pytesseract.image_to_string(invert, lang='eng', config='--psm 6')
# print(data)

@application.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@application.route('/api/search', methods=['GET'])
def processJSON():
    # Parsing data as JSON.
    data = request.get_json()

    # load image url from the get request
    image_url = data['image_url']

    print(image_url)

    # Retrieve the image into a location on disk.
    urllib.request.urlretrieve(image_url, "image/local.jpg")

    # Loading image from the directory
    image = cv2.imread("image\\local.jpg")

    # First we have to convert the image into RGB format for pytesseract api to understand
    # as by default cv2 loads the image in GBR format
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Returns the result of a Tesseract OCR run on the provided image to string
    text = pytesseract.image_to_string(img_rgb)

    return jsonify({'result': text})


if __name__ == '__main__':
    application.run()
