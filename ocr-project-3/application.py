import urllib
from urllib import request
from flask import Flask, jsonify, request
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import cv2

application = Flask(__name__)

# initializing mtcnn for face detection
mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20)

# initializing resnet for face img to embedding conversion
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# photos folder path
dataset = datasets.ImageFolder('photos')
# accessing names of peoples from folder names
idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}


def collate_fn(x):
    return x[0]


loader = DataLoader(dataset, collate_fn=collate_fn)

# list of cropped faces from photos folder
face_list = []
# list of names corresponding to cropped photos
name_list = []
# list of embedding matrix after conversion from cropped faces to embedding matrix using resnet
embedding_list = []

for img, idx in loader:
    face, prob = mtcnn(img, return_prob=True)
    if face is not None and prob > 0.90:  # if face detected and probability > 90%
        emb = resnet(face.unsqueeze(0))  # passing cropped face into resnet model to get embedding matrix
        embedding_list.append(emb.detach())  # resulted embedding matrix is stored in a list
        name_list.append(idx_to_class[idx])  # names are stored in a list

# Saving data into data.pt file
data = [embedding_list, name_list]

# saving data.pt file
torch.save(data, 'data.pt')


# Matching face id of the given photo with available data from data.pt file

def face_match(img_path, data_path):  # img_path= location of photo, data_path= location of data.pt
    # getting embedding matrix of the given img
    given_image = Image.open(img_path)
    face, prob = mtcnn(given_image, return_prob=True)  # returns cropped face and probability
    emb = resnet(face.unsqueeze(0)).detach()  # detech is to make required gradient false

    saved_data = torch.load('data.pt')  # loading data.pt file
    embedding_list = saved_data[0]  # getting embedding data
    name_list = saved_data[1]  # getting list of names
    dist_list = []  # list of matched distances, minimum distance is used to identify the person

    for idx, emb_db in enumerate(embedding_list):
        dist = torch.dist(emb, emb_db).item()
        dist_list.append(dist)

    idx_min = dist_list.index(min(dist_list))
    return name_list[idx_min], min(dist_list)


def get_name_of_image(name):
    result2 = face_match(name, 'data.pt')
    print('Face matched with: ', result2[0], '  With distance: ', result2[1])
    return 'Face matched with: ', result2[0], '  With distance: ', result2[1]


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
    urllib.request.urlretrieve(image_url, "test.jpg")

    # Loading image from the directory
    image = cv2.imread("test.jpg")

    text = get_name_of_image('test.jpg')

    return jsonify({'result': text})


if __name__ == '__main__':
    application.run()
