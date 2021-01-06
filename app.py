import io
import json
import os

##########################
# turn pytorch model to api
#########################

import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request

app = Flask(__name__)
# Trained on 1000 classes from ImageNet
model = models.densenet121(pretrained=True)
# Turns off autograd and
model.eval()

img_class_map = None
# Human-readable names for Imagenet classes
mapping_file_path = 'index_to_name.json'
if os.path.isfile(mapping_file_path):
    with open(mapping_file_path) as f:
        img_class_map = json.load(f)


# when sending an RGB image
def transform_image(infile):
    # use multiple TorchVision transforms to ready the image
    input_transforms = [transforms.ToTensor(),
                        transforms.Lambda(lambda x: x.mul(255))]
    my_transforms = transforms.Compose(input_transforms)
    # Open the image file
    image = Image.open(infile)
    # Transform PIL image to appropriately-shaped PyTorch tensor
    timg = my_transforms(image)
    # PyTorch models expect batched input; create a batch of 1
    timg.unsqueeze_(0)
    return timg


# # Get a prediction
# def get_prediction(input_tensor):
#     # Get likelihoods for all ImageNet classes
#     outputs = model.forward(input_tensor)
#     # Extract the most likely class
#     _, y_hat = outputs.max(1)
#     # Extract the int value from the PyTorch tensor
#     prediction = y_hat.item()
#     return prediction
#
#
# # Make the prediction human-readable
# def render_prediction(prediction_idx):
#     stridx = str(prediction_idx)
#     class_name = 'Unknown'
#     if img_class_map is not None:
#         if stridx in img_class_map is not None:
#             class_name = img_class_map[stridx][1]
#
#     return prediction_idx, class_name


# !python3 "neural_style.py" eval --content-image './image1.jpg' --model './rain_princess.pth' --output-image './output/out1.jpg' --cuda 1

@app.route('/', methods=['GET'])
def root():
    return jsonify({'msg': 'Try POSTing to the /generate endpoint with an RGB image attachment'})


@app.route('/generate', methods=['POST'])
def generate():
    if request.method == 'POST':
        file = request.files['file']
        if file is not None:
            input_tensor = transform_image(file)
            prediction_idx = get_prediction(input_tensor)
            class_id, class_name = render_prediction(prediction_idx)
            return jsonify({'class_id': class_id, 'class_name': class_name})


###########################
# code to test connection with android app
###########################

# from flask import Flask, request, jsonify
#
# app = Flask(__name__)
#
#
# # root
# @app.route("/")
# def index():
#     """
#     this is a root dir of my server
#     :return: str
#     """
#     return "This is root!!!!"
#
#
# # GET
# @app.route('/users/<user>')
# def hello_user(user):
#     """
#     this serves as a demo purpose
#     :param user:
#     :return: str
#     """
#     return "Hello %s!" % user
#
#
# # POST
# @app.route('/api/post_some_data', methods=['POST'])
# def get_text_prediction():
#     """
#     predicts requested text whether it is ham or spam
#     :return: json
#     """
#     json = request.get_json()
#     print(json)
#     if len(json['text']) == 0:
#         return jsonify({'error': 'invalid input'})
#
#     return jsonify({'you sent this': json['text']})
#
###########################

# running web app in local machine
# server on http://0.0.0.0:5000/
# visible across the network
# BaseUrl for Android http://<your ip address>:5000/...
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

# development way
# server on http://127.0.0.1:5000/
# (invisible across the network) won't work on other device, other than development machine
# if __name__ == '__main__':
#     app.run()
