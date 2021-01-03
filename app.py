import io
import json
import os

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


# Transform input into the form our model expects
def transform_image(infile):
    # use multiple TorchVision transforms to ready the image
    input_transforms = [transforms.Resize(255),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        # Standard normalization for ImageNet model input
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    my_transforms = transforms.Compose(input_transforms)
    # Open the image file
    image = Image.open(infile)
    # Transform PIL image to appropriately-shaped PyTorch tensor
    timg = my_transforms(image)
    # PyTorch models expect batched input; create a batch of 1
    timg.unsqueeze_(0)
    return timg


# Get a prediction
def get_prediction(input_tensor):
    # Get likelihoods for all ImageNet classes
    outputs = model.forward(input_tensor)
    # Extract the most likely class
    _, y_hat = outputs.max(1)
    # Extract the int value from the PyTorch tensor
    prediction = y_hat.item()
    return prediction


# Make the prediction human-readable
def render_prediction(prediction_idx):
    stridx = str(prediction_idx)
    class_name = 'Unknown'
    if img_class_map is not None:
        if stridx in img_class_map is not None:
            class_name = img_class_map[stridx][1]

    return prediction_idx, class_name


@app.route('/', methods=['GET'])
def root():
    return jsonify({'msg': 'Try POSTing to the /predict endpoint with an RGB image attachment'})


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file is not None:
            input_tensor = transform_image(file)
            prediction_idx = get_prediction(input_tensor)
            class_id, class_name = render_prediction(prediction_idx)
            return jsonify({'class_id': class_id, 'class_name': class_name})


if __name__ == '__main__':
    app.run()
