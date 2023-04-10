from flask import Blueprint, request
import torch
from arc_model import CnnModel as Net
from arc_controller import PredictionController as inference

net = Net.net
net.load_state_dict(torch.load("./ds_trained/SinhalaTamil_CNN_Trained.pt", map_location=torch.device('cpu')))
net.eval()

characterBP = Blueprint('character', __name__)


@characterBP.route('/predict/', methods=['GET', 'POST'])
def predict():
    string_data = request.get_data().decode('utf-8')
    prediction = inference.get_prediction(string_data, net)
    return prediction