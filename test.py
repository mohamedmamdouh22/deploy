from models.models import MBR_model
from PIL import Image
from torchvision import transforms
import numpy as np
import torch
import pandas as pd
model_path = 'models/best_mAP.pt'
y_length= 256
x_length= 256
n_mean= [0.5, 0.5, 0.5]
n_std= [0.5, 0.5, 0.5]
model= MBR_model(13164, ["R50", "R50", "BoT", "BoT"], n_groups=0, losses ="LBS", LAI=False)
model_path = 'models/best_mAP.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_state_dict = torch.load(model_path, map_location=device)
data={}
current_model=[]
print("Current model keys:")
for k in model.state_dict().keys():
    current_model.append(k)
data['current mode']=current_model

loaded_model=[]
# Print loaded state dict keys
print("\nLoaded state dict keys:")
for k in model_state_dict.keys():
    loaded_model.append(k)
data['loaded model']=loaded_model
# df=pd.DataFrame(data)
print(len(current_model),len(loaded_model))
# df.to_csv('test.csv')
model.load_state_dict(model_state_dict)  # Load it properly into the model instance
model.eval()  # Set the model to evaluation mode