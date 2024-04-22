from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
import os
from models.models import MBR_model
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import torch
# from sklearn.metrics.pairwise import cosine_similarity
app = Flask(__name__)
app.secret_key = os.urandom(24)
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit file size to 16MB
model_path = 'models/best_mAP.pt'
y_length= 256
x_length= 256
n_mean= [0.5, 0.5, 0.5]
n_std= [0.5, 0.5, 0.5]
model=MBR_model(13164, ["R50", "R50", "BoT", "BoT"], n_groups=0, losses ="LBS", LAI=False)
model_path = 'models/best_mAP.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(model_state_dict)  # Load it properly into the model instance
model.eval()  # Set the model to evaluation mode
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_gallery')
def upload_gallery():
    return render_template('upload.html')

@app.route('/upload_query')
def upload_query():
    return render_template('upload_query.html')

@app.route('/upload_gallery_images', methods=['POST'])
def upload_gallery_images():
    return handle_upload('gallery')

@app.route('/upload_query_images', methods=['POST'])
def upload_query_images():
    return handle_upload('query')

def handle_upload(subfolder):
    if 'images' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    files = request.files.getlist('images')
    images=[]
    gf=[]
    
    for id,file in enumerate(files):
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], subfolder)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            file_path = os.path.join(save_path, filename)
            file.save(file_path)

            # Open the image file
            with Image.open(file_path) as img:
                img = img.convert('RGB')  # Convert image to RGB
                # x_length, y_length = img.size

                # Convert image data to a numpy array
                img_data = np.array(img)
                # Define transformations
                test_transform = transforms.Compose([
                    transforms.Resize((y_length, x_length), antialias=True),
                    transforms.ToTensor(),  # Convert image to tensor
                    transforms.Normalize(n_mean, n_std)
                ])   
                img_tensor = test_transform(img)
                if len(img_tensor.shape) == 3:  # If the tensor is C x H x W
                    img_tensor = img_tensor.unsqueeze(0) 
                with torch.no_grad():  # Turn off gradients to speed up this part
                    prediction = model(img_tensor)
                # print(len(prediction[2]))
                ffs=prediction[2]
                # for item in ffs:
                end_vec=[]
                for i in ffs:
                    end_vec.append(F.normalize(i))
                # gf.append(torch.cat(end_vec, 1))
                images.append(
                    {
                        f'{id}':torch.cat(end_vec, 1)
                    }
                )
    for i in range(len(images)):
        print(images[i][f'{i}'])
                
                # print(prediction[2][0][0].size)
                # {
                #     filename:prediction
                # }

                # Store stats for each image
                # image_stats.append({
                #     'filename': filename,
                #     'x_length': x_length,
                #     'y_length': y_length,
                #     'mean': mean.tolist(),  # Convert numpy array to list for JSON serializable
                #     'std': std.tolist()
                # })
            

    # flash(f'Image stats: {image_stats}')
    return redirect(url_for('success'))

@app.route('/success')
def success():
    return render_template('success.html')

if __name__ == '__main__':
    app.run(debug=True)
