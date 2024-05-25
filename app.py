from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_from_directory,
)
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from test import process_video
from werkzeug.utils import secure_filename
import os
import shutil
from models.models import MBR_model
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import torch
from similarity_check import find_most_similar
# from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = os.urandom(24)
UPLOAD_FOLDER = "static/uploads/"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # Limit file size to 16MB
y_length = 256
x_length = 256
n_mean = [0.5, 0.5, 0.5]
n_std = [0.5, 0.5, 0.5]
q_images = []
g_images = []
gallery = {}
processing_status = {'status': 'idle'}
model = MBR_model(
    13164, ["R50", "R50", "BoT", "BoT"], n_groups=0, losses="LBS", LAI=False
)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {
        "png",
        "jpg",
        "jpeg",
        "gif",
        "mp4",
        "avi",
    }

def clear_query_directory():
    query_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'query')
    for filename in os.listdir(query_folder):
        file_path = os.path.join(query_folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def load_model():
    global model
    model_path = "models/best_mAP.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(model_state_dict)
    model.eval()
    return model

@app.route('/')
def splash():
    return render_template('splash.html')
@app.route("/index")
def index():

    return render_template("index.html")


@app.route("/selection")
def selection():
    return render_template("selection.html")


@app.route("/process_selection", methods=["POST"])
def process_selection():
    processing_type = request.form.get("processing_type")
    if processing_type == "image_to_image":
        return redirect(url_for("upload_gallery"))
    elif processing_type == "video_to_image":
        return redirect(url_for("upload_video"))
    else:
        flash("Invalid selection")
        return redirect(url_for("selection"))


@app.route("/upload_gallery")
def upload_gallery():
    return render_template("upload.html")


@app.route("/upload_query")
def upload_query():
    return render_template("upload_query.html")


@app.route("/upload_video")
def upload_video():
    return render_template("upload_video.html")


@app.route('/upload_gallery_images', methods=['POST'])
def upload_gallery_images():
    global processing_status

    if 'images' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    files = request.files.getlist('images')
    if files and all(allowed_file(file.filename) for file in files):
        processing_status['status'] = 'processing'
        
        # Save files
        filenames = []
        for file in files:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'gallery', filename)
            if not os.path.exists(os.path.dirname(file_path)):
                os.makedirs(os.path.dirname(file_path))
            file.save(file_path)
            filenames.append(file_path)
        
        # Process images in a separate thread
        threading.Thread(target=process_gallery_images, args=(filenames,)).start()

        return redirect(url_for('processing'))

    flash('Invalid file type')
    return redirect(request.url)
def process_gallery_images(image_paths):
    global processing_status
    handle_uploaded_car_images(image_paths)
    processing_status['status'] = 'done'

@app.route("/upload_query_images", methods=["POST"])
def upload_query_images():
    return handle_upload("query")


@app.route('/upload_video_file', methods=['POST'])
def upload_video_file():
    global processing_status

    if 'video' not in request.files:
        flash('No video file part')
        return redirect(request.url)
    
    file = request.files['video']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)

        top_left = (75, 200)  
        bottom_right = (1205, 600) 

        # Set processing status to 'processing'
        processing_status['status'] = 'processing'

        # Process the video in a separate thread
        threading.Thread(target=process_video_and_handle_images, args=(video_path, top_left, bottom_right)).start()

        return redirect(url_for('processing'))

    flash('Invalid file type')
    return redirect(request.url)
def process_video_and_handle_images(video_path, top_left, bottom_right):
    global processing_status

    results = process_video(video_path, top_left, bottom_right)
    car_image_paths = [result['path'] for result in results]
    start_time = time.time()
    handle_uploaded_car_images(car_image_paths)
    end_time = time.time()
    model_time=end_time-start_time
    print(f"model time is: {model_time} s")
    # Set processing status to 'done'
    processing_status['status'] = 'done'

def process_image(file_path, model):
    print(file_path)
    if allowed_file(file_path):
        with Image.open(file_path) as img:
            img = img.convert("RGB")
            test_transform = transforms.Compose(
                [
                    transforms.Resize((y_length, x_length), antialias=True),
                    transforms.ToTensor(),
                    transforms.Normalize(n_mean, n_std),
                ]
            )
            img_tensor = test_transform(img)
            if len(img_tensor.shape) == 3:
                img_tensor = img_tensor.unsqueeze(0)
            with torch.no_grad():
                prediction = model(img_tensor)

            ffs = prediction[2]
            end_vec = [F.normalize(item) for item in ffs]

            g_images.append(torch.cat(end_vec, 1))
            gallery.update({f"{file_path}": torch.cat(end_vec, 1)})
@app.route('/processing')
def processing():
    return render_template('processing.html')

@app.route('/check_processing_status')
def check_processing_status():
    return processing_status

def handle_uploaded_car_images(image_paths):
    model = load_model()
    model.eval()  
    
    # Use ThreadPoolExecutor to handle threading
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = [executor.submit(process_image, file_path, model) for file_path in image_paths]
        for future in futures:
            future.result()

def handle_upload(subfolder):
    if "images" not in request.files:
        flash("No file part")
        return redirect(request.url)

    files = request.files.getlist("images")
    model = load_model()
    # gf = []
    for id, file in enumerate(files):
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], subfolder)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            file_path = os.path.join(save_path, filename)
            file.save(file_path)

            # Open the image file
            with Image.open(file_path) as img:
                img = img.convert("RGB")  # Convert image to RGB

                # Define transformations
                test_transform = transforms.Compose(
                    [
                        transforms.Resize((y_length, x_length), antialias=True),
                        transforms.ToTensor(),  # Convert image to tensor
                        transforms.Normalize(n_mean, n_std),
                    ]
                )
                img_tensor = test_transform(img)
                if len(img_tensor.shape) == 3:  # If the tensor is C x H x W
                    img_tensor = img_tensor.unsqueeze(0)
                with torch.no_grad():  # Turn off gradients to speed up this part
                    prediction = model(img_tensor)
                # print(len(prediction[2]))
                ffs = prediction[2]
                # for item in ffs:
                end_vec = []
                for item in ffs:
                    end_vec.append(F.normalize(item))
                # gf.append(torch.cat(end_vec, 1))
                if subfolder == "gallery":
                    g_images.append(torch.cat(end_vec, 1))
                    gallery.update({f"{file_path}": torch.cat(end_vec, 1)})
                else:
                    q_images.append(torch.cat(end_vec, 1))
    if subfolder == "gallery":
        return redirect(url_for("success_gallery"))
    else:
        return redirect(url_for("success_query"))


@app.route("/success_gallery")
def success_gallery():
    return render_template("success_gallery.html")


@app.route("/success_query")
def success_query():
    return render_template("success_query.html")

@app.route("/predict", methods=["POST"])
def predict():
    global q_images
    if not q_images:
        flash("No query images uploaded.")
        return redirect(url_for("upload_query"))

    if not g_images:
        flash("No gallery images uploaded.")
        return redirect(url_for("upload_gallery"))

    indices, scores = find_most_similar(q_images[0], g_images, top_k=1)
    image_list = list(gallery.items())
    path_of_most_similar_image = image_list[indices[0][0]][0]
    similarity_score = scores[0] * 100  # Convert to percentage
    path_of_most_similar_image = path_of_most_similar_image.replace("\\", "/")
    # Clear the query directory and reset q_images
    clear_query_directory()
    q_images = []
    return render_template("predict.html", image_url=path_of_most_similar_image, similarity_score=similarity_score)


if __name__ == "__main__":
    app.run(debug=True)
