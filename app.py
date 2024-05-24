from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_from_directory,
)
from test import process_video
from werkzeug.utils import secure_filename
import os
from models.models import MBR_model
from PIL import Image
from torchvision import transforms

# from results import *
import torch.nn.functional as F
import numpy as np
import torch
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


def load_model():
    global model
    model_path = "models/best_mAP.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(model_state_dict)
    model.eval()
    return model


@app.route("/")
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


@app.route("/upload_gallery_images", methods=["POST"])
def upload_gallery_images():
    return handle_upload("gallery")


@app.route("/upload_query_images", methods=["POST"])
def upload_query_images():
    return handle_upload("query")


@app.route("/upload_video_file", methods=["POST"])
def upload_video_file():
    if "video" not in request.files:
        flash("No video file part")
        return redirect(request.url)

    file = request.files["video"]
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(video_path)

        top_left = (75, 200)  # Replace with your top-left coordinates
        bottom_right = (1205, 600)  # Replace with your bottom-right coordinates

        # Process the video
        results = process_video(video_path, top_left, bottom_right)
        # Extract the paths of the saved car images
        car_image_paths = [result["path"] for result in results]

        # Handle the uploaded car images
        handle_uploaded_car_images(car_image_paths)

        flash("Video processed and car images saved.")
        return redirect(url_for("success_gallery"))

    flash("Invalid file type")
    return redirect(request.url)


def handle_uploaded_car_images(image_paths):
    model = load_model()
    for file_path in image_paths:
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
                # x_length, y_length = img.size

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
                    # q_images.append(
                    #     {
                    #         f'{id}':torch.cat(end_vec, 1)
                    #     }
                    # )

    # flash(f'Image stats: {image_stats}')
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


def find_most_similar(query, gallery, top_k=5):
    """
    Find the most similar tensors in the gallery to the query tensor using cosine similarity.

    Parameters:
    - query (torch.Tensor): A 1xN tensor representing the query image features.
    - gallery (list of torch.Tensor): A list of 1xN tensors representing the gallery image features.
    - top_k (int): The number of top similar items to return.

    Returns:
    - list of int: Indices of the top_k most similar tensors in the gallery.
    """
    gallery_tensor = torch.stack(gallery)

    query_normalized = F.normalize(query, p=2, dim=1)
    gallery_normalized = F.normalize(gallery_tensor, p=2, dim=1)
    gallery_normalized = gallery_normalized.squeeze(1)  # Reshape to [3, 8192]
    # gallery_normalized_t = gallery_normalized.transpose(0, 1)  # Transpose to [8192, 3]

    similarities = torch.mm(query_normalized, gallery_normalized.transpose(0, 1))
    _, top_indices = torch.topk(similarities, top_k, largest=True, sorted=True)

    return top_indices[0].tolist()


# @app.route('/predict', methods=['POST'])
# def predict():
#     indices = find_most_similar(q_images[0], g_images, top_k=1)
#     image_list = list(gallery.items())
#     path_of_most_similar_image = image_list[indices[0]][0]


#     image_url = path_of_most_similar_image.replace("\\", "/")
#     print(image_url)
#     return render_template('predict.html', image_url=image_url)
@app.route("/predict", methods=["POST"])
def predict():
    if not q_images:
        flash("No query images uploaded.")
        return redirect(url_for("upload_query"))

    if not g_images:
        flash("No gallery images uploaded.")
        return redirect(url_for("upload_gallery"))

    indices = find_most_similar(q_images[0], g_images, top_k=1)
    image_list = list(gallery.items())
    path_of_most_similar_image = image_list[indices[0]][0]
    path_of_most_similar_image = path_of_most_similar_image.replace("\\", "/")
    # Convert file path to URL path
    print(path_of_most_similar_image)
    # image_url = url_for('static', filename='uploads/gallery/' + path_of_most_similar_image)

    return render_template("predict.html", image_url=path_of_most_similar_image)


if __name__ == "__main__":
    app.run(debug=True)
