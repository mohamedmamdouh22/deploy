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
from werkzeug.utils import secure_filename
import os
from PIL import Image
import torch.nn.functional as F
import torch
from utils import load_models, video_embeddings, find_most_similar
import threading
from helper import *
from globals import processing_status, data_transform
from image import gallery_embeddings
import concurrent.futures
from memory import vehicles_detection
# from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = os.urandom(24)
UPLOAD_FOLDER = "static/uploads/"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # Limit file size to 16MB

q_images = []
g_images = []
gallery = {}

model, yolo = load_models()
device = next(model.parameters()).device


@app.route("/")
def splash():
    return render_template("splash.html")


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


@app.route("/upload_gallery_images", methods=["POST"])
def upload_gallery_images():
    gallery_path = os.path.join(app.config["UPLOAD_FOLDER"], "gallery")

    if "images" not in request.files:
        flash("No file part")
        return redirect(request.url)

    files = request.files.getlist("images")
    if files and all(allowed_file(file.filename) for file in files):
        processing_status["status"] = "processing"

        # Save files
        filenames = []
        for file in files:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], "gallery", filename)
            if not os.path.exists(os.path.dirname(file_path)):
                os.makedirs(os.path.dirname(file_path))
            file.save(file_path)
            filenames.append(file_path)

        # Process images in a separate thread
        threading.Thread(target=process_gallery_images, args=(gallery_path,)).start()

        return redirect(url_for("processing"))

    flash("Invalid file type")
    return redirect(request.url)


def process_gallery_images(gallery_path):
    gallery_embeddings(gallery_path, model, g_images, gallery, batch_size=32)
    processing_status["status"] = "done"


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
        try:
            top_left = (
                int(float(request.args.get("top_left_x"))),
                int(float(request.args.get("top_left_y"))),
            )
            bottom_right = (
                int(float(request.args.get("bottom_right_x"))),
                int(float(request.args.get("bottom_right_y"))),
            )
        except (TypeError, ValueError) as e:
            flash("Invalid coordinates for the rectangle.")
            return redirect(request.url)

        # Set processing status to 'processing'
        processing_status["status"] = "processing"

        # Process the video in a separate thread
        threading.Thread(
            target=process_video_and_handle_images,
            args=(video_path, top_left, bottom_right),
        ).start()

        return redirect(url_for("processing"))

    flash("Invalid file type")
    return redirect(request.url)


# def process_video_and_handle_images(video_path, top_left, bottom_right):
#     start_time = time.time()
#     embeddings = video_embeddings(
#         video_path, model, yolo, top_left, bottom_right, skip_frames=3
#     )
#     end_time = time.time()
#     print("video processing time:", end_time - start_time)
#     g_images.extend(embeddings[0])
#     gallery.update(embeddings[1])
#     processing_status["status"] = "done"
def process_video_and_handle_images(video_path, top_left, bottom_right):
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], "gallery")
    os.makedirs(save_path, exist_ok=True)
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(
            vehicles_detection, yolo, video_path, save_path, top_left, bottom_right
        )
    gallery_embeddings(save_path, model, g_images, gallery)
    end_time = time.time()
    print("video processing time:", end_time - start_time)
    # g_images.extend(embeddings)
    processing_status["status"] = "done"


@app.route("/processing")
def processing():
    return render_template("processing.html")


@app.route("/check_processing_status")
def check_processing_status():
    return processing_status


def handle_upload(subfolder):
    if "images" not in request.files:
        flash("No file part")
        return redirect(request.url)

    files = request.files.getlist("images")
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

                img_tensor = data_transform(img).unsqueeze(0).to(device)
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
    global UPLOAD_FOLDER
    if not q_images:
        flash("No query images uploaded.")
        return redirect(url_for("upload_query"))

    if not g_images:
        flash("No gallery images uploaded.")
        return redirect(url_for("upload_gallery"))
    if len(g_images) == 2:
        indices, scores = find_most_similar(q_images[0], g_images[0], top_k=1)
    else:
        indices, scores = find_most_similar(q_images[0], g_images, top_k=1)
    if indices is None:
        clear_query_directory(UPLOAD_FOLDER)
        q_images = []
        # Redirect to the "No Match" page if no matches are found
        return redirect(url_for("no_match"))
    image_list = list(gallery.items())
    path_of_most_similar_image = image_list[indices[0][0]][0]
    similarity_score = scores[0] * 100  # Convert to percentage
    path_of_most_similar_image = path_of_most_similar_image.replace("\\", "/")
    # Clear the query directory and reset q_images
    clear_query_directory(UPLOAD_FOLDER)
    q_images = []
    return render_template(
        "predict.html",
        image_url=path_of_most_similar_image,
        similarity_score=similarity_score,
    )


@app.route("/no_match")
def no_match():
    return render_template("no_match.html")


if __name__ == "__main__":
    app.run(debug=True)
