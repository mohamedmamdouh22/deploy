from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit file size to 16MB

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
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], subfolder)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            file.save(os.path.join(save_path, filename))
    return redirect(url_for('success'))

@app.route('/success')
def success():
    return render_template('success.html')

if __name__ == '__main__':
    app.run(debug=True)
