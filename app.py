from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_images():
    """Handle multiple image uploading."""
    files = request.files.getlist('images')
    filenames = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filenames.append(filename)
    return redirect(url_for('uploaded_files', filenames=','.join(filenames)))

@app.route('/uploads/<filenames>')
def uploaded_files(filenames):
    """Display the uploaded files."""
    files = filenames.split(',')
    return render_template('uploaded.html', files=files)
@app.route('/uploads/<filename>')
def send_file(filename):
    """Serve a specific file."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/')
def index():
    """Render the main page with the upload form."""
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
