import os
import shutil

# this function checks the extension of the file passed
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {
        "png",
        "jpg",
        "jpeg",
        "gif",
        "mp4",
        "avi",
    }

def clear_query_directory(parent_dir):
    query_folder = os.path.join(parent_dir, 'query')
    for filename in os.listdir(query_folder):
        file_path = os.path.join(query_folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
