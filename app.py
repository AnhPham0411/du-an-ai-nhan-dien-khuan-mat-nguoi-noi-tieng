from flask import Flask, request, render_template, url_for, redirect, current_app, send_from_directory

import face_recognition
import os
import pickle
import train_faces

app = Flask(__name__)

# Gọi hàm train_faces.train_faces() khi ứng dụng khởi động
train_faces.train_faces()

# Tải cơ sở dữ liệu
with open('known_faces.pickle', 'rb') as f:
    known_faces = pickle.load(f)

def recognize_face(image_path):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    recognized_faces = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(
            [enc for enc_list in known_faces.values() for enc in enc_list], face_encoding)
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = list(known_faces.keys())[first_match_index // len(list(known_faces.values())[0])]
        recognized_faces.append(name)

    return recognized_faces





@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['file']
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)
    recognized_faces = recognize_face(file_path)
    return render_template('result.html', faces=recognized_faces, image=file.filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

@app.route('/create', methods=['GET', 'POST'])
def create():
    if request.method == 'POST':
        folder_name = request.form['folder']
        if folder_name:
            folder_path = os.path.join(os.getcwd(), 'known_faces', folder_name)
            if os.path.exists(folder_path):
                return render_template('create.html', error="Folder already exists!")
            else:
                os.makedirs(folder_path)
                file = request.files['file']
                if file:
                    file_path = os.path.join(folder_path, file.filename)
                    file.save(file_path)
                    train_faces.train_faces()
                    return redirect(url_for('home'))
                else:
                    return render_template('create.html', error="No file uploaded!")
    return render_template('create.html')


if __name__ == '__main__':
    app.run(debug=True)
