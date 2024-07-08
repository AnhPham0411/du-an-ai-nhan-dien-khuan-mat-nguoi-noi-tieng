import face_recognition # Thư viện xử lí khuôn mặt
import os # Thư viện xử lí hình ảnh
import pickle # Thư viện tải và lưu trữ dữ liệu

def train_faces():
    # Thư mục chứa ảnh của những người nổi tiếng (sử dụng đường dẫn tuyệt đối)
    known_faces_dir = os.path.abspath('known_faces')

    # Tạo dictionary để lưu mã nhận diện
    known_faces = {}

    # Duyệt qua thư mục và mã hóa ảnh
    for name in os.listdir(known_faces_dir):
        person_dir = os.path.join(known_faces_dir, name)
        known_faces[name] = []
        for filename in os.listdir(person_dir):
            image_path = os.path.join(person_dir, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_faces[name].append(encodings[0])

    # Lưu cơ sở dữ liệu vào file
    with open('known_faces.pickle', 'wb') as f:
        pickle.dump(known_faces, f)

# Gọi hàm train_faces để thực hiện việc huấn luyện và lưu cơ sở dữ liệu
train_faces()
