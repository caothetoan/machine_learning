import face_recognition
from annoy import AnnoyIndex

NUMBER_OF_TREES = 100
f = 128
t = AnnoyIndex(f, 'angular')

paths = ''
imagePaths = list(paths.list_images('/data'))

def image_encoding(imagePath):
    img = face_recognition.load_image_file(imagePath)
    img_ = face_recognition.face_locations(img)
    top, right, bottom, left = [ v for v in img_[0] ]
    face = img[top:bottom, left:right]
    img_emb = face_recognition.face_encodings(face)[0]
    return img_emb


for i, imagePath in tqdm(enumerate(imagePaths)):
    img_emb = image_encoding(imagePath)
    t.add_item(i, img_emb)

t.build(NUMBER_OF_TREES) # 100trees
t.save('images.ann')
# Bước 3: Load data từ annoy

# Sau khi đã có file lưu index là images.ann ta sẽ bắt đầu load từ file đó:
f = 128
u = AnnoyIndex(f, 'angular')
u.load('images.ann')
# Bước 4: Get name

imagePaths = list(paths.list_images('path_of_you'))

for i, imagePath in tqdm(enumerate(imagePaths)):
    name = imagePath.split(os.path.sep)[-2]
    known_face_names.append(name)
# Bước 5: Nhận dạng khuôn mặt
face_names = []
while True:
    ret, frame = video_capture.read()
    # resize hình ảnh xuống 1/4 để  quá trình nhận dạng mặt nhanh hơn
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Chuyển đổi từ BGR sang RGB (mặc định opencv dùng là BGR thay vì RGB)
    rgb_small_frame = small_frame[:, :, ::-1]

   
    face_names = []
        
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    for face_encoding in face_encodings:
        #Lấy index của vector trong annoy 
        matches_id = u.get_nns_by_vector(face_encoding, 1)[0]
        #Lấy vector ra  từ index tương ứng đã lấy ở trên
        known_face_encoding = u.get_item_vector(matches_id)
        #Hàm này trả về giá trị True or False, nếu giống là True không giống là False 
        compare_faces = face_recognition.compare_faces([known_face_encoding], face_encoding)
        name = "unknown"

        if compare_faces[0]:
            #Lấy tên từ mảng đã tạo bước 3 dựa vào id tương ứng
            name = known_face_names[matches_id]
        face_names.append(name)
        print(face_names)
    # Sau khi đã locate được khuôn mặt và tên của người có trong database rồi ta sẽ tiến hành show nó lên camera:
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Lúc đầu ra scale nó xuống nhỏ gấu 4 lần để detect faces tốt hơn, bây giờ ta sẽ nhân trả lại tọa độ gốc cho nó
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # draw line cho face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw label cho face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        output_names.append(name)


cv2.imshow('Video', frame)

if cv2.waitKey(1) & 0xFF == ord('q'):
    break

video_capture.release()
cv2.destroyAllWindows()