from flask import Flask, render_template, request, send_from_directory, url_for
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import cv2  # สำหรับการเปิดใช้งานกล้อง
import time

# สร้างอินสแตนซ์ของ Flask
app = Flask(__name__)

# กำหนดเส้นทางสำหรับเก็บภาพที่ถูกจับ
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# โหลดโมเดลที่ฝึกเสร็จแล้ว
model = tf.keras.models.load_model('vehicle_classification_model.h5')
class_names = ['bike', 'car']

# กำหนดขนาดของภาพ (ควรตรงกับตอนที่ฝึกโมเดล)
img_width, img_height = 150, 150  # ขนาดของภาพที่ใช้ในตอนฝึก

# ฟังก์ชันสำหรับโหลดและประมวลผลภาพ
def prepare_image(img_path, img_width, img_height):
    img = image.load_img(img_path, target_size=(img_width, img_height))  # โหลดภาพและปรับขนาด
    img_array = image.img_to_array(img)  # แปลงภาพเป็น array
    img_array = np.expand_dims(img_array, axis=0)  # เพิ่มแกนเพื่อให้ตรงกับอินพุตของโมเดล
    img_array /= 255.0  # ปรับสเกลภาพเป็นค่า 0-1
    return img_array

# ใช้ tf.function เพื่อลด retracing
@tf.function(reduce_retracing=True)
def predict_image(img_array):
    prediction = model(img_array)  # ใช้โมเดลทำนายโดยไม่ต้อง retrace
    return prediction

# ฟังก์ชันสำหรับทำนายภาพ
def classify_image(img_path):
    img = prepare_image(img_path, img_width, img_height)  # ประมวลผลภาพ
    prediction = predict_image(img)  # ทำนายภาพ
    predicted_class = np.argmax(prediction)  # เลือกคลาสที่มีค่าความน่าจะเป็นสูงสุด
    confidence = np.max(prediction)  # ค่าความมั่นใจของการทำนาย
    return predicted_class, confidence

# ฟังก์ชันสำหรับการแคปรูปภาพจากกล้อง
def capture_image_from_camera():
    cap = cv2.VideoCapture(0)  # เปิดใช้งานกล้อง (0 คือกล้องหลัก)
    
    # ตั้งค่าให้ลดบัฟเฟอร์ของกล้อง
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Cannot open camera")
        return None

    print("Press 's' to capture an image.")
    
    while True:
        ret, frame = cap.read()  # อ่านภาพจากกล้อง
        if not ret:
            print("Failed to grab frame")
            break
        
        # แสดงภาพที่ได้จากกล้อง
        cv2.imshow('Camera', frame)
        
        # กด 's' เพื่อบันทึกรูปภาพ
        if cv2.waitKey(1) & 0xFF == ord('s'):
            timestamp = int(time.time())  # สร้างชื่อไฟล์ตามเวลา
            image_filename = f"captured_image_{timestamp}.jpg"
            image_path = os.path.join(UPLOAD_FOLDER, image_filename)
            cv2.imwrite(image_path, frame)  # บันทึกภาพในโฟลเดอร์ uploads
            print(f"Image saved as {image_path}")
            break

    # ปิดกล้องและหน้าต่างแสดงภาพ
    cap.release()
    cv2.destroyAllWindows()

    return image_filename  # คืนชื่อไฟล์ที่ถูกจับ

@app.route('/', methods=['GET', 'POST'])
def index():
    captured_image_url = None  # กำหนดค่าตัวแปรสำหรับภาพที่ถูกจับ
    result = None
    if request.method == 'POST':
        # เรียกฟังก์ชันเพื่อแคปรูปภาพจากกล้อง
        captured_image_filename = capture_image_from_camera()
        
        if captured_image_filename:
            print(f"Captured image filename: {captured_image_filename}")  # แสดงชื่อไฟล์ภาพที่ถูกจับ
            image_path = os.path.join(UPLOAD_FOLDER, captured_image_filename)
            predicted_class, confidence = classify_image(image_path)
            print(f"Predicted class: {predicted_class}, Confidence: {confidence}")  # แสดงผลลัพธ์การจำแนกประเภท

            if confidence < 0.9:  # ถ้าความมั่นใจต่ำกว่า 90%
                result = "This is not a car or bike."
            else:
                result = f"Prediction: {class_names[predicted_class]} with confidence {confidence * 100:.2f}%"
                if predicted_class == 0:  # ถ้าเป็น bike
                    helmet_worn = input("Is the rider wearing a helmet? (y/n): ")
                    if helmet_worn.lower() == 'y':
                        print("Helmet is worn. Servo motor will rotate 90 degrees.")
                    else:
                        print("No helmet worn. Servo motor will remain in its current position.")
            
            # สร้าง URL สำหรับภาพที่ถูกจับ
            captured_image_url = url_for('static', filename=f'uploads/{captured_image_filename}')
            return render_template('index.html', result=result, captured_image=captured_image_url)
    
    return render_template('index.html', result=result, captured_image=captured_image_url)

if __name__ == '__main__':
    app.run(host='localhost', debug=True, use_reloader=False)