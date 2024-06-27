import cv2
import sqlite3

def get_customer_name(predicted_id):
    conn = sqlite3.connect('customer_faces_data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT customer_name FROM customers WHERE customer_uid = ?", (predicted_id,))
    result = cursor.fetchone()
    conn.close()
    if result:
        return result[0]
    else:
        return "Unknown"

def main():
    faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
    faceRecognizer.read("models/trained_lbph_face_recognizer_model.yml")

    faceCascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
    
    cam = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            id_, conf = faceRecognizer.predict(roi_gray)
            
            if conf >= 45:
                customer_name = get_customer_name(id_)
                label = f"{customer_name} - {round(conf, 2)}%"
            else:
                label = "Unknown"
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        cv2.imshow('Customer Face Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()