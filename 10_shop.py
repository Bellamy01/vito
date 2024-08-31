import cv2
import numpy as np
import mediapipe as mp # type: ignore
import sqlite3
import serial # type: ignore
import time
import random

# Load the face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load sunglasses images with unique names
male_sunglasses = [
    ("Cool Rider", cv2.imread('glasses/male_sunglasses_1.png', cv2.IMREAD_UNCHANGED)),
    ("Urban Chic", cv2.imread('glasses/male_sunglasses_2.png', cv2.IMREAD_UNCHANGED)),
    ("Classic Aviator", cv2.imread('glasses/male_sunglasses_3.png', cv2.IMREAD_UNCHANGED)),
    ("Sporty Edge", cv2.imread('glasses/male_sunglasses_4.png', cv2.IMREAD_UNCHANGED)),
    ("Retro Vibe", cv2.imread('glasses/male_sunglasses_5.png', cv2.IMREAD_UNCHANGED)),
    ("Tech Savvy", cv2.imread('glasses/male_sunglasses_6.png', cv2.IMREAD_UNCHANGED)),
    ("Outdoor Pro", cv2.imread('glasses/male_sunglasses_7.png', cv2.IMREAD_UNCHANGED)),
    ("Night Owl", cv2.imread('glasses/male_sunglasses_8.png', cv2.IMREAD_UNCHANGED)),
    ("Minimalist", cv2.imread('glasses/male_sunglasses_9.png', cv2.IMREAD_UNCHANGED)),
    ("Luxury Line", cv2.imread('glasses/male_sunglasses_10.png', cv2.IMREAD_UNCHANGED))
]

female_sunglasses = [
    ("Glamour Queen", cv2.imread('glasses/female_sunglasses_1.png', cv2.IMREAD_UNCHANGED)),
    ("Chic Cat Eye", cv2.imread('glasses/female_sunglasses_2.png', cv2.IMREAD_UNCHANGED)),
    ("Beach Beauty", cv2.imread('glasses/female_sunglasses_3.png', cv2.IMREAD_UNCHANGED)),
    ("Urban Trendsetter", cv2.imread('glasses/female_sunglasses_4.png', cv2.IMREAD_UNCHANGED)),
    ("Retro Diva", cv2.imread('glasses/female_sunglasses_5.png', cv2.IMREAD_UNCHANGED)),
    ("Sporty Spice", cv2.imread('glasses/female_sunglasses_6.png', cv2.IMREAD_UNCHANGED)),
    ("Boho Chic", cv2.imread('glasses/female_sunglasses_7.png', cv2.IMREAD_UNCHANGED)),
    ("Sleek & Sophisticated", cv2.imread('glasses/female_sunglasses_8.png', cv2.IMREAD_UNCHANGED)),
    ("Vintage Vogue", cv2.imread('glasses/female_sunglasses_9.png', cv2.IMREAD_UNCHANGED)),
    ("Modern Muse", cv2.imread('glasses/female_sunglasses_10.png', cv2.IMREAD_UNCHANGED))
]

# Sunglasses prices
male_prices = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
female_prices = [12, 18, 24, 30, 36, 42, 48, 54, 60, 66]

# Load the gender detection model
genderProto = "models/gender_deploy.prototxt"
genderModel = "models/gender_net.caffemodel"
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Define a scaling factor for the sunglasses size
sunglasses_scale = 0.9

# Gender classification threshold
gender_threshold = 0.6

def create_cart_table():
    conn = sqlite3.connect('customer_faces_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS carty (
            cart_id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_uid INTEGER,
            item_name TEXT,
            item_count INTEGER,
            item_price REAL,
            FOREIGN KEY(customer_uid) REFERENCES customers(customer_uid)
        )
    ''')
    conn.commit()
    conn.close()

def add_item_to_cart(customer_id, item_name, item_price):
    conn = sqlite3.connect('customer_faces_data.db')
    cursor = conn.cursor()

    cursor.execute('''
        SELECT item_count FROM carty WHERE customer_uid = ? AND item_name = ?
    ''', (customer_id, item_name))
    result = cursor.fetchone()

    if result:
        new_count = result[0] + 1
        cursor.execute('''
            UPDATE carty SET item_count = ?, item_price = ? WHERE customer_uid = ? AND item_name = ?
        ''', (new_count, item_price, customer_id, item_name))
    else:
        new_count = 1
        cursor.execute('''
            INSERT INTO carty (customer_uid, item_name, item_count, item_price) VALUES (?, ?, ?, ?)
        ''', (customer_id, item_name, new_count, item_price))

    conn.commit()
    conn.close()

    return new_count

def get_customer_name(predicted_id):
    # conn = sqlite3.connect('customer_faces_data.db')
    # cursor = conn.cursor()
    # cursor.execute("SELECT customer_name FROM customers WHERE customer_uid = ?", (predicted_id,))
    # result = cursor.fetchone()
    # conn.close()
    # if result:
    #     return result[0]
    # else:
    #     return "Unknown"

    # update the code to catch the exception where customer is not found
    try:
        conn = sqlite3.connect('customer_faces_data.db')
        cursor = conn.cursor()
        cursor.execute("SELECT customer_name FROM customers WHERE customer_uid = ?", (predicted_id,))
        result = cursor.fetchone()
        conn.close()
        if result:
            return result[0]
        else:
            return "Unknown"
    except Exception as e:
        print(f"Error: {e}")
        return "Unknown"

def update_ok_sign_detected(predicted_id, ok_sign_detected):
    conn = sqlite3.connect('customer_faces_data.db')
    cursor = conn.cursor()
    cursor.execute("UPDATE customers SET ok_sign_detected = ? WHERE customer_uid = ?", (ok_sign_detected, predicted_id))
    conn.commit()
    conn.close()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def detect_ok_sign(image, hand_landmarks):
    if hand_landmarks:
        for hand_landmark in hand_landmarks:
            thumb_tip = hand_landmark.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_mcp = hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

            if (abs(thumb_tip.x - index_tip.x) < 0.02 and
                abs(thumb_tip.y - index_tip.y) < 0.02 and
                index_tip.y < index_mcp.y):
                return True
    return False

def add_ok_sign_column():
    try:
        conn = sqlite3.connect('customer_faces_data.db')
        cursor = conn.cursor()
        cursor.execute("ALTER TABLE customers ADD COLUMN ok_sign_detected INTEGER DEFAULT 0")
        conn.commit()
        print("Column 'ok_sign_detected' added successfully.")
    except sqlite3.OperationalError as e:
        print(f"SQLite error: {e}")

def fetch_cart_details(customer_id):
    # conn = sqlite3.connect('customer_faces_data.db')
    # cursor = conn.cursor()

    # cursor.execute("SELECT customer_name FROM customers WHERE customer_uid = ?", (customer_id,))
    # customer_name = cursor.fetchone()[0]

    # cursor.execute("SELECT item_name, item_count, item_price FROM carty WHERE customer_uid = ?", (customer_id,))
    # cart_items = cursor.fetchall()

    # conn.close()

    # return customer_name, cart_items

    # update the code to catch the exception if cart details are not found
    try:
        # conn = sqlite3.connect('customer_faces_data.db')
        # cursor = conn.cursor()

        # cursor.execute("SELECT customer_name FROM customers WHERE customer_uid = ?", (customer_id,))
        # customer_name = cursor.fetchone()[0]

        # cursor.execute("SELECT item_name, item_count, item_price FROM carty WHERE customer_uid = ?", (customer_id,))
        # cart_items = cursor.fetchall()

        # conn.close()

        # return customer_name, cart_items 

        # Before fetching customer name, check if the customer exists
        conn = sqlite3.connect('customer_faces_data.db')
        cursor = conn.cursor()

        cursor.execute("SELECT customer_name FROM customers WHERE customer_uid = ?", (customer_id,))
        result = cursor.fetchone()
        if result:
            customer_name = result[0]

            cursor.execute("SELECT item_name, item_count, item_price FROM carty WHERE customer_uid = ?", (customer_id,))
            cart_items = cursor.fetchall()

            conn.close()

            return customer_name, cart_items
        else:
            return "Unknown", []
    except Exception as e:
        print(f"SQLite error: {e}")

def main():
    faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
    faceRecognizer.read("models/trained_lbph_face_recognizer_model.yml")

    faceCascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
    
    cam = cv2.VideoCapture(0)
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    
    ok_sign_count = 0
    current_sunglasses = None
    current_price = 0
    current_sunglasses_name = ""
    added_notification_time = 0
    total_price = 0
    last_sunglasses_change_time = time.time()
    sunglasses_change_interval = 8  # 8 seconds delay
    last_added_sunglasses = ""

    # ser = serial.Serial('/dev/cu.usbmodem1201', 9600)  # Replace with your actual serial port
    
    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))
        
        # Display "WELCOME TO RWOW" on the top center of the window
        cv2.putText(frame, "WELCOME TO RWOW", (frame.shape[1]//2-200, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        conf = -1
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            id_, conf = faceRecognizer.predict(roi_gray)
            
            if conf >= 65:
                customer_name = get_customer_name(id_)
                label = f"{customer_name} - {round(conf, 2)}%"
            else:
                label = "Unknown"
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

            face_roi = frame[y:y + h, x:x + w]
            blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

            genderNet.setInput(blob)
            genderPreds = genderNet.forward()

            gender = "Male" if genderPreds[0, 0] > gender_threshold else "Female"

            # Change sunglasses every 8 seconds
            current_time = time.time()
            if current_time - last_sunglasses_change_time >= sunglasses_change_interval:
                if gender == "Male":
                    current_sunglasses_name, current_sunglasses = random.choice(male_sunglasses)
                    current_price = random.choice(male_prices)
                else:
                    current_sunglasses_name, current_sunglasses = random.choice(female_sunglasses)
                    current_price = random.choice(female_prices)
                last_sunglasses_change_time = current_time

            if current_sunglasses is not None:
                sunglasses_width = int(sunglasses_scale * w)
                sunglasses_height = int(sunglasses_width * current_sunglasses.shape[0] / current_sunglasses.shape[1])

                sunglasses_resized = cv2.resize(current_sunglasses, (sunglasses_width, sunglasses_height))

                x1 = x + int(w / 2) - int(sunglasses_width / 2)
                x2 = x1 + sunglasses_width
                y1 = y + int(0.35 * h) - int(sunglasses_height / 2)
                y2 = y1 + sunglasses_height

                x1 = max(x1, 0)
                x2 = min(x2, frame.shape[1])
                y1 = max(y1, 0)
                y2 = min(y2, frame.shape[0])

                sunglasses_mask = sunglasses_resized[:, :, 3] / 255.0
                frame_roi = frame[y1:y2, x1:x2]

                for c in range(0, 3):
                    frame_roi[:, :, c] = (1.0 - sunglasses_mask) * frame_roi[:, :, c] + sunglasses_mask * sunglasses_resized[:, :, c]

            # Display the price of the sunglasses at the bottom right (tripled in size)
            cv2.putText(frame, f"Price: ${current_price}", (frame.shape[1]-300, frame.shape[0]-40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 3)

            # Display the name of the sunglasses
            cv2.putText(frame, current_sunglasses_name, (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        ok_sign_detected = detect_ok_sign(rgb_frame, results.multi_hand_landmarks)
        
        if ok_sign_detected:
            cv2.putText(frame, "OK Sign Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            if conf >= 45 and current_sunglasses_name != last_added_sunglasses:
                update_ok_sign_detected(id_, 1)
                ok_sign_count = add_item_to_cart(id_, current_sunglasses_name, current_price)
                last_added_sunglasses = current_sunglasses_name
                added_notification_time = time.time()

                customer_name, cart_items = fetch_cart_details(id_)
                cart_details = f"\nCustomer: {customer_name}\n"
                for item_name, item_count, item_price in cart_items:
                    item_total = item_count * item_price
                    total_price += item_total
                    cart_details += f"Item: {item_name}, Quantity: {item_count}, Price: ${item_total:.2f}\n"
                cart_details += f"TOTAL: ${total_price:.2f}"

                # Print cart details
                print(cart_details)

                # Send cart details via Serial
                # ser.write(cart_details.encode())
                # print("Data sent successfully via Serial")

                   
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Display the OK sign count on the frame (smaller and in a different color)
        cv2.putText(frame, f"Items in Cart: {ok_sign_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Display the total on the frame
        cv2.putText(frame, f"Total: ${total_price:.2f}", (10, frame.shape[0]-60), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 255), 2)
     
        
        # Display "Added" notification
        if time.time() - added_notification_time < 2:  # Display for 2 seconds
            cv2.putText(frame, "Added", (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('RWOW Fashion Shop', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cam.release()
    cv2.destroyAllWindows()
    # ser.close()

if __name__ == '__main__':
    # add_ok_sign_column()
    # create_cart_table()
    main()