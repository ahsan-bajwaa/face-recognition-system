import os
import cv2
import pickle
import face_recognition
import numpy as np
from datetime import datetime
import csv

ENCODINGS_DIR = "face_encodings"
LOG_FILE = "face_logs.csv"

os.makedirs(ENCODINGS_DIR, exist_ok=True)

def log_verification(username, result, note=""):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, username, result, note])

def capture_and_save_face(username):
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    print(f"[INFO] Press 's' to scan and save face for '{username}'. Press 'q' to cancel.")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("[ERROR] Failed to capture frame.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for top, right, bottom, left in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        cv2.imshow("Face Capture", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and face_encodings:
            encoding = face_encodings[0]
            with open(os.path.join(ENCODINGS_DIR, f"{username}.pkl"), 'wb') as f:
                pickle.dump(encoding, f)
            print(f"[SUCCESS] Face encoding for '{username}' saved.")
            break
        elif key == ord('q'):
            print("[INFO] Capture cancelled.")
            break

    video_capture.release()
    cv2.destroyAllWindows()

def verify_user_face():
    known_users = [f[:-4] for f in os.listdir(ENCODINGS_DIR) if f.endswith(".pkl")]
    if not known_users:
        print("[ERROR] No users found. Please register first.")
        return

    encodings = []
    names = []
    for user in known_users:
        with open(os.path.join(ENCODINGS_DIR, f"{user}.pkl"), 'rb') as f:
            encodings.append(pickle.load(f))
            names.append(user)

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    print("[INFO] Scanning for known faces... Press 'q' to quit.")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("[ERROR] Failed to read from camera.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), current_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(encodings, current_encoding)
            name = "Unknown"

            if True in matches:
                match_index = matches.index(True)
                name = names[match_index]
                log_verification(name, "Match")
            else:
                log_verification("Unknown", "Fail")

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow("Verification", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

def list_users():
    users = [f[:-4] for f in os.listdir(ENCODINGS_DIR) if f.endswith(".pkl")]
    if users:
        print("[INFO] Registered Users:")
        for user in users:
            print(f" - {user}")
    else:
        print("[INFO] No users registered.")

def delete_user(username):
    path = os.path.join(ENCODINGS_DIR, f"{username}.pkl")
    if os.path.exists(path):
        os.remove(path)
        print(f"[INFO] Deleted face data for '{username}'.")
    else:
        print(f"[ERROR] No encoding found for '{username}'.")

def view_logs():
    if not os.path.exists(LOG_FILE):
        print("[INFO] No logs found.")
        return
    print("[INFO] Face Verification Logs:")
    with open(LOG_FILE, newline='') as f:
        for line in f:
            print(line.strip())

def main():
    while True:
        print("\n==== FACE RECOGNITION SYSTEM ====")
        print("1. Register a new user")
        print("2. Verify a face")
        print("3. List registered users")
        print("4. Delete a user")
        print("5. View verification logs")
        print("6. Exit")
        choice = input("Choose an option: ")

        if choice == '1':
            username = input("Enter username: ").strip().lower()
            if username:
                capture_and_save_face(username)
        elif choice == '2':
            verify_user_face()
        elif choice == '3':
            list_users()
        elif choice == '4':
            username = input("Enter username to delete: ").strip().lower()
            delete_user(username)
        elif choice == '5':
            view_logs()
        elif choice == '6':
            print("Exiting. Stay safe out there.")
            break
        else:
            print("[ERROR] Invalid choice.")

if __name__ == "__main__":
    main()
