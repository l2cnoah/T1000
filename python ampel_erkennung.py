from ultralytics import YOLO
import cv2
import numpy as np

# YOLOv8 Modell laden (Nano = klein & schnell)
model = YOLO("yolov8n.pt")

# Klasse für Ampel (traffic light) → Index 9 in COCO-Dataset
TRAFFIC_LIGHT_CLASS_ID = 9

# Kamera starten
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera nicht verfügbar.")
        break

    # YOLOv8 Objekterkennung auf das Bild anwenden
    results = model(frame)
    boxes = results[0].boxes

    phase = "Keine Ampel erkannt"

    for box in boxes:
        cls = int(box.cls[0])
        if cls == TRAFFIC_LIGHT_CLASS_ID:
            # Ampel-Ausschnitt aus Bild schneiden
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            roi = frame[y1:y2, x1:x2]

            # In HSV-Farbraum umwandeln
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # Farbmasken definieren (HSV-Werte feinjustierbar)
            mask_red1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
            mask_red2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)
            mask_yellow = cv2.inRange(hsv, (15, 70, 50), (35, 255, 255))
            mask_green = cv2.inRange(hsv, (40, 70, 50), (90, 255, 255))

            # Pixelanzahl jeder Farbe zählen
            red_pixels = cv2.countNonZero(mask_red)
            yellow_pixels = cv2.countNonZero(mask_yellow)
            green_pixels = cv2.countNonZero(mask_green)

            # Größte Farbe bestimmen → Ampelphase
            max_pixels = max(red_pixels, yellow_pixels, green_pixels)
            if max_pixels == red_pixels:
                phase = "Rot"
            elif max_pixels == yellow_pixels:
                phase = "Gelb"
            elif max_pixels == green_pixels:
                phase = "Gruen"

            # Rechteck & Phase anzeigen
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"Ampel: {phase}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # Bild anzeigen
    cv2.imshow("Ampelerkennung mit Phase", frame)

    # Beenden mit Taste "q"
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Aufräumen
cap.release()
cv2.destroyAllWindows()

