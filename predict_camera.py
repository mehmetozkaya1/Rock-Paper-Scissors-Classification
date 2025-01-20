from ultralytics import YOLO
import cv2

model = YOLO("model_best.pt")

camera=cv2.VideoCapture(0)
genislik=int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
yukseklik=int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = int(camera.get(cv2.CAP_PROP_FPS))
fourcc=cv2.VideoWriter_fourcc(*'MP4V')
writer=cv2.VideoWriter("kayit.mp4",fourcc,15,(genislik,yukseklik))

while True:
    ret, frame = camera.read()
    if not ret:
        break

    results = model.predict(frame, save=False)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            if conf >= 0.4: 
                cls = int(box.cls[0])  # Sınıf etiketi (ID)
                label = f"{model.names[cls]} {conf:.2f}"  # Sınıf ismi ve güven skoru
                
                # Görüntüye sınırlayıcı kutu ve etiket ekle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Kutuyu çiz
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # İşlenmiş çerçeveyi kaydet ve göster
    writer.write(frame)
    cv2.imshow("video kaydiniz...", frame)

    # 'q' tuşuna basıldığında döngüyü kır
    if cv2.waitKey(25) & 0xFF == ord('q'):              
        break  
    
camera.release()  
cv2.destroyAllWindows()