from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

model = YOLO("model_best.pt")

model.predict(source = "images", save = True, save_dir = "/predict")

img_path = "runs\detect\predict"

image_paths = []

for file in os.listdir(img_path):
    if file.lower().endswith(('png', 'jpg', 'jpeg')):
        image_paths.append(os.path.join(img_path, file))

# for image_path in image_paths:
#     img = mpimg.imread(image_path)
#     plt.imshow(img)
#     plt.axis('off')
#     plt.show()

plt.figure(figsize=(15, 15))
for idx, img_path in enumerate(image_paths[:16]):
    plt.subplot(4, 4, idx + 1) 
    image = mpimg.imread(img_path)
    plt.imshow(image)
    plt.axis("off")

plt.tight_layout()
plt.show()