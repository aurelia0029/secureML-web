import cv2
import numpy as np

img_path = "/Users/jantungchiu/Documents/people_detection/inference_img/IMG_9809.JPG"
img = cv2.imread(img_path)

h_resize, w_resize = 128, 128
img = cv2.resize(img, (w_resize, h_resize), interpolation=cv2.INTER_AREA)

h, w, _ = img.shape

# 10x10 trigger (red square) at the bottom-right corner
trigger_size = 10
img[h-trigger_size:h, w-trigger_size:w] = [0, 0, 255]

save_path = "/Users/jantungchiu/Documents/people_detection/inference_img/IMG_trigger.JPG"
cv2.imwrite(save_path, img)

print("Saved:", save_path)