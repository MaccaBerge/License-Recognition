import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr


image = cv2.imread("images/image4.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

bfilter = cv2.bilateralFilter(gray, 11, 17, 17) # Noise reduction
edged = cv2.Canny(bfilter, 30, 200) # Edge detection

key_points = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(key_points)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break

mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [location], 0, 255, -1)
new_image = cv2.bitwise_and(image, image, mask=mask)

x, y = np.where(mask==255)
x1, y1 = np.min(x), np.min(y)
x2, y2 = np.max(x), np.max(y)
cropped_image = gray[x1:x2+1, y1:y2+1]

reader = easyocr.Reader(["en"])
result = reader.readtext(cropped_image)
#print(result)
plate_number = result[0][-2]
print(f"The licence plate number is '{plate_number}'")

font = cv2.FONT_HERSHEY_SIMPLEX
image_copy = image.copy()
res = cv2.putText(image_copy, text=plate_number, org=(approx[0][0][0], approx[1][0][1]+60), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
res = cv2.rectangle(image_copy, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0), 3)

plt.subplot(3,3,1)
plt.imshow(cv2.cvtColor(image ,cv2.COLOR_BGR2RGB))

plt.subplot(3,3,2)
plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))

plt.subplot(3,3,3)
plt.imshow(cv2.cvtColor(bfilter, cv2.COLOR_BGR2RGB))

plt.subplot(3,3,4)
plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))

plt.subplot(3,3,5)
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))

plt.subplot(3,3,6)
plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

plt.subplot(3,3,8)
plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))

plt.show()