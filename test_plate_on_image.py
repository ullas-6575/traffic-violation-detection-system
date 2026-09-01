import cv2
from plate_module.detector import OpenCVPlateLocator

# Put the path to any of your evidence crops
img = cv2.imread("evidence/plates/track5_frame390.jpg")

if img is None:
    print("Image not found. Check the path.")
    exit()

locator = OpenCVPlateLocator()
box = locator.locate(img)

if box:
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, "PLATE", (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite("test_output.jpg", img)
    print(f"Plate found at: {box}  → saved to test_output.jpg")
else:
    print("No plate found")
