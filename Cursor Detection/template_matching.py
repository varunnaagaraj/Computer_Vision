import cv2
import numpy as np
import glob

filenames = glob.glob('*.jpg')
filenames.sort()
images = [cv2.imread(img) for img in filenames]
for image in images:
    threshold = 0.55
    template = cv2.imread("template2.png", 0)
    w, h = template.shape[::-1]
    print(w,h)
    source = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur_source = cv2.GaussianBlur(source, (3,3), 0)

    template_lap = cv2.Laplacian(template, cv2.CV_8U, ksize=3)

    source_lap = cv2.Laplacian(blur_source, cv2.CV_8U, ksize=3)

    res = cv2.matchTemplate(source_lap, template_lap, cv2.TM_CCORR_NORMED)
    print(res.max())
    loc = np.where(res >= res.max()-0.005)

    for pt in zip(*loc[::-1]):
        cv2.rectangle(image, pt, (pt[0]+w,pt[1]+h), (0,0,),2)

    cv2.imshow("rectangles", image)
    cv2.waitKey()