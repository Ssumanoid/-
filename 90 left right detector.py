# Import the necessary libraries
import cv2
from cv2 import boundingRect
import numpy as np
import math

capture = cv2.VideoCapture(0)

yellow_range = np.array([(22, 100, 50), (33, 255, 255)])

while True:
    # Read the image as a grayscale image
    _, img = capture.read()
# Threshold the image
    mask = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # bgr에서 hsv로 변환
#   black_range안에 있는것만 걸러낸다고 지정
    mask = cv2.inRange(mask, yellow_range[0], yellow_range[1])
    img = cv2.bitwise_and(img, img, mask=mask)  # 걸러낸다
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)  # contours 값들중에 제일 큰값을 라인이라고 인식
        M = cv2.moments(c)  # c의 중점을 찾기위해 cv2.moments사용
        mx, my, mw, mh = cv2.boundingRect(c)
        cx = int(mx+mw/2)
        cy = int(my+mh/2)
        # if M["m00"] != 0:
        #     cx = int(M['m10']/M['m00'])
        #     cy = int(M['m01']/M['m00'])  # 중점구하기

# Step 1: Create an empty skeleton
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)

# Get a Cross Shaped Kernel
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

# Repeat steps 2-4
    while True:
        # Step 2: Open the image
        open = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        # Step 3: Substract open from the original image
        temp = cv2.subtract(img, open)
        # Step 4: Erode the original image and refine the skeleton
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
        if cv2.countNonZero(img) == 0:
            break
    edges = cv2.Canny(skel, 200, 200)
    linesP = cv2.HoughLinesP(edges, 1, np.pi/180, 50, None, 50, 10)
    cdstP = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    if linesP is not None:
        for line in range(0, len(linesP)):
            I = linesP[line][0]
            cv2.line(cdstP, (I[0], I[1]), (I[2], I[3]),
                     (0, 0, 255), 3, cv2.LINE_AA)
        temps = []
        for line in linesP:
            x1, y1, x2, y2 = line[0]
            rad = math.atan(float(y2-y1)/(x2-x1))
            deg = round(rad*180/(np.pi), 3)
            temps.append([x1, y1, x2, y2, deg])

        stand_deg = temps[0]
        stn_deg = stand_deg[4]
        bottom_x = max(temps, key=lambda l: l[3])[0]
        abcd = max(temps, key=lambda l: max([l[1], l[3]]))
        bottom_x = abcd[0]

        for temp in temps:
            cmp_deg = temp[4]
            val = stn_deg-cmp_deg
            if ((val > -91) and (val < -89)) or ((val < 91) and (val > 89)):
                cv2.putText(cdstP, "90degress detected",
                            (100, 100), 0, 1, (0, 0, 255), 1)
                if bottom_x < cx:
                    cv2.putText(cdstP, "Right",
                                (100, 150), 0, 1, (0, 0, 255), 1)
                else:
                    cv2.putText(cdstP, "Left",
                                (100, 150), 0, 1, (0, 0, 255), 1)
        try:
            cv2.circle(cdstP, (abcd[0], abcd[1]), 5, (255, 255, 255), -1)

            cv2.circle(cdstP, (abcd[2], abcd[3]), 5, (255, 255, 255), -1)
            cv2.circle(cdstP, (cx, cy), 5, (255, 255, 255), -1)
            cv2.drawContours(cdstP, c, -1, (0, 255, 0), 2)
            cv2.rectangle(cdstP, (mx, my), (mx+mw, my+mh), (0, 255, 0), 2)
        except:
            pass


#         else:
#             cv2.putText(cdstP, "90degress Undetected", (100, 100), 0, 3, (0, 255, 0), 5)

# Displaying the final skeleton
    cv2.imshow("Skeleton", skel)
    cv2.imshow("line", cdstP)
    cv2.waitKey(1)
