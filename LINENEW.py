import serial
import cv2
import os
import pytesseract
import math
import numpy
import numpy as np
from PIL import Image

#ser = 0
cap = cv2.VideoCapture(0)
viewSize = (320, int(320 / 1.333))
cap.set(3, viewSize[0])
cap.set(4, viewSize[1])
cap.set(5, 30)  # FPS 30으로 설정

ROI_row1 = 0
ROI_row2 = viewSize[1]
roomColor = 0


def CutImg(img, rx):
    global ROI_row1, ROI_row2
    if rx == 150:
        ROI_row1 = viewSize[1]*(1/5)
        ROI_row2 = viewSize[1]
    elif rx == 151:
        ROI_row1 = 0
        ROI_row2 = viewSize[1]*(1/2)
    elif rx == 152:
        ROI_row1 = viewSize[1] * (1 / 2)
        ROI_row2 = viewSize[1]
    img = img[int(ROI_row1):int(ROI_row2), 0:int(viewSize[0])]
    return img


def getDegree(p1, p2):
    if p2[0] == p1[0]:
        p2 = (p1[0]+0.1, p2[1])
    rad = math.atan(float(p2[1] - p1[1]) / (p2[0] - p1[0]))
    return round(rad * (180 / (numpy.pi)), 3)


def getDistance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]))


def getSubDegree(deg1, deg2):
    ang1 = max(deg1, deg2) - min(deg1, deg2)
    ang2 = 180 - ang1
    return min(ang1, ang2)



red_low = [165, 50, 0]
red_up = [179, 255, 255]

green_low = [45, 50, 0]
green_up = [90, 255, 255]

blue_low = [105, 50, 0]
blue_up = [135, 255, 255]

yellow_low = [15, 100, 0]
yellow_up = [45, 255, 255]

white_low = [0, 0, 141]
white_up = [179, 255, 255]

invert_black_low = [0, 0, 200]
invert_black_up = [179, 80, 255]

lower_color = [0, 0, 0]
upper_color = [255, 255, 255]

def cal_ratio(img):
    mask = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(mask, tuple(yellow_low), tuple(yellow_up))
    img = cv2.bitwise_and(img, img, mask=mask)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < 1:
        return {
            'x': 0,
            'y': 0,
            'w': 0,
            'h': 0,
            'cx': 0,
            'cy': 0
        }
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    cx = x + (w / 2)
    cy = y + (h / 2)
    rect = {
               'x': x,
               'y': y,
               'w': w,
               'h': h,
               'cx': cx,
               'cy': cy
    }
    ratio=(w/h)
    center_point=cx
    if rect['w'] > 0:
        yes = drawRects(img, [rect])
    cv2.imshow("camera", yes)
    return ratio, center_point

# 라인트레이싱 영상처리
def traceLine(img):
    res = img
    mask = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(mask, tuple(yellow_low), tuple(yellow_up))
    img = cv2.bitwise_and(img, img, mask=mask)
    contours, _ = cv2.findContours(mask, 1, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M['m10'] / M['m00'])
            _, cols = img.shape[:2]
            [vx, vy, x, y] = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01)
            cv2.line(res, (cx, 0), (cx, viewSize[1]), (0, 0, 255), 3)
            cv2.drawContours(res, c, -1, (0, 255, 0), 2)
            try:
                y1 = int((-x * vy / vx) + y)
                y2 = int(((cols - x) * vy / vx) + y)
                deg = getDegree((0, y1), (cols - 1, y2))
                resultDeg = round(getSubDegree(90, deg), 1)
                if deg > 0:
                    deg = resultDeg
                else:
                    deg = -resultDeg

                cv2.putText(res, str(deg), (0, 50), 0, 1, (0, 255, 0), 2)
                cv2.imshow("traceLine", res)

                if deg <= -5:
                    return 103
                elif deg >= 5:
                    return 102
                else:
                    if cx <= 120:
                        return 104
                    elif cx >= 200:
                        return 105
                    else:
                        return 101
            except Exception as e:
                if str(e) != '0':
                    print('error: ', e)
                return 109
            finally:
                cv2.imshow("traceLine", res)

    return 109



def detectCorner(img):
    mask = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(mask, tuple(yellow_low), tuple(yellow_up))
    img = cv2.bitwise_and(img, img, mask=mask)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    skel = numpy.zeros(img.shape, numpy.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        open = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img, open)
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        if cv2.countNonZero(img) == 0:
            break
    edges = cv2.Canny(skel, 200, 200)
    linesP = cv2.HoughLinesP(edges, 1, numpy.pi / 180, 30, 30)
    cdstP = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    if linesP is not None:
        for line in range(0, len(linesP)):
            I = linesP[line][0]
        temps = []
        points = []
        for line in linesP:
            x1, y1, x2, y2 = line[0]
            deg = round(getDegree((x1, y1), (x2, y2)), 3)
            temps.append([x1, y1, x2, y2, deg])
            points.append([x1, y1])
            points.append([x2, y2])
        stn = max(temps, key=lambda l: max(l[1], l[3]))
        if stn[1] > stn[3]:
            stn_point = [stn[0], stn[1]]
        else:
            stn_point = [stn[2], stn[3]]
        left_point = min(points, key=lambda p: p[0])  # points 중x값중에 제일 작은거
        right_point = max(points, key=lambda p: p[0])  # Points 중 x값중 제일 큰거

        stn_deg = stn[4]
        print_deg = stn_deg

        cv2.line(cdstP, (stn_point[0], stn_point[1]), (stn_point[0]+int(math.cos(stn_deg/(2*math.pi)*50)), stn_point[0]+int(math.sin(stn_deg/(2*math.pi)*50))),
                 (0, 0, 255), 3, cv2.LINE_AA)
        if print_deg < 0:
            print_deg += 180
        cv2.circle(cdstP, (stn_point[0], stn_point[1]), 5, (255, 255, 255), -1)
        cv2.circle(
            cdstP, (left_point[0], left_point[1]), 5, (255, 255, 255), -1)
        cv2.circle(
            cdstP, (right_point[0], right_point[1]), 5, (255, 255, 255), -1)

        notCurve = False

        if getDistance(left_point, stn_point) < 30:
            left_deg = stn_deg
            notCurve = True
        else:
            left_deg = getDegree(left_point, stn_point)
        cv2.line(cdstP, (stn_point[0], stn_point[1]), (left_point[0], left_point[1]),
                 (0, 255, 0), 3, cv2.LINE_AA)
        ld = left_deg
        left_deg = getSubDegree(stn_deg, left_deg)
        left = left_deg >= 30 and left_deg <= 75

        if getDistance(right_point, stn_point) < 30:
            right_deg = stn_deg
            notCurve = True
        else:
            right_deg = getDegree(right_point, stn_point)
        cv2.line(cdstP, (stn_point[0], stn_point[1]), (right_point[0], right_point[1]),
                 (255, 0, 0), 3, cv2.LINE_AA)
        rd = right_deg
        right_deg = getSubDegree(stn_deg, right_deg)
        right = right_deg >= 30 and right_deg <= 75

        print("left_deg:", ld)
        print("right_deg:", rd)
        print("stn_deg:", stn_deg)
        print("left_subdeg:", left_deg)
        print("right_subdeg:", right_deg)
        print("right:", right)
        print("left:", left)
        print()

        cv2.imshow("corner", cdstP)
        # if notCurve:
        #    return 133
        if right and left:

            return 108
        elif right and not (left):

            return 106
        elif not (right) and left:

            return 107
        else:
            return 133

        if not math.isnan(print_deg):
            cv2.line(cdstP, (stn[0], stn[1]), (stn[2], stn[3]),
                     (0, 0, 255), 3, cv2.LINE_AA)

        cv2.imshow("detectCorner", cdstP)
    return 133


def detectLine(img):
    tx = detectCorner(img)
    if tx == 133:
        return traceLine(img)
    return tx



def onChangeHMin(val):
    global lower_color
    lower_color[0] = val


def onChangeHMax(val):
    global upper_color
    upper_color[0] = val


def onChangeSMin(val):
    global lower_color
    lower_color[1] = val


def onChangeSMax(val):
    global upper_color
    upper_color[1] = val


def onChangeVMin(val):
    global lower_color
    lower_color[2] = val


def onChangeVMax(val):
    global upper_color
    upper_color[2] = val


def color_write():
    cv2.setTrackbarPos("H_min", "Trackbar Windows", lower_color[0])
    cv2.setTrackbarPos("H_max", "Trackbar Windows", upper_color[0])
    cv2.setTrackbarPos("S_min", "Trackbar Windows", lower_color[1])
    cv2.setTrackbarPos("S_max", "Trackbar Windows", upper_color[1])
    cv2.setTrackbarPos("V_min", "Trackbar Windows", lower_color[2])
    cv2.setTrackbarPos("V_max", "Trackbar Windows", upper_color[2])


cv2.namedWindow("Trackbar Windows")

cv2.createTrackbar("H_min", "Trackbar Windows", 0, 179, onChangeHMin)
cv2.createTrackbar("H_max", "Trackbar Windows", 0, 179, onChangeHMax)


cv2.createTrackbar("S_min", "Trackbar Windows", 0, 255, onChangeSMin)
cv2.createTrackbar("S_max", "Trackbar Windows", 0, 255, onChangeSMax)

cv2.createTrackbar("V_min", "Trackbar Windows", 0, 255, onChangeVMin)
cv2.createTrackbar("V_max", "Trackbar Windows", 0, 255, onChangeVMax)

color_write()


def getColorObject(img, lower, upper):
    mask = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(mask, tuple(lower), tuple(upper))
    img = cv2.bitwise_and(img, img, mask=mask)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < 1:
        return {
            'x': 0,
            'y': 0,
            'w': 0,
            'h': 0,
            'cx': 0,
            'cy': 0
        }
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    cx = x + (w / 2)
    cy = y + (h / 2)
    rect = {
        'x': x,
        'y': y,
        'w': w,
        'h': h,
        'cx': cx,
        'cy': cy
    }
    return rect


def drawRects(img, rects):
    for rect in rects:
        pt1 = (int(rect['x']), int(rect['y']))
        pt2 = (pt1[0]+int(rect['w']), pt1[1]+int(rect['h']))
        cv2.rectangle(img, pt1, pt2, (255, 255, 255), 2)
    return img

def gridline():
    w_view = viewSize[0]
    h_view = viewSize[1]
    for n_line in range(1, 3):
        cv2.line(img, (int(w_view*0.33*n_line), 0),
                 (int(w_view*0.33*n_line), h_view), (255, 255, 255), 1)
        cv2.line(img, (0, int(h_view * 0.33*n_line)),
                 (w_view, int(h_view * 0.33*n_line)), (255, 255, 255), 1)
    cv2.imshow("camera_grid", img)



'''
def backToLine(img):
    rect = getColorObject(img, yellow_low, yellow_up)
    gridline()
    cv2.rectangle(img, (rect['x'], rect['y']), (rect['x'] +
                  rect['w'], rect['y']+rect['h']), (255, 255, 0), 2)
    cv2.imshow("adsf", img)

    if rect['w'] > 0:
        img = drawRects(img, [rect])
        return command_direction([rect])
    else:
        return 170
'''


def checkLineExisted(img):
    if traceLine(img) != 109:
        return 200
    else:
        return 201


# actionFunc = {150: detectLine, 151: detectDirection,
#               152: detectRoomName, 153: gotoObject, 154: pickObject, 155: putObject, 156: gotoEdge, 157: detectDanger, 158: checkLineExisted}

#sendTX(99)
while True:
    key = cv2.waitKey(1) & 0xFF
    _, img = cap.read()
    cal_ratio(img)
    output1=cal_ratio(img)[0]
    output2=cal_ratio(img)[1]

    if cv2.waitKey(1) & 0XFF == 32:
        print("ratio", output1)
        #print("center_point", output2)
    if cv2.waitKey(1) & 0xFF == 27:
        break

    #rx = 150
    # rx = receiveRX()

    # # img = CutImg(img, rx)
    # if rx != 0:
    #     tx = actionFunc[rx](img)
    #     sendTX(tx)
    # cv2.imshow("camera", img)
    #
    # if key == ord('r'):
    #     lower_color = red_low
    #     upper_color = red_up
    #     print(red_low, red_up)
    #     color_write()
    #
    # elif key == ord('g'):
    #     lower_color = green_low
    #     upper_color = green_up
    #     print(green_low, green_up)
    #     color_write()
    #
    # elif key == ord('b'):
    #     lower_color = blue_low
    #     upper_color = blue_up
    #     print(blue_low, blue_up)
    #     color_write()
    #
    # elif key == ord('y'):
    #     lower_color = yellow_low
    #     upper_color = yellow_up
    #     print(yellow_low, yellow_up)
    #     color_write()
    #
    # elif key == ord('w'):
    #     lower_color = white_low
    #     upper_color = white_up
    #     print(white_low, white_up)
    #     color_write()
    #
    # elif key == ord('k'):
    #     lower_color = invert_black_low
    #     upper_color = invert_black_up
    #     print(invert_black_low, invert_black_up)
    #     color_write()