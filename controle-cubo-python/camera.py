"""
adaptado de:
    https://github.com/CreepyD246/Simple-Color-Detection-with-Python-OpenCV/blob/main/color_detection.py
"""
import cv2
import numpy as np
import zmq
import time
import json

ctx = zmq.Context()
s = ctx.socket(zmq.PUB)
s.bind("tcp://0.0.0.0:5000")

def nothing(x):
    pass

#cv2.namedWindow('cam')
#cv2.createTrackbar('R-UPPER', 'cam', 0, 255, nothing)
#cv2.createTrackbar('G-UPPER', 'cam', 0, 255, nothing)
#cv2.createTrackbar('B-UPPER', 'cam', 0, 255, nothing)
#cv2.createTrackbar('R-LOWER', 'cam', 0, 255, nothing)
#cv2.createTrackbar('G-LOWER', 'cam', 0, 255, nothing)
#cv2.createTrackbar('B-LOWER', 'cam', 0, 255, nothing)


# range de azul claro...
lower_color = np.array([145, 78, 66]) 
upper_color = np.array([255, 255, 128])


red = (0, 0, 255)
blue = (255, 0, 0)
white = (255, 255, 255)

vid = cv2.VideoCapture(2)

# axes
ret, bgr_frame = vid.read()
height = bgr_frame.shape[0]
width = bgr_frame.shape[1]

v_start = (width // 2, 0)
v_end = (width // 2, height)
h_start = (0, height // 2)
h_end = (width, height // 2)
z_start = (0, height)
z_end = (width, 0)

#text
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
thickness = 1
font_color = (0, 0, 0)

while(True):
    #r_upper = cv2.getTrackbarPos('R-UPPER','cam')
    #g_upper = cv2.getTrackbarPos('G-UPPER','cam')
    #b_upper = cv2.getTrackbarPos('B-UPPER','cam')
    #r_lower = cv2.getTrackbarPos('R-LOWER','cam')
    #g_lower = cv2.getTrackbarPos('G-LOWER','cam')
    #b_lower = cv2.getTrackbarPos('B-LOWER','cam')
    #_lower_color = (b_lower, g_lower, r_lower)
    #_upper_color = (b_upper, g_upper, r_upper)

    ret, bgr_frame = vid.read()

    
    
    bgr_fliped_frame = cv2.flip(bgr_frame, 1)

    hsv_frame = cv2.cvtColor(bgr_fliped_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(bgr_fliped_frame, lower_color, upper_color)

    mask_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    usable_mask_contours = [mask_contour for mask_contour \
                                in mask_contours \
                                if len(mask_contours) != 0 and \
                                   cv2.contourArea(mask_contour) > 500]

    contour_area = 0
    for mask_contour in usable_mask_contours:
        x, y, w, h = cv2.boundingRect(mask_contour)
        contour_area = cv2.contourArea(mask_contour)
        
        #cv2.rectangle(bgr_fliped_frame, (x, y), (x + w, y + h), white, 1)

        line_v_start = (x + (w // 2), y)
        line_v_end = (x + (w // 2), y + h)

        line_h_start = (x, y + (h // 2))
        line_h_end = (x + w, y + (h // 2))

        cx = 1 - ((x + (w // 2)) / (width // 2))
        cx = round(cx / 10, 3) * -1

        cy = 1 - ((y + (h // 2)) / (height // 2))
        cy = round(cy / 10, 3)

        cz = 1 - (contour_area / 10_000)
        cz = round(cz / 10, 3) * -1

        msg = json.dumps({"x" : cx, "y" : cy, "z" : cz})
        s.send_multipart([b'', msg.encode("utf8")])
        print(f"sent: {msg}")
        
        cv2.line(bgr_fliped_frame, line_v_start, line_v_end, red, 1)
        cv2.line(bgr_fliped_frame, line_h_start, line_h_end, red, 1)

        cv2.putText(bgr_fliped_frame, f'x: {cx}', (10, 20), font, fontScale, font_color, thickness, cv2.LINE_AA)
        cv2.putText(bgr_fliped_frame, f'y: {cy}', (10, 40), font, fontScale, font_color, thickness, cv2.LINE_AA)
        cv2.putText(bgr_fliped_frame, f'z: {cz}', (10, 60), font, fontScale, font_color, thickness, cv2.LINE_AA)
        break


    cv2.line(bgr_fliped_frame, v_start, v_end, white, 1)
    cv2.line(bgr_fliped_frame, h_start, h_end, white, 1)
    #cv2.line(bgr_fliped_frame, z_start, z_end, blue, 1)

  
    cv2.imshow('controle', bgr_fliped_frame)
    #cv2.imshow('mask', mask)
    #cv2.imshow('hsv_frame', hsv_frame)
   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


vid.release()
cv2.destroyAllWindows()