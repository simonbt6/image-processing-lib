from typing import List
import cv2
import numpy as np
import time

class ImageProcessing:
    
    # CONFIG
    MINLINELENGTH = 300
    MAXLINEGAP = 5
    SCALEPERCENT = 50

    # ENVIRONNEMENT SETTINGS
    DEBUG = False

    def legend(image, width, height):
        cv2.putText(image, "Cars", (width, height), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255))
        return image

    def process_image(image):
        start_time = time.time()
        processed_image = image
        # Find edges
        edges = ImageProcessing.canny_image(processed_image)
        # Blur edges
        edges = ImageProcessing.blur_image(edges)
        
        processed_image = ImageProcessing.good_features(processed_image)

        # Find lines with edges than apply on image.
        #processed_image = ImageProcessing.image_lines(processed_image, edges)
        end_time = time.time()

        if ImageProcessing.DEBUG:
            print("FPS: " + str(1000/((end_time - start_time)*1000)))

        return processed_image
    
    def good_features(image, _from):
        gray  = cv2.cvtColor(_from, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, 200, 0.01, 10)
        corners = np.int0(corners)

        for i in corners:
            x, y = i.ravel()
            cv2.circle(image, (x, y), 3, 255, -1)
        return image

    def filter_image(image, width, height, partition):
        return cv2.fillPoly(image, [np.array([[0,0 + (height/2*partition)], [width, 0 + (height/2*partition)], [width, height/2 + (height/2*partition)], [0, height/2 + (height/2*partition)]], dtype='int32')], (0, 0, 0))

    def canny_image(image):
        processed_image = image
        processed_image = cv2.Canny(processed_image, 100, 200)
        return processed_image

    def blur_image(image):
        return cv2.blur(image, (5, 5))

    def estimate_distance(object) -> float:
        return (0.0)

    def image_lines(image, edges):
        lines = cv2.HoughLinesP(edges, 2, np.pi/180, 50, minLineLength=ImageProcessing.minLineLength, maxLineGap=ImageProcessing.maxLineGap)
        
        if lines is None:
            return image

        lines = ImageProcessing.sortLines(lines) 
        for (x1, y1, x2, y2) in lines: 
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 5)

        return image

    def slope(x1, y1, x2, y2):
        return ((y2-y1)/(x2-x1))

    def sortLines(lines):
        print("Before %s" % len(lines))
        partition = lines[:int(len(lines)/2)]
        print("First partitionning %s" % len(partition))
        slope_dict = {}

        for line in partition:
            (x1, y1, x2, y2) = line[0]
            slope_dict[(x1, y1, x2, y2)] = ImageProcessing.slope(x1, y1, x2, y2)

        sorted_dict = {k: v for k, v in sorted(slope_dict.items(), key=lambda item: item[1])}
        left_partition = [k for k, v in sorted_dict.items()  if v < 1]
        right_partition = [k for k, v in sorted_dict.items() if v > 0]
        
        right_partition = ImageProcessing.unpack_and_sort(right_partition, 0)
        left_partition = ImageProcessing.unpack_and_sort(left_partition, 0)

        partition = right_partition + left_partition

        print("After %s" % len(partition))

        return partition
    
    def unpack_and_sort(lines : List, side) -> List:
        lines.sort(key=lambda tup: tup[1])
        if side:
            return lines[:len(lines)//2]
        else:
            return lines[len(lines)//2::]

    def slopeC(line1, line2):
        (ax1, ay1, ax2, ay2) = line1
        (bx1, by1, bx2, by2) = line2

        a1 = ((ay2-ay1)/(ax2-ax1)) 
        a2 = ((by2-by1)/(bx2-bx1))

        a = (a2-a1)

        if a > 0:
            return line2
        return None

    def chooseLines(image, lines):
        global cache
        global first_frame

        slope_l, slope_r = [], []
        lane_l, lane_r = [], []
        shape0 = image.shape[0]
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = (y2-y1)/(x2-x1)

                if slope > 0.4:
                    slope_r.append(slope)
                    lane_r.append(line)
                elif slope < -0.4:
                    slope_l.append(slope)
                    lane_l.append(line)
            shape0 = min(y1, y2, image.shape[0])
        if ((len(lane_l) == 0) or (len(lane_r) == 0)):
            print("No lane detected.")
            return image, None
        slope_mean_l = np.mean(slope_l, axis = 0)
        slope_mean_r = np.mean(slope_r, axis = 0)
        mean_l = np.mean(np.array(lane_l), axis = 0)
        mean_r = np.mean(np.array(lane_r), axis = 0)

        if ((slope_mean_r == 0) or (slope_mean_l == 0)):
            print("Dividing by zero")
            return image, None
        x1_l = int((shape0 - mean_l[0][1] - (slope_mean_l * mean_l[0][0]))/slope_mean_l) 
        x2_l = int((shape0 - mean_l[0][1] - (slope_mean_l * mean_l[0][0]))/slope_mean_l)   
        x1_r = int((shape0 - mean_r[0][1] - (slope_mean_r * mean_r[0][0]))/slope_mean_r)
        x2_r = int((shape0 - mean_r[0][1] - (slope_mean_r * mean_r[0][0]))/slope_mean_r)
    
        if x1_l > x1_r:
            x1_l = int((x1_l + x1_r)/2)
            x1_r = x1_l
            y1_l = int((slope_mean_l * x1_l) + mean_l[0][1] - (slope_mean_l * mean_l[0][0]))
            y1_r = int((slope_mean_r * x1_r ) + mean_r[0][1] - (slope_mean_r * mean_r[0][0]))
            y2_l = int((slope_mean_l * x2_l ) + mean_l[0][1] - (slope_mean_l * mean_l[0][0]))
            y2_r = int((slope_mean_r * x2_r ) + mean_r[0][1] - (slope_mean_r * mean_r[0][0]))
        else:
            y1_l = shape0
            y2_l = shape0
            y1_r = shape0
            y2_r = shape0
        print([(x1_l, y1_l, x2_l, y2_l), (x1_r, y1_r, x2_r, y2_r)])
        return image, [(x1_l, y1_l, x2_l, y2_l), (x1_r, y1_r, x2_r, y2_r)]

    def image_circles(image, edges):
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=50, minRadius=0, maxRadius=150)
        
        try:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                try:
                    cv2.circle(image, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
                    cv2.circle(image, (circle[0], circle[1]), circle[2], (0, 255, 0), 3)
                except:
                    pass
        except:
            pass
        return image
