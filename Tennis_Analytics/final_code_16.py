import cv2
import numpy as np
import pandas as pd


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Print x and y coordinates on left mouse click
        print(f"X: {x}, Y: {y}")


image_path1 = "ss25%.png"
img1 = cv2.imread(image_path1)
if img1 is None:
    print("Error: Could not load image!")
    exit()
p1warped_x=0
p1warped_y=0
p2warped_x=0
p2warped_y=0
pwarped_x = 0
pwarped_y = 0
ballPosition=[]
p1f = []
p2f = []
eventtype = []


speed = []
frame_name1 = []
bounce_count = 0
Pmeasurement_in_metery = 0
Pmeasurement_in_meterx = 0
warpedpoint = []
source_points = np.float32([[745, 438], [1036, 438], [1865, 767], [10, 755]])

# Define destination points for a flat wrap (1920x1080 output)
destination_points = np.float32([[0, 0], [1080, 0], [1080, 1920], [0, 1920]])

# Calculate the transformation matrix
transformation_matrix = cv2.getPerspectiveTransform(source_points, destination_points)
frameid = 0
bounceframeskip = 30
bounceframeskipp2bb = 50

vid = "MUKIL_RALLY_1.mp4"
# vid = "output_video_MUKHIL_1.mp4"

player_foot = []
ball_contours = []
player_contours = []
cap = cv2.VideoCapture(vid)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("fps:", fps, "width:", width, "height:", height)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("mukil_output1.mp4", fourcc, fps, (width, height))
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter("ball_det_mukhil_1MIN_1_2_4_5_6.mp4", fourcc, fps, (width, height))

centroid = []

kernel = np.ones((3, 3), np.uint8)
backgroundObject = cv2.createBackgroundSubtractorKNN(detectShadows=False)
frame_height = height
bottom_limit = frame_height * 0.6
frame_skipped=1394
ball_contours_dict = {}

while True:

    # Read a new frame.
    ret, frame = cap.read()

    # Check if frame is not read correctly.
    if not ret :
        # Break the loop.

        break
    frameid+=1
   
    if frameid <= frame_skipped:
            continue


    # frameid += 1
    cv2.putText(frame, f'Frame: {frameid}', (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    print(frameid)


    frame_name1.append(frameid)
    max_length = len(frame_name1)
    bounceframeskip += 1
    bounceframeskipp2bb += 1


    fgmask = backgroundObject.apply(frame)

    ret, thresh1 = cv2.threshold(fgmask, 10, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    top_30_percent = int(0.30 * 1080)
    thresh1[:top_30_percent, :] = 0  # MAKE 35 PRCNT TOP AS ZERO FOR BOTH PLAYERS AND BALL
    bottom_percent = int(0.30 * 1080)
    thresh1[1080 - bottom_percent:, :] = 0

    # Set left and right sides of the frame to zero
    side_percent1 = int(0.30 * 1920)
    side_percent2 = int(0.70 * 1920)

    # thresh1[:, :side_percent] = 0
    # thresh1[:, 1920 - side_percent:] = 0

    y_pixel_distance1 = 316
    x_pixel_distance2 = 267
    origin_pixel_point = (135, 291)

    contours, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ball_area_threshold = 1000  # Adjust according to the size of the ball in pixels
    # ball_circularity_threshold = 0.8
    ball_area_threshold = 100  # Adjust according to the size of the ball in pixels
    ball_circularity_threshold = 0.5  # Adjust according to the circularity of the ball

    player_area_threshold = 400  # Adjust according to the size of the players in pixels
    largest_contours = {1: None, 2: None}
    largest_areas = {1: 0, 2: 0}
    prev_centr = []
    no_of_ball_det = 0
    ball_check=[]
    for contour in contours:
        def euclidean_distance(x1, y1, x2, y2):
            return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


        area = cv2.contourArea(contour)
        _, radius = cv2.minEnclosingCircle(contour)
        circularity = area / (np.pi * (radius ** 2))
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h



        if area < ball_area_threshold and area > 5:
            if circularity > 0.4 and circularity < 1:

                x, y, w, h = cv2.boundingRect(contour)
                p = (x + w)
                q = (y + h)
                cx = (x + p) // 2
                cy = (y + q) // 2
                prev_centr.append((cx, cy))
                no_of_ball_det = no_of_ball_det + 1

                if cy < 655 and cy > 380 and cx > 576 and cx < 1344:

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                    ball_check.append((cx,cy))
                    ball_contours.append((cx, cy))

    

        elif area > 400 and aspect_ratio < 2:

            x, y, w, h = cv2.boundingRect(contour)
            # print("aspect_ratio",aspect_ratio)
            p = (x + w)
            q = (y + h)
            cx = (x + p) // 2
            cy = (y + q) // 2
            px = (w // 2)
            px1 = x + px
            py = y + h
            

            original_point = np.float32(
                [(px1), (py), 1])  # Add a 1 for homogeneous representation

            # Apply transformation to get warped coordinates in homogeneous form
            warped_point_homog = np.dot(transformation_matrix, original_point)

            warped_coordinates = warped_point_homog[:2] / warped_point_homog[2]
            pwarped_x = (warped_coordinates[0]) / 4
            pwarped_y = (warped_coordinates[1]) / 4

            # cv2.circle(img1, (int(warped_x), int(warped_y)), 4, (0, 255, 255), -1)

            y_pixel_distance1 = 316
            x_pixel_distance2 = 267
            origin_pixel_point = (135, 291)
            pixel_point_x, pixel_point_y = origin_pixel_point

            # Normalize to remove scaling and get final warped coordinates

            new_pixel_point_x = pwarped_x - pixel_point_x
            new_pixel_point_y = pwarped_y - pixel_point_y
            Pmeasurement_in_metery = (new_pixel_point_y / y_pixel_distance1) * 23.77
            Pmeasurement_in_meterx = (new_pixel_point_x / x_pixel_distance2) * 10.97
            # print("measuremnt_x",measurement_in_meterx)
            # print("measuremnt_y",measurement_in_metery)

            if py > 510 and area > 500:
                if area > largest_areas[1]:
                    largest_areas[1] = area
                    largest_contours[1] = contour
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
                    # cv2.circle(frame,(px1,py), 5, (255, 0, 255), -1)
                    
                    p1warped_x=pwarped_x
                    p1warped_y=pwarped_y
                    p1f.append([round(Pmeasurement_in_meterx, 3), round(Pmeasurement_in_metery, 3), 0])

                else:
                   
                    p1f.append([0, 0, 0])



            else:
                if area > largest_areas[2]:
                    largest_areas[2] = area
                    largest_contours[2] = contour
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
                    #  cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
                    # cv2.circle(frame,(px1,py), 5, (255, 0, 255), -1)

                    p2warped_x=pwarped_x
                    p2warped_y=pwarped_y
                    p2f.append([round(Pmeasurement_in_meterx, 3), round(Pmeasurement_in_metery, 3), 0])

                else:
                    p2f.append([0, 0, 0])

    ball_contours_dict[frameid] = ball_check
    # print("values",ball_contours_dict)



    if len(ball_contours) >= 3:  #p1bb

        last_three_centroids = ball_contours[-3:]

        # Extract x and y coordinates of the centroids
        x_coords = [point[0] for point in last_three_centroids]
        # print("x_cord",x_coords)
        y_coords = [point[1] for point in last_three_centroids]
        # print("y_cord",y_coords)


        if (430 < y_coords[0] < 500 and 430 < y_coords[1] < 500 and 430 < y_coords[2] < 500):  # fix 500

                # Check if the ball is hitting or not
                if y_coords[2] < y_coords[1] and y_coords[1] > y_coords[0]:
                    d = x_coords[1]
                    f = y_coords[1]

                    #  print("ddd",d)
                    #  print("fff",f)

                    # y_distance = abs(y_coords[2] - y_coords[0])

                    original_point = np.float32(
                        [(d), (f), 1])  # Add a 1 for homogeneous representation

                    # Apply transformation to get warped coordinates in homogeneous form
                    warped_point_homog = np.dot(transformation_matrix, original_point)

                    # Normalize to remove scaling and get final warped coordinates
                    y_pixel_distance1 = 316
                    x_pixel_distance2 = 270
                    origin_pixel_point = (135, 291)
                    pixel_point_x, pixel_point_y = origin_pixel_point

                    # Normalize to remove scaling and get final warped coordinates
                    warped_coordinates = warped_point_homog[:2] / warped_point_homog[2]
                    warped_x = (warped_coordinates[0]) / 4
                    warped_y = (warped_coordinates[1]) / 4
                    new_pixel_point_x = warped_x - pixel_point_x
                    new_pixel_point_y = warped_y - pixel_point_y
                    measurement_in_metery = (new_pixel_point_y / y_pixel_distance1) * 23.77
                    measurement_in_meterx = (new_pixel_point_x / x_pixel_distance2) * 10.97
                    time1 = 0.9

                    xcap=(warped_x-p1warped_x)*(291-warped_y)/(warped_y-p1warped_y) +warped_x
                    print("cap",xcap-pixel_point_x)
                    speed_distance = ((measurement_in_meterx - Pmeasurement_in_meterx) ** 2 + (
                                    Pmeasurement_in_metery - measurement_in_meterx) ** 2) ** (0.5)
                    speed_pixel = (((warped_x - p1warped_x) ** 2 + (warped_y - p1warped_y) ** 2) ** 0.5) / 28
                    # print("nc",warped_x,warped_y,pwarped_x,pwarped_y)
                    print("speed pixel",speed_pixel)
                    countnc = (int((((warped_x-xcap) ** 2 + (new_pixel_point_y) ** 2) ** (0.5)) / speed_pixel))
                    print("countnc",countnc)


                    # print("ball counter",ball_contours)
                    if bounceframeskipp2bb>=50:
                        text = "player1_bouncing_ball_frame"

                        ballPosition.extend([0] * (max_length - len(ballPosition)))
                        ballPosition[max_length-2]=([round(measurement_in_meterx, 3), round(measurement_in_metery, 3), 0])
                        cv2.circle(img1, (int(warped_x), int(warped_y)), 4, (0, 255, 255), -1)

                        eventtype.extend([0] * (max_length - len(eventtype)))
                        eventtype[max_length-2]=("p1bb")

                        # print("hi",-countnc)

                        eventtype[-countnc-1] = "p1nc"
                        nccoords = 0

                        ncxcoords = 870
                        for i in range(3):
                            if len(ball_contours_dict[frameid - countnc - 1 - i]) > 0:
                                nccoords = ball_contours_dict[frameid - countnc - 1 - i][0][1]
                                ncxcoords = ball_contours_dict[frameid - countnc - 1 - i][0][0]
                        ncy = nccoords
                        ncx = ncxcoords
                        heightball_pixel = abs(498 - ncy)
                        heightball = (heightball_pixel / 51) * 0.97

                        original_point = np.float32(
                            [(ncx), (498), 1])  # Add a 1 for homogeneous representation

                        # Apply transformation to get warped coordinates in homogeneous form
                        warped_point_homog = np.dot(transformation_matrix, original_point)

                        # Normalize to remove scaling and get final warped coordinates
                        y_pixel_distance1 = 316
                        x_pixel_distance2 = 270
                        origin_pixel_point = (135, 291)
                        pixel_point_x, pixel_point_y = origin_pixel_point

                        # Normalize to remove scaling and get final warped coordinates
                        warped_coordinates = warped_point_homog[:2] / warped_point_homog[2]
                        warped_x = (warped_coordinates[0]) / 4
                        warped_y = (warped_coordinates[1]) / 4
                        new_pixel_point_x = warped_x - pixel_point_x
                        new_pixel_point_y = warped_y - pixel_point_y

                        measurement_in_metery = (new_pixel_point_y / y_pixel_distance1) * 23.77
                        measurement_in_meterx = (new_pixel_point_x / x_pixel_distance2) * 10.97





                        ballPosition.extend([0] * (max_length - len(ballPosition)))

                        cv2.circle(img1, (int(warped_x), 291), 5, (255, 255, 255), -1)
                        ballPosition[max_length-countnc-1]=((round((measurement_in_meterx),3)),0,round(heightball,3))
                        # text = "person2 racket_hit"
                        # cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        print("height",heightball)

                        for i in range(17):
                            eventtype.append(0)
                        eventtype.append("p2cp")
                         # text = "person2 racket_hit"
                        # cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        # text="racket hit for person1"
                        # cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        speed.append(speed_distance / time1)

                    # p2bb.append((round(measurement_in_meterx,3),round(measurement_in_metery,3),0))

                        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        bounceframeskipp2bb = 0





    if len(ball_contours_dict) >= 3:    #p2bb
         
        last_three_keys = list(ball_contours_dict.keys())[-3:]

# Get the values corresponding to the last three keys
        last_three_centroids_values = [ball_contours_dict[key] for key in last_three_keys]
        

        last_three_centroids = [max(sublist, key=lambda x: x[1]) for sublist in last_three_centroids_values if len(sublist)>0]
        if len(last_three_centroids) == 3:
            print(last_three_centroids)

        # Extract x and y coordinates of the centroids
            x_coords = [point[0] for point in last_three_centroids]
            # print("x_cord",x_coords)
            y_coords = [point[1] for point in last_three_centroids]
            # print("y_cord",y_coords)
            print("passed1")
            print("skip",bounceframeskip)
            if bounceframeskip >= 20:
                if (755 > y_coords[0] > 500 and 755 > y_coords[1] > 500 and 755 > y_coords[2] > 500):  # fix 500


                    # Check if the ball is hitting or not
                    if y_coords[2] < y_coords[1] and y_coords[1] > y_coords[0]:
                        d = x_coords[1]
                        f = y_coords[1]
                        print("passed2")


                        original_point = np.float32(
                            [(d), (f), 1])  # Add a 1 for homogeneous representation

                        # Apply transformation to get warped coordinates in homogeneous form
                        warped_point_homog = np.dot(transformation_matrix, original_point)

                        # Normalize to remove scaling and get final warped coordinates
                        y_pixel_distance1 = 316
                        x_pixel_distance2 = 270
                        origin_pixel_point = (135, 291)
                        pixel_point_x, pixel_point_y = origin_pixel_point

                        # Normalize to remove scaling and get final warped coordinates
                        warped_coordinates = warped_point_homog[:2] / warped_point_homog[2]
                        warped_x = (warped_coordinates[0]) / 4
                        warped_y = (warped_coordinates[1]) / 4
                        new_pixel_point_x = warped_x - pixel_point_x
                        new_pixel_point_y = warped_y - pixel_point_y
                        measurement_in_metery = (new_pixel_point_y / y_pixel_distance1) * 23.77
                        measurement_in_meterx = (new_pixel_point_x / x_pixel_distance2) * 10.97
                        time1 = 0.9
                        xcap=(warped_x-p2warped_x)*(291-warped_y)/(warped_y-p2warped_y) +warped_x



                        speed_distance = ((measurement_in_meterx - Pmeasurement_in_meterx) ** 2 + (
                                    Pmeasurement_in_metery - measurement_in_meterx) ** 2) ** (0.5)
                        speed_pixel = (((warped_x - p2warped_x) ** 2 + (warped_y - p2warped_y) ** 2) ** 0.5) / 28
                        # print("nc",warped_x,warped_y,pwarped_x,pwarped_y)
                        print("speed pixel",speed_pixel)
                        countnc = (int((((warped_x-xcap) ** 2 + (new_pixel_point_y) ** 2) ** (0.5)) / speed_pixel))
                        print("countnc",countnc)

                        text = "player2_bouncing_ball_frame"
                        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        cv2.circle(img1, (int(warped_x), int(warped_y)), 4, (0, 255, 255), -1)

                        ballPosition.extend([0] * (max_length - len(ballPosition)))
                        ballPosition[max_length-2]=((round(measurement_in_meterx, 3), round(measurement_in_metery, 3), 0))
                            # print((round(measurement_in_meterx, 3), round(measurement_in_metery, 3), 0))

                        eventtype.extend([0] * (max_length - len(eventtype)))
                        eventtype[max_length-2]=("p2bb")
                        eventtype[-countnc - 1] = "p2nc"





                        nccoords=0
                        ncxcoords=500
                        for i in range (3):
                            if len(ball_contours_dict[frameid-countnc-1-i])>0:
                                nccoords=ball_contours_dict[frameid-countnc-1-i][0][1]
                                ncxcoords = ball_contours_dict[frameid - countnc - 1 - i][0][0]
                        ncy=nccoords
                        ncx=ncxcoords
                        heightball_pixel=abs(498-ncy)
                        heightball=(heightball_pixel/51)*0.97





                        original_point = np.float32(
                                [(ncx), (498), 1])  # Add a 1 for homogeneous representation

                            # Apply transformation to get warped coordinates in homogeneous form
                        warped_point_homog = np.dot(transformation_matrix, original_point)

                            # Normalize to remove scaling and get final warped coordinates
                        y_pixel_distance1 = 316
                        x_pixel_distance2 = 270
                        origin_pixel_point = (135, 291)
                        pixel_point_x, pixel_point_y = origin_pixel_point

                            # Normalize to remove scaling and get final warped coordinates
                        warped_coordinates = warped_point_homog[:2] / warped_point_homog[2]
                        warped_x = (warped_coordinates[0]) / 4
                        warped_y = (warped_coordinates[1]) / 4
                        new_pixel_point_x = warped_x - pixel_point_x
                        new_pixel_point_y = warped_y - pixel_point_y

                        measurement_in_metery = (new_pixel_point_y / y_pixel_distance1) * 23.77
                        measurement_in_meterx = (new_pixel_point_x / x_pixel_distance2) * 10.97






                        ballPosition.extend([0] * (max_length - len(ballPosition)))
                            # print("xcap",xcap)
                        cv2.circle(img1, (int(warped_x), 291), 3, (255, 255, 0), -1)
                        ballPosition[max_length-countnc-1]=((round((measurement_in_meterx),3)),0,round(heightball,3))
                        print("height",heightball)
                            

                            # text="racket hit for person2"
                            # cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        for i in range(14):
                            eventtype.append(0)
                        eventtype.append("p1cp")
                            # print("hi",-countnc)
                        speed.append(speed_distance / time1)


                        # p2bb.append((round(measurement_in_meterx,3),round(measurement_in_metery,3),0))

                        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        bounceframeskip = 0


    p1f.extend([0] * (max_length - len(p1f)))
    p2f.extend([0] * (max_length - len(p2f)))
    ballPosition.extend([0] * (max_length - len(ballPosition)))
    speed.extend([0] * (max_length - len(speed)))
    eventtype.extend([0] * (max_length - len(eventtype)))

    cv2.circle(img1, (100, 100), 2, (0, 0, 255), -1)
    # print("frame",frameid,"list",eventtype)
    # out.write(frame)

    cv2.namedWindow("Image")

    # Set the callback function for mouse events
    cv2.setMouseCallback("Image", click_event)


    # Display the image
    # cv2.imshow("Image", img1)
    # out.write(frame) 
    # cv2.imshow("Image with Coordinates", img1)
    # cv2.imshow("fraem", frame)



    cv2.waitKey(0)
cap.release()
out.release()       
# out.release()
cv2.destroyAllWindows()

print(len(frame_name1))
print(len(p1f))
print(len(p2f))


# data = {"frameid": frame_name1[1000:len(frame_name1)],"eventType":eventtype[1000:len(frame_name1)],
#         "p1pos": p1f[1000:len(frame_name1)], "p2pos": p2f[1000:len(frame_name1)],
#         "p1bb":p1bb[1000:len(frame_name1)], "p2bb": p2bb[1000:len(frame_name1)],
#         "p1cp": p1rh[1000:len(frame_name1)],
#         "p12nc": han[1000:len(frame_name1)]
#         }

data = {"frameID": frame_name1[0:len(frame_name1)], "eventType": eventtype[0:len(frame_name1)],
        "player1Position": p1f[0:len(frame_name1)], "player2Position": p2f[0:len(frame_name1)],
        "ballPosition": ballPosition[0:len(frame_name1)],"speed":speed[0:len(frame_name1)]
        }

#

# Create a DataFrame from the dictionary
finalmetrics = pd.DataFrame(data)
finalmetrics.to_csv("check2.csv", index=False)

# j=finalmetrics.to_json(orient='records', indent=4)
# with open("finaljson.json","w") as f:
#     f.write(j)

