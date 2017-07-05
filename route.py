import cv2 

import numpy  as np
import math 
import bluetooth
from time import sleep
import sys

west = 'west'
east = 'east'
south = 'south'
north= 'north'


thres_line = 1.0

Yellow_lower = np.array([22,125,117])   #(0,232,75), (90,255,175))### 
Yellow_upper = np.array([30,255,255])


Green_lower = np.array([31,76,81])
Green_upper = np.array([90,255,255])   #(38,135,85), (90,255,161))

Blue_lower = np.array([80,65,85])  #(71,63,103), (127,215,205))
Blue_upper = np.array([124,205,225])

Orange_lower = np.array([0,131,100])  #(71,63,103), (127,215,205))
Orange_upper = np.array([20,255,255])

def HSVadjust():
  def nothing(x):
   pass;
  
  def color_code(color):
    cv2.namedWindow(color)
    if color == 'orange':
      iSliderValue1 = int(Orange_upper[0])
      iSliderValue2 = int(Orange_upper[1])
      iSliderValue3 =int(Orange_upper[2]);
      iSliderValue4 = int(Orange_lower[0]);
      iSliderValue5 = int(Orange_lower[1]);
      iSliderValue6 =int(Orange_lower[2]);
    elif color == 'green':
      iSliderValue1 = int(Green_upper[0])
      iSliderValue2 = int(Green_upper[1])
      iSliderValue3 =int(Green_upper[2]);
      iSliderValue4 = int(Green_lower[0]);
      iSliderValue5 = int(Green_lower[1]);
      iSliderValue6 =int(Green_lower[2]);
    elif color == 'yellow':
      iSliderValue1 = int(Yellow_upper[0])
      iSliderValue2 = int(Yellow_upper[1])
      iSliderValue3 =int(Yellow_upper[2]);
      iSliderValue4 = int(Yellow_lower[0]);
      iSliderValue5 = int(Yellow_lower[1]);
      iSliderValue6 =int(Yellow_lower[2]);
    elif color == 'blue':
      iSliderValue1 = int(Blue_upper[0])
      iSliderValue2 = int(Blue_upper[1])
      iSliderValue3 =int(Blue_upper[2]);
      iSliderValue4 = int(Blue_lower[0]);
      iSliderValue5 = int(Blue_lower[1]);
      iSliderValue6 =int(Blue_lower[2]);
    cv2.createTrackbar("H_MAX", color, iSliderValue1, 255, nothing)
    cv2.createTrackbar("S_MAX", color, iSliderValue2, 255, nothing)
    cv2.createTrackbar("V_MAX", color, iSliderValue3, 255, nothing)
    cv2.createTrackbar("H_MIN", color, iSliderValue4, 255, nothing);
    cv2.createTrackbar("S_MIN", color, iSliderValue5, 255, nothing)
    cv2.createTrackbar("V_MIN", color, iSliderValue6, 255, nothing)

  color= 'orange'
  color_code(color)
  i = int(0)
  cam =cv2.VideoCapture(1)

  while 1:
      key=cv2.waitKey(22)
      
      if key == ord('s'):
        cv2.destroyAllWindows()
        i = i+1
        if i==1:
          color='green'
          color_code(color)
          
        elif i==2:
          color = 'blue'
          color_code(color)
        elif i==3:
          color = 'yellow'
          color_code(color)
        elif i ==4:
          color == 'orange'
          color_code(color)
        elif i ==5  :
          break
      
      elif key == 27:
          break
      #img= cv2.imread('capture.jpg')
      ret, img = cam.read()
      hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

          
      iSliderValue1 = cv2.getTrackbarPos("H_MAX", color)
      iSliderValue2 = cv2.getTrackbarPos("S_MAX", color)
      iSliderValue3= cv2.getTrackbarPos("V_MAX", color)
      iSliderValue4= cv2.getTrackbarPos("H_MIN", color)
      iSliderValue5= cv2.getTrackbarPos("S_MIN", color)
      iSliderValue6= cv2.getTrackbarPos("V_MIN", color)

      
      slider = cv2.inRange(hsv_image, (iSliderValue4, iSliderValue5, iSliderValue6), (iSliderValue1, iSliderValue2, iSliderValue3));
     
      
      cv2.imshow(color+"test", slider);
      cv2.waitKey(10)

  cv2.destroyAllWindows()

  return;



def detect_features():
  cam = cv2.VideoCapture(1)
  def get_img():             
   cv2.waitKey(10)   
   ret, image = cam.read()
   return image;
    
  captured_img = get_img()
  cv2.waitKey(35)                     #to avoid initial frames from camera
  captured_img = get_img()           # which are not accurate in color
  cv2.waitKey(35)

  while 1:
    img = get_img()
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    thres_range = cv2.inRange(hsv_image, Orange_lower, Orange_upper)### YELLOW ###
    kernel = np.ones((4,4),np.uint8)
    opening = cv2.morphologyEx(thres_range, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    temp,contours, hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for i in range(0, len(contours)):
        cnt = contours[i]
        M = cv2.moments(cnt)
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    if len(contours) !=4:
        print "Corners not detected properly only", len(contours)
        
        
    else:    
        cen = []
        for i in range(4):
            cen.append([])
            cen[i].append(0)
            cen[i].append(0)
            cnt = contours[i]
            M = cv2.moments(cnt)
            cen[i][0] = int(M['m10']/M['m00'])
            cen[i][1] = int(M['m01']/M['m00'])   ## corner pts

        height, width = img.shape[:2]
          

        for i in range(4):
            if cen[i][0] < (width/2):
                if cen[i][1]<(height/2):
                    top_left = cen[i]
                else:
                    bottom_left = cen[i]
            else:
                 if cen[i][1]<(height/2):
                    top_right = cen[i]
                 else:
                    bottom_right = cen[i]
        px = img[250,250]   ######if BLACk fiilling
        #print px
        color = np.array([0,0,0])
        color[0] = px[0]
        color[1] = px[1]
        color[2] = px[2]
        
        #print top_left, top_right, bottom_left, bottom_right                            
        top = np.array([[0,0],[width,0],top_right,top_left])
        left= np.array([[0,0],[0,height] ,bottom_left ,top_left])
        bottom= np.array([[0,height] ,bottom_left ,bottom_right, [width, height]])
        right= np.array([[width,0],top_right, bottom_right ,[width, height]])
        
        cv2.fillConvexPoly(img, top, color)
        cv2.fillConvexPoly(img, left, color)
        cv2.fillConvexPoly(img, bottom , color)
        cv2.fillConvexPoly(img, right, color)
        corner_img = img.copy()
        
        cv2.putText(img,"Corner-1", (bottom_left[0]-45, bottom_left[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,58, 255),2)
        cv2.putText(img,"Corner-2", (top_left[0]-42, top_left[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,58, 255),2)
        cv2.putText(img,"Corner-3", (bottom_right[0]-35, bottom_right[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,58, 255),2)
        cv2.putText(img,"Corner-4", (top_right[0]-35, top_right[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,58, 255),2)

    thres_range = cv2.inRange(hsv_image, Green_lower, Green_upper)
    opening = cv2.morphologyEx(thres_range, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    temp,contours, hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.drawContours(img,contours,-1,(0,0,255),3)
    if len(contours)>0:
        cnt = contours[0]
        M = cv2.moments(cnt)
        cx_start = int(M['m10']/M['m00']) 
        cy_start = int(M['m01']/M['m00'])
        cv2.putText(img,"Start",(cx_start, cy_start) , cv2.FONT_HERSHEY_SIMPLEX, 0.5,(250,58, 255),2)
    
    
        
    else:
        cx_start = -1 
        cy_start = -1
        print "no GREEN coontour detected"
    
    ### for BLUE = STOP  ###
    thres_range = cv2.inRange(hsv_image, Blue_lower, Blue_upper)
    opening = cv2.morphologyEx(thres_range, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    temp,contours, hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    
    if len(contours)>0:
        cnt = contours[0]
        M = cv2.moments(cnt)
        cx_stop = int(M['m10']/M['m00']) 
        cy_stop = int(M['m01']/M['m00'])

        cv2.drawContours(img,contours,-1,(0,0,255),3)
        cv2.putText(img,"Stop", (cx_stop, cy_stop), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(212,58, 55),2)
    
    else:
        cx_stop = -1
        cy_stop = -1

        print "no BLUE coontour detected"
    

    
    cv2.imshow('Vehicle Guidance ', img)
    key =cv2.waitKey()
    if key == 13:
      cv2.imwrite('./data/capture.jpg',corner_img)
      cv2.destroyAllWindows()
      break
    else:
      pass
    
  
  return[top_left, top_right, bottom_left, bottom_right,\
           cx_start, cy_start,cx_stop, cy_stop];





def recheck(imge, curr_pos_x, curr_pos_y,row,col,ls,color,north, east, west, south):
        separation =25
        color = np.array([3,2,7])
        if west ==1:   #  check in north
                 
         cv2.rectangle(imge,(0, 0), (curr_pos_x - separation-14, row), color,-1)        #left
         cv2.rectangle(imge, (0, 0), (col, curr_pos_y - 20),color, -1)                #top
         cv2.rectangle(imge, (curr_pos_x, 0), (col, row),color, -1)              #right
         cv2.rectangle(imge, (0,curr_pos_y), (col, row), color, -1)#bottom

        elif east == 1:  #south
            
            cv2.rectangle(imge, (0, 0), (curr_pos_x , row), color, -1, 8);             #left
            cv2.rectangle(imge, (0, 0), (col, curr_pos_y), color, -1, 8);           #top
            cv2.rectangle(imge, (curr_pos_x + separation+14, 0), (col, row), color, -1, 8)    #right
            cv2.rectangle(imge, (0, curr_pos_y+20), (col, row), color, -1, 8);         #bottom
                     
        elif north == 1:   #check in east
            
            cv2.rectangle(imge, (0, 0), (curr_pos_x , row), color, -1, 8);             #left
            cv2.rectangle(imge, (0, 0), (col, curr_pos_y - separation-14),color, -1, 8);    #top
            cv2.rectangle(imge, (curr_pos_x + 20, 0), (col, row), color, -1, 8);        #/right
            cv2.rectangle(imge, (0, curr_pos_y), (col, row), color, -1, 8);        # bottom

                                            
        elif south==1:#west
         
         cv2.rectangle(imge, (0, 0), (curr_pos_x-20, row), color, -1, 8);            #left
         cv2.rectangle(imge, (0, 0), (col, curr_pos_y), color, -1, 8);           #top
         cv2.rectangle(imge, (curr_pos_x , 0), (col, row), color, -1, 8);          #right
         cv2.rectangle(imge, (0, curr_pos_y + separation+12), (col, row), color, -1, 8);     #bottom

        lines = ls.detect(imge)
        '''
        #print "auxiliaty", len(lines[0])
        img3 = ls.drawSegments(imge, lines[0])
        cv2.imshow("detd lines", img3)
        cv2.waitKey(12)'''

    
        return len(lines[0]);


    

def findpath(start_x, start_y, stop_x, stop_y,\
             top_left, top_right, bottom_left, bottom_right ):

    img =  cv2.imread('./data/capture.jpg')
    img = cv2.medianBlur(img,5)     ## blurrrrrrrrrrrr
    kernel = np.ones((2,2),np.uint8)
    imgcircle = img.copy()
    imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ##maynot be necessary
    bottom_start_pt = bool(0)
    left_start_pt = bool(0)
    right_start_pt = bool(0)
    top_start_pt = bool(0)
    '''
    imggray= cv2.adaptiveThreshold(imggray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                              cv2.THRESH_BINARY,7,18)
    #cv2.imshow("teasd", imggray)
    #cv2.waitKey()
    #imggray = cv2.erode(imggray,kernel,iterations = 1)
    
    #cv2.imshow("teasd", imggray)
    #cv2.waitKey()
   ##above may not be necessary in daylight   11, 2
  '''   
    north = bool(0)
    south = bool(0)
    east = bool(0)
    west =bool(0)
    direc_change =bool(0)

    nodes = []
    node_count = int(0)

    row,col = imggray.shape[:2] 

    if ( (bottom_left[1]+bottom_right[1])/2-start_y) <30 and ((bottom_left[1]+bottom_right[1])/2-start_y)>0: #gap betwn start and bottom
        print "bottom start"
        bottom_start_pt = True
    elif ( (bottom_left[0]+top_left[0])/2-start_x) >-30 and ((bottom_left[0]+top_left[0])/2-start_x) <0:  #gap betwn start and bottom
        print "LEFT START"
        left_start_pt = True
    elif ( (top_left[1]+top_right[1])/2-start_y) >-30 and ((top_left[1]+top_right[1])/2-start_y) <0 :  #gap betwn start and bottom
        print "TOP START"
        top_start_pt = True
    elif ( (bottom_right[0]+top_right[0])/2-start_x) <30 and ((bottom_right[0]+top_right[0])/2-start_x)>0:  #gap betwn start and bottom
        print "right START"
        right_start_pt = True
    else:
      print 'CANNOT DECIDE THE STARTING POINT'
      
      
    if bottom_start_pt == True:
        north= True
    elif left_start_pt == True:
        east= True
    elif top_start_pt == True:
        south= True
    elif right_start_pt == True:
        west= True

    curr_pos_x = start_x
    curr_pos_y = start_y #move up little

    prev_x = curr_pos_x
    prev_y= curr_pos_y


    #initail node
    nodes.append([])
    nodes[node_count].append(curr_pos_x)
    nodes[node_count].append(curr_pos_y)
    node_count = node_count+1
        

    ls=cv2.createLineSegmentDetector(1,0.8,thres_line, 2.0, 22.5, 0, 0.7, 1024)
    lines = ls.detect(imggray)
    img12 = ls.drawSegments(imggray, lines[0])
        
    cv2.imshow("Vehicle Guidance", img12)
    cv2.waitKey()
    cv2.destroyAllWindows()
    color = np.array([255,255,255])
    
    cv2.circle(imgcircle, (stop_x, stop_y), 3, (224, 06, 225), 3, 8, 0);
    cv2.circle(imgcircle, (start_x, start_y), 3, (224, 06, 225), 3, 8, 0);

    while True:
        imge = imggray.copy()
        
        if north==1:
         curr_pos_y -= 10;            #move little forward
         if curr_pos_y - 20>0:
               roi = imge[(curr_pos_y - 20):curr_pos_y,(curr_pos_x-45):curr_pos_x]
         elif curr_pos_y - 20<0:
               roi = imge[(0):curr_pos_y,(curr_pos_x-45):curr_pos_x]

         height, width = roi.shape[:2] 
    

         #y1:y2, x1:x2
        elif south == 1:
            curr_pos_y += 10;
            
            roi = imge[(curr_pos_y ):curr_pos_y+20,(curr_pos_x):curr_pos_x+45]
            height, width = roi.shape[:2] 
             
        elif east == 1:
            curr_pos_x += 10;
            if curr_pos_y - 45>0:
                  roi = imge[(curr_pos_y-45):curr_pos_y,(curr_pos_x):curr_pos_x+20]
            else:
              roi = imge[0:curr_pos_y,(curr_pos_x):curr_pos_x+20]
                  
            height, width = roi.shape[:2] 
    
                                            
        elif west==1:
         print "here"
         curr_pos_x -= 10;
         
         roi = imge[(curr_pos_y):curr_pos_y+45,(curr_pos_x-20):curr_pos_x]
         height, width = roi.shape[:2] 
    
        else:
            print "no direction detected"
        
        lines = ls.detect(roi)

        if lines[0] != None:
            line_no= len(lines[0])            #detect lines in the moving direction
        else:
            line_no = 0
            print "Zero line in moving direction"
        
        roi = ls.drawSegments(roi, lines[0])
        
        #cv2.imshow("ROI", roi)
        #constant separation between lines
        
        for i in range(0,line_no):
            if abs(lines[0][i][0][0]-lines[0][i][0][2])<2 and abs(lines[0][i][0][1]-lines[0][i][0][3])>2:#verical
                if north == 1:
                    curr_pos_x= curr_pos_x -int(width - lines[0][i][0][2])+ 35   # adjust x
                    curr_pos_x = ( curr_pos_x+prev_x)/2
                elif south==1:
                    curr_pos_x= curr_pos_x +int( lines[0][i][0][2])- 35
                    curr_pos_x = ( curr_pos_x+prev_x)/2
                    
                    
                
            if abs(lines[0][i][0][1]-lines[0][i][0][3])<2 and lines[0][i][0][0]-lines[0][i][0][2]>2 : #horizontl
                if west == 1:
                    curr_pos_y= curr_pos_y +int(lines[0][i][0][1])- 35
                    curr_pos_y = ( curr_pos_y+prev_y)/2 

                elif east ==1:
                    curr_pos_y= curr_pos_y -int(height-lines[0][i][0][1])+ 35
                    curr_pos_y = ( curr_pos_y+prev_y)/2                    
                  
                
     
        cv2.imshow("Vehicle Guidance", imgcircle)

        imge = imggray.copy()
        right_lines = recheck(imge, curr_pos_x, curr_pos_y,row,col,ls,color,north, east, west, south)-4
        cv2.circle(imgcircle, (int(curr_pos_x), int(curr_pos_y)), 3, (204, 206, 005, 205), 2, 8, 0);

        if north == 1:    #NORTH
            print "NORTH"
            if line_no >1 and right_lines == 0:
                north=1
                south=0
                east=0
                west=0
            elif line_no>1 and right_lines>1:
                print "north-east"
                east=1
                south=0
                north=0
                west =0
                nodes.append([])
                nodes[node_count].append(curr_pos_x)
                nodes[node_count].append(curr_pos_y)
                node_count = node_count+1
            elif line_no < 1 and right_lines<1:
                print "NORTH-WEST"
                
                east=0
                south=0
                north=0
                west =1
                

                nodes.append([])
                nodes[node_count].append(curr_pos_x)
                nodes[node_count].append(curr_pos_y-29)
                node_count = node_count+1
                curr_pos_x = curr_pos_x-25
                curr_pos_y = curr_pos_y-25
        
            else:
                print "north cant decide direction"

                
        elif west == 1 :
            print "WEST"
            if line_no >1 and right_lines == 0:
                north=0
                south=0    
                east=0
                west=1
               
            elif line_no>1 and right_lines>1:
               
                east=0
                south=0
                north=1
                west =0
                nodes.append([])
                nodes[node_count].append(curr_pos_x)
                nodes[node_count].append(curr_pos_y)
                node_count = node_count+1
            elif line_no < 1 and right_lines<1:
               
                print "WEST--SOUTH"
                east=0
                south=1
                north=0
                west =0
            
                nodes.append([])
                nodes[node_count].append(curr_pos_x-20)
                nodes[node_count].append(curr_pos_y)
                node_count = node_count+1

                curr_pos_x = curr_pos_x-22
                curr_pos_y = curr_pos_y+32
            else:
                print "west cant decide direction"

        elif east == 1 :
            print "EAST"
            if line_no >1 and right_lines == 0:
                north=0
                south=0    
                east=1
                west=0
            elif line_no>1 and right_lines>1:
                print "EAST-SOUTH"
                east=0
                south=1
                north=0
                west =0
                nodes.append([])
                nodes[node_count].append(curr_pos_x)
                nodes[node_count].append(curr_pos_y)
                node_count = node_count+1
            elif line_no < 1 and right_lines<1:
               
                print "EAST-NORTH"
                east=0
                south=0
                north=1
                west =0
                nodes.append([])
                nodes[node_count].append(curr_pos_x+20)
                nodes[node_count].append(curr_pos_y)
                node_count = node_count+1
                curr_pos_x = curr_pos_x+25
                curr_pos_y = curr_pos_y-30
            else:
                print "east cant decide direction"
                
        elif south == 1:
            print "SOUTH"

            if line_no >1 and right_lines == 0:
                north=0
                south=1   
                east=0
                west=0
            elif line_no>1 and right_lines>1:
               
                east=0
                south=0
                north=0
                west =1
                nodes.append([])
                nodes[node_count].append(curr_pos_x)
                nodes[node_count].append(curr_pos_y)
                node_count = node_count+1
            elif line_no < 1 and right_lines<1:
               
                print "SOUTH-EAST"
                east=1
                south=0
                north=0
                west =0
                nodes.append([])
                nodes[node_count].append(curr_pos_x)
                nodes[node_count].append(curr_pos_y+25)
                node_count = node_count+1
                curr_pos_x = curr_pos_x+30
                curr_pos_y = curr_pos_y+25
                
            else:
                print "west cant decide direction"

        prev_x = curr_pos_x
        prev_y= curr_pos_y


        if abs( curr_pos_x-stop_x)<28 and abs(curr_pos_y -stop_y)<28:
              print "stop point detected"
              cv2.circle(imgcircle, (curr_pos_x, curr_pos_y), 3, (24, 06, 252), 4, 8, 0);
              
              break
        elif abs( curr_pos_x-stop_x)<15 and abs(curr_pos_y -stop_y)<47:
              print "stop point detected"

              cv2.circle(imgcircle, (curr_pos_x, curr_pos_y), 3, (24, 06, 252), 4, 8, 0);
              
              break
        elif abs( curr_pos_x-stop_x)<47 and abs(curr_pos_y -stop_y)<15:
              print "stop point detected"
              cv2.circle(imgcircle, (curr_pos_x, curr_pos_y), 3, (24, 06, 252), 4, 8, 0);
              
              break  
         
        key = cv2.waitKey(3)
        if key== 27:
          break

    
    nodes.append([])
    nodes[node_count].append(curr_pos_x)
    nodes[node_count].append(curr_pos_y)
    node_count = node_count+1
    nodes.append([])
    nodes[node_count].append(stop_x)
    nodes[node_count].append(stop_y)    


    for i in range(0, len(nodes)):
          
      cv2.circle(imgcircle, (nodes[i][0], nodes[i][1]), 5, (204, 06, 205), 3, 8, 0);     
      cv2.imshow("Vehicle Guidance", imgcircle)
    
    k = int(0)
    i = int(1)
    final_nodes = []
    
    final_nodes.append([])
    final_nodes[k].append(start_x)
    final_nodes[k].append(start_y)
    cv2.circle(imgcircle, (final_nodes[k][0], final_nodes[k][1]), 8, (204, 06, 05), 2, 8, 0)
    k = k+1
    while i <= len(nodes):
      for j in range(0, len(nodes)-i):
        j = len(nodes)-j-1

        if abs(nodes[i][0]-nodes[j][0]) ==0 and abs(nodes[i][1]-nodes[j][1]) == 0:
                        
                        final_nodes.append([])
                        final_nodes[k].append((nodes[i][0]+nodes[j][0])/2)
                        final_nodes[k].append((nodes[i][1]+nodes[j][1])/2)
                        cv2.circle(imgcircle, (final_nodes[k][0], final_nodes[k][1]), 8, (204, 06, 05), 2, 8, 0);     
                        cv2.imshow("Vehicle Guidance", imgcircle)
                        k = k+1
                        print i, '==', j
                        i=j
                        
                        cv2.waitKey(3)
                        found_flag =1
                        break
        elif abs(nodes[i][0]-nodes[j][0])<25 and abs(nodes[i][1]-nodes[j][1])<25  :
                        i=j
                        i = i+1
                        final_nodes.append([])
                        final_nodes[k].append(nodes[i][0])
                        final_nodes[k].append(nodes[i][1])
                        cv2.circle(imgcircle, (final_nodes[k][0], final_nodes[k][1]), 8, (204, 06, 05), 2, 8, 0);     
                        cv2.imshow("Vehicle Guidance", imgcircle)
                        k = k+1
                        print i, '=1=', j

                        cv2.waitKey(3)
                        found_flag =1
                        break
      i = i+1
  
    cv2.imwrite('./data/node_img.jpg', imgcircle) 
    print "Solved"
    return[final_nodes];


  

def find_direction(prev_direction, end , start):
      print prev_direction
      diff_x = end[0]-start[0]
      diff_y = end[1]-start[1]
      if abs(diff_x)>abs(diff_y):  #east/west
            if diff_x>0:         #easr
               direction ='east'

            else:
               direction ='west'

      else:          #north/south
           if diff_y>0:         #south
               direction ='south'

           else:      # north
               direction ='north'

      if (prev_direction == 'north' and direction == 'west') or (prev_direction == 'west' and direction == 'south') or (prev_direction == 'south' and direction == 'east') or (prev_direction == east and direction == north)or (prev_direction == west and direction == east)or (prev_direction == 'north' and direction == 'south')or (prev_direction == east and direction == west)or (prev_direction == south and direction == north):
            rotate = 'left'
      elif (direction == 'north' and prev_direction == 'west') or (direction == 'west' and prev_direction == 'south') or (direction == 'south' and prev_direction == 'east') or (direction == east and prev_direction == north):
            rotate = 'right'
      elif prev_direction == direction or prev_direction ==0:
            rotate = 0
      print 'direction = '+ direction+'  Turn = ', rotate  
      return [rotate,direction];





def control_bot():
      size = 600,800, 3
      m = np.zeros(size, dtype=np.uint8)
      cv2.putText(m,  "PLEASE PLACE THE BOT AT ", (65,180), cv2.FONT_HERSHEY_SIMPLEX, 1.5,(190,198, 155),2)
      cv2.putText(m,  "STARTING POINT", (210,240), cv2.FONT_HERSHEY_SIMPLEX, 1.5,(190,198, 155),2)
      cv2.imshow('Vehicle Guidance' ,m)
      cv2.waitKey()
      cv2.destroyAllWindows()
        
      file= open('./data/nodes.txt', 'r')
      lines = [line.rstrip('\n') for line in open('./data/nodes.txt')]
      nodes = []
      j = int(0)
      for i in range(0, len(lines)/2):
        nodes.append([])
        nodes[i].append(int(lines[j]))
        nodes[i].append(int(lines[j+1]))
        j = 2*(i+1)

      print nodes
      forward = 'a'
      backward = 'b'
      turn_left ='c'
      turn_right = 'd'
      stop = 'e'

      long_delay = 0.75
      short_delay = 0.10
      rotate_delay = 0.06
      adjust_delay = 0.047

      
      i = int(0)
      print len(nodes)
      check_flag = bool(0)
      rotate_flag = bool(0)
      rest = bool(1)  # the bot is at rest
      long_dist = 55
      prev_x = nodes[0][0]
      prev_y = nodes[0][1]

      print nodes[i+1],nodes[i]
      rotate,direction = find_direction(0,nodes[i+1],nodes[i])

      cam = cv2.VideoCapture(1)

      bd_addr = '20:14:10:30:03:58' # The MAC address of a Bluetooth adapter on the server.
      port = 1
      backlog = 1
      size = 1024
      s = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
      s.connect((bd_addr, port))
      print "conected to "+ bd_addr+ "to bot"
      
      while 1:
        
          ret,img =  cam.read()
          cv2.waitKey(5)
          ret,img =  cam.read()
          
          hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
          thres_range = cv2.inRange(hsv_image, Green_lower, Green_upper)### YELLOW ###
          kernel = np.ones((7,7),np.uint8)
          opening = cv2.morphologyEx(thres_range, cv2.MORPH_OPEN, kernel)
          closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
          temp,contours, hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

          thres_range1 = cv2.inRange(hsv_image, Yellow_lower, Yellow_upper)### YELLOW ###
          kernel1 = np.ones((7,7),np.uint8)
          opening1 = cv2.morphologyEx(thres_range1, cv2.MORPH_OPEN, kernel1)
          closing1 = cv2.morphologyEx(opening1, cv2.MORPH_CLOSE, kernel1)
          temp,contours1, hierarchy = cv2.findContours(closing1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

          if len(contours)>0:
              cnt = contours[0]
              M = cv2.moments(cnt)

              cx_blue = int(M['m10']/M['m00'])
              cy_blue = int(M['m01']/M['m00'])
              #perimeter = cv2.arcLength(cnt,True)
              x,y,w,h = cv2.boundingRect(cnt)
              cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)


              #print center

          else:
              s.send(stop)
              print "no BOT coontour detected"
              center = [prev_x,prev_y]
              cx_blue = prev_x -12
              cy_blue = prev_y-12


          if len(contours1)>0:
              cnt = contours1[0]
              M = cv2.moments(cnt)
              cx_yellow = int(M['m10']/M['m00'])
              cy_yellow= int(M['m01']/M['m00'])
              x,y,w,h = cv2.boundingRect(cnt)
              cv2.rectangle(img,(x,y),(x+w,y+h),(150,175,0),2)


              #print center

          else:
              s.send(stop)
              print "no BOT coontour detected no 1"
              center1 = [prev_x,prev_y]
              cx_yellow = prev_x - 12
              cy_yellow = prev_y -12


            
          center = [(cx_yellow+cx_blue)/2,(cy_yellow+cy_blue)/2]    
          dist_y = nodes[i+1][1]-center[1]
          dist_x = nodes[i+1][0]-center[0]

          if abs(prev_x - center[0])<3 and abs(prev_y - center[1])<3:  # means at rest
            stopped = 1
            
          else:
            stopped = 0
            

          if check_flag == 1:
                
                rotate, direction = find_direction(direction ,nodes[i+1],nodes[i])
                rotate_flag = 1
                check_flag =0

         
          if rotate_flag == True:
                if direction == west:
                  if abs(cy_blue-cy_yellow)<3 and (cx_blue-cx_yellow)>3:
                    rotate_flag = 0
                elif direction == east :
                  if abs(cy_blue-cy_yellow)<3 and (cx_blue-cx_yellow)<-3:
                    rotate_flag = 0
                elif direction == south:
                  if abs(cx_blue-cx_yellow)<3 and (cy_blue-cy_yellow)<-3:
                    rotate_flag = 0
                elif direction ==north:
                  if abs(cx_blue-cx_yellow)<3 and (cy_blue-cy_yellow)>3:
                    rotate_flag = 0

                if rotate_flag == 1:
                  if rotate == 'left':
                        
                        s.send(turn_left);
                        sleep(rotate_delay)
                        s.send(stop)
     
                  elif rotate == 'right':
                         s.send(turn_right)
                         sleep(rotate_delay)
                         s.send(stop)
                  elif rotate == 0:
                        s.send(stop)
                        rotate_flag =0
                        print "no direction change"
                  else:
                        s.send(stop)
                        
                        print "cannot decide turning direction"



         
          if direction == north and rotate_flag == False and stopped == 1:
             stopped =0

             if dist_y <= -long_dist:
                if dist_x <-5:
                  s.send(turn_left);
                  sleep(adjust_delay)
                  s.send(stop)
                  s.send(forward)
                  sleep(long_delay)
                  s.send(stop)
                  s.send(turn_right);
                  sleep(adjust_delay)
                  s.send(stop)
                elif dist_x >5:
                  s.send(turn_right);
                  sleep(adjust_delay)
                  s.send(stop)
                  s.send(forward)
                  sleep(long_delay)
                  s.send(stop)
                  s.send(turn_left);
                  sleep(adjust_delay)
                  s.send(stop)
                else:
                  s.send(forward)
                  sleep(long_delay)
                  s.send(stop)
            
             elif dist_y <-5 and dist_y> -long_dist:
                s.send(forward)
                sleep(short_delay)
                s.send(stop)
             elif dist_y >8:
                s.send(backward)
                sleep(short_delay)
                s.send(stop)
                
             else:
               s.send(stop)
          elif direction == south and rotate_flag == False and stopped == 1:
             stopped =0
            
             if dist_y >= long_dist:
                if dist_x <-5:
                  s.send(turn_right);
                  sleep(adjust_delay)
                  s.send(stop)
                  s.send(forward)
                  sleep(long_delay)
                  s.send(stop)
                  s.send(turn_left);
                  sleep(adjust_delay)
                  s.send(stop)
                elif dist_x >5:
                  s.send(turn_left);
                  sleep(adjust_delay)
                  s.send(stop)
                  s.send(forward)
                  sleep(long_delay)
                  s.send(stop)
                  s.send(turn_right);
                  sleep(adjust_delay)
                  s.send(stop)
                else:
                  s.send(forward)
                  sleep(long_delay)
                  s.send(stop)
                        
              
                
             elif dist_y >5 and dist_y<long_dist:
                s.send(forward)
                sleep(short_delay)
                s.send(stop)
             elif dist_y < -8:
                s.send(backward)
                sleep(short_delay)
                s.send(stop)
                
             else:
               s.send(stop)
                            
          elif direction == west and rotate_flag == False and stopped == 1:
             stopped =0
         
             if dist_x <= -long_dist:
                if dist_y <-5:
                  s.send(turn_right);
                  sleep(adjust_delay)
                  s.send(stop)
                  s.send(forward)
                  sleep(long_delay)
                  s.send(stop)
                  s.send(turn_left);
                  sleep(adjust_delay)
                  s.send(stop)
                elif dist_y >5:
                  s.send(turn_left);
                  sleep(adjust_delay)
                  s.send(stop)
                  s.send(forward)
                  sleep(long_delay)
                  s.send(stop)
                  s.send(turn_right);
                  sleep(adjust_delay)
                  s.send(stop)
                else:
                  s.send(forward)
                  sleep(long_delay)
                  s.send(stop)

             elif dist_x <-5 and dist_x> -long_dist:
                s.send(forward)
                sleep(short_delay)
                s.send(stop)
             elif dist_x >8:
                s.send(backward)
                sleep(short_delay)
                s.send(stop)
                
             else:
               s.send(stop)

          elif direction == east and rotate_flag == False and stopped == 1:
             stopped =0
             
             if dist_x >= long_dist:
                if dist_y <-5:
                  s.send(turn_left);
                  sleep(adjust_delay)
                  s.send(stop)
                  s.send(forward)
                  sleep(long_delay)
                  s.send(stop)
                  s.send(turn_right);
                  sleep(adjust_delay)
                  s.send(stop)
                elif dist_y >5:
                  s.send(turn_right);
                  sleep(adjust_delay)
                  s.send(stop)
                  s.send(forward)
                  sleep(long_delay)
                  s.send(stop)
                  s.send(turn_left);
                  sleep(adjust_delay)
                  s.send(stop)
                else:
                  s.send(forward)
                  sleep(long_delay)
                  s.send(stop)
               
             elif dist_x >5 and dist_x<long_dist:
                s.send(forward)
                sleep(short_delay)
                s.send(stop)
             elif dist_x < -8:
                s.send(backward)
                sleep(short_delay)
                s.send(stop)
                
             else:
               s.send(stop)

              
          
          #print dist_x, dist_y, direction, center1
          if (abs(dist_x) <=5 and (direction == 'east'or direction == 'west'))or(abs(dist_y) <=5 and (direction == 'south'or direction == 'north')):  # destination node reached
                i = i+1
                check_flag= 1
                print "***node reached****:  ", i
                if i == (len(nodes)-1):
                  break
               
                
                
          cv2.imshow('Vehicle Guidance', img)
          prev_x = center[0]
          prev_y = center[1]
          k= cv2.waitKey(5)
          if k == 27:
           s.close()
           break
      s.close()
      print "**solved**"
      del(cam)
      
      return;


def write_file(nodes):
      nodes = nodes[0]  # decrese by 1 dimension 
      file= open('./data/nodes.txt', 'w')  # 'encoding = 'utf-8' windows:cp1252' but 'utf-8' for linix

      for i in range(0, len(nodes)):
        file.write("%s\n" % nodes[i][0])
        file.write("%s\n" % nodes[i][1])
  
      file.close()
      print "Nodes saved in a file"
      print " "

      return

def main_program():
    
    size = 600,800, 3
    m = np.zeros(size, dtype=np.uint8)
    cv2.putText(m,  " Menu:", (25,80), cv2.FONT_HERSHEY_SIMPLEX, 1.5,(190,198, 155),2)
    cv2.putText(m,"1.Press Enter to solve puzzle", (25,155), cv2.FONT_HERSHEY_SIMPLEX, 1.5,(0,58, 255),2)
    cv2.putText(m,"2.Press 'c' to adjust HSV value", (25,225), cv2.FONT_HERSHEY_SIMPLEX, 1.5,(0,58, 255),2)
    cv2.putText(m,"3.Press 'b' to resume from ", (25,295), cv2.FONT_HERSHEY_SIMPLEX, 1.5,(0,58, 255),2)
    cv2.putText(m, "  saved nodes", (25,332), cv2.FONT_HERSHEY_SIMPLEX, 1.5,(0,58, 255),2)

    cv2.imshow('Vehicle Guidance' ,m)

    key = cv2.waitKey()
    cv2.destroyAllWindows()
    if key == ord('c'):
      HSVadjust()
      
    elif key == 13:
      top_left, top_right, bottom_left, bottom_right,\
              start_x, start_y, stop_x, stop_y = detect_features()
    
      nodes= findpath(start_x, start_y,\
          stop_x, stop_y, top_left, top_right, bottom_left, bottom_right )

      write_file(nodes)
      cv2.waitKey()
      control_bot()
      
    elif key == ord('b'):
      control_bot()
    cv2.destroyAllWindows()
    return;
  

main_program()

