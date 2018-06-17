    # -*- coding: utf-8 -*-
    """
    Created on Sun Jun 17 21:16:15 2018
    
    @author: yatheen!
    """
    """Step 1: Import Neccessary Packages """
    import cv2
    import numpy as np
    
    """Step 2: Read the image input"""
    image = cv2.imread('./images/image_4.jpg') # Image input
    image_gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY) # GrayScale Conversion
    template_image = cv2.imread('./images/template_4.jpg', 0) # Template input
    
    """Step 3 : Get the width and heigth of template in w and h """
    w, h = template_image.shape[::-1]
     
    """ Step 4 : Template Matching """
    result = cv2.matchTemplate(image_gray,template_image,cv2.TM_CCOEFF_NORMED)
     
    """ Step 5 : Specify a threshold to stipulate the search with matching accuracy """
    threshold = 0.8
    
    """ Step 6:  Store the coordinates of matched area in a numpy array """
    loc = np.where( result >= threshold) 
     
    """Step 7:  Draw a rectangle around the matched region """
    for pt in zip(*loc[::-1]):
        cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0,255,255), 1)
    
    cv2.imshow("Template Succesfully matched!" , image)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    
