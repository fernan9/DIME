# -*- coding: utf-8 -*-
"""
DIME
Detector of Insect Motion Endpoint
Version 1

-- quick prompt
import sys
sys.argv = ['DIME-cl.py','-v', '254-15112019.mp4', '-s', '84', '-i', '3', '-k', '1']
exec(open("BATTSI-v2.0.py").read())
--

@author: Rodrigo Perez
"""
import cv2
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
import argparse

# Function to extract frames 
def FirstFrame(folder, vidName): 
    # Path to video file
    vidObj = cv2.VideoCapture(os.path.join(folder, vidName))
    # checks whether frames were extracted 
    success = 1
    while success:
        # vidObj object calls read 
        # function extract frames 
        success, image = vidObj.read()
        # Saves the frames with frame-count
        fFrameName = "{}-firstFrame.jpg".format(vidName.split('.')[0])
        midDir = "DIME-{}".format(vidName.split('.')[0])
        imageDir = os.path.join(folder, midDir, fFrameName)
        print(imageDir)
        cv2.imwrite(imageDir, image)
        print(" - captured!\n")
        return imageDir

# Function to select a rectangle on image
def shape_selection(event, x, y, flags, param): 
    # grab references to the global variables 
    global ref_point, crop
    # if the left mouse button was clicked, record the starting 
    # (x, y) coordinates and indicate that cropping is being performed 
    if event == cv2.EVENT_LBUTTONDOWN: 
        ref_point = [(x, y)]
        #print('click')
    # check to see if the left mouse button was released 
    elif event == cv2.EVENT_LBUTTONUP: 
        # record the ending (x, y) coordinates and indicate that 
        # the cropping operation is finished 
        ref_point.append((x, y))
        #print('click2')
        # draw a rectangle around the region of interest 
        cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2) 
        cv2.imshow("DIME", image)

# function to find a file
def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)

def nothing(x):
    pass
                  
def main():            
    # caputure variables from terminal arguments
    # example:
    # BATTSI_V2.0.py HKDBerea826F4.mp4 --sampleSize (int) --calibration (default/manual) --iteration --kernel 
    parser = argparse.ArgumentParser(description="Detector of Insect Motion Endpoint")
    
    parser.add_argument('-v', '--videoName', type=str, required=True,
                        help='How many individuals to evaluate?')
    
    parser.add_argument('-s', '--sampleSize', type=int, required=True,
                        help='How many individuals to evaluate?')
    
    parser.add_argument('-i', '--iterationNumber', type=int, default = 3,
                        help='How many cycles of dilation filter?')
    
    parser.add_argument('-k', '--kernelSize', type=int, default=3,
                        help='What size of kernel (2k+1) for blur filter?')
    
    parser.add_argument('-f', '--firstFrame', action='store_true',
                        help='Return first frame instead of last')
    args = parser.parse_args()
    
    videoName = args.videoName
    sampleSize = args.sampleSize
    dilationIter = args.iterationNumber
    blurKernel = args.kernelSize*2+1
    ff = args.firstFrame
    #if args.calibration:
       #calibrate = True
    
    # Initiaize some variables
    directory = os.getcwd()
    isFile = False
    
    print('--------------------')
    print('### DIME - v1.0 ###')
    print('--------------------\n')
    
    print('--------------')
    print("#   SET UP   #")
    print('--------------')
    
    while isFile != True:
        path = os.path.join(directory, videoName)
        print(path)
        isFile = os.path.isfile(path)
        if isFile == False:
            print(" - Video file not found")
            quit()
        else:
            print(" - Video file found!\n")
    
    print(" - Directory setup...")
    middleDir = "DIME-{}".format(videoName.split('.')[0])
    isExist = os.path.exists(os.path.join(directory, middleDir))
    resultsDir = os.path.join(directory,middleDir)
    if not isExist:
        os.makedirs(resultsDir)
        print(" - Results directory for {} created".format(videoName))
    
    else:
        print(" -> Directory for {} results already exists".format(videoName))
        print(" -> Results with new parameters will be created...")
        
    
    print("\n - looking for first frame...")
    videoImage = FirstFrame(directory, videoName)
    print('-------------------')
    print('DIME will analyze {} samples in {}'.format(sampleSize, videoName))
    print('-------------------\n')
    
    print("Looking for a previous well file for your video...")
    print("* sample sizes should match *")
    
    # look for a possible array file with the same video name
    findPozos = find( videoName.split('.')[0] + "-" + str(sampleSize) + ".npy", resultsDir)
    # if there is such file, use it, otherwise proceed to well detection
    if findPozos is None:
        print("Well file not found, let's make a new one!")
        print('#  DEFINE WELLS  #\n------\nTO draw one Region of Interest (ROI)')
        print('1. click-and-hold TOP RIGHT corner')
        print('2. drag-while-holding BOTTOM LEFT corner')
        print('3. release')
        print('------\n')
        # initialize the well column/row list
        nombres = []
        for i in range(sampleSize):
            a = 'well-%i' %(i+1)
            nombres.append(a)
        #print(nombres)
        # now let's initialize the list of reference point 
        global ref_point
        ref_point = []
        pozos = []
        global image
        image = cv2.imread(videoImage) 
        clone = image.copy()
        cv2.namedWindow("DIME")
        cv2.setMouseCallback("DIME", shape_selection)
        # run for all cases
        for i in nombres:
            print("-> Draw ROI for {} <------> Press 'c' to confirm <-".format(i))
            # keep looping until the 'q' key is pressed 
            while True: 
                # display the image and wait for a keypress 
                cv2.imshow("DIME", image) 
                key = cv2.waitKey(1) & 0xFF
                # press 'r' to reset the window 
                if key == ord("r"): 
                    image = clone.copy()
                # if the 'c' key is pressed, break from the loop 
                elif key == ord("c"): 
                    break
            if len(ref_point) == 2: 
                pozos.append(ref_point)
                #print(ref_point)
                print("   - coordinates: {}\n\n     <-> TO continue, press 'c'".format(ref_point))
                cv2.waitKey(0)
        # close all open windows 
        cv2.destroyAllWindows()
        # save pozos list as an array
        pozosSave = np.asarray(pozos)
        pozosSaveName = videoName.split('.')[0] + "-" + str(sampleSize)
        np.save(os.path.join(middleDir, pozosSaveName),pozosSave)
    else:
        print(' -> Well file found!\n')
        pozos = np.load(find(videoName.split('.')[0] + "-" + str(sampleSize) + ".npy", resultsDir))
        pozos = pozos.tolist()
    
    
    print('--------------------')
    print('#  TRANSFORMATION  #')
    print('--------------------')
    print(" - 0 %")
    cap = cv2.VideoCapture(path)
    cuadrosTot = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cuadros = np.ceil(cap.get(cv2.CAP_PROP_FPS))
    
    salida = [[]]
    
    i = 0
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    
    nombres = []
    for i in range(sampleSize):
        a = '%i' %(i+1)
        nombres.append(a)
        
    #
    while cap.isOpened():
        if ret:
            diff = cv2.absdiff(frame1, frame2)
            gray = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (blurKernel,blurKernel), 0)
            _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
            dilated = cv2.dilate(thresh, None, iterations=dilationIter)
    
            frame1 = frame2
            ret, frame2 = cap.read()
            
            temp2 = []
            
            clone=dilated.copy()
            for j in range(len(pozos)):
                temp = clone[pozos[j][0][1]:pozos[j][1][1],pozos[j][0][0]:pozos[j][1][0]]
                actividad =  np.sum(temp)/(temp.shape[0]*temp.shape[1])
                temp2.append(actividad)
            salida.append(temp2)
            
            i+=1
            if i % 10000 == 0: print(" - {} %".format(round(i/cuadrosTot*100,2)))
        elif i > cuadrosTot-1 :         # video length should be here
            print(" - 100 %")
            break
        else:
            print(" - Invalid frame at {} position".format(i))
            ret, frame2 = cap.read()
            i+=1
    
    cv2.destroyAllWindows()
    cap.release()
    
    print("\n------\nVideo file: {}".format(videoName))
    print("Frame rate: {}".format(cuadros))
    print("Frame differences analyzed: {}".format(cuadrosTot))
    
    
    print('--------------------')
    print('#  RESULT SUMMARY  #')
    print('--------------------')
    
    print("\n - Plotting figure...")
    # Plot of activity traces
    df=pd.DataFrame(salida,columns=nombres)
    df['seconds'] = df.index / cuadros
    df['minutes'] = df.index / cuadros / 60
    colnames = list(df.columns)
    df.plot(x='minutes', y=colnames[:-1], figsize=(10,4*sampleSize), subplots = True)
    
    datasetName = '{}-Activity-k{}-i{}.pdf'.format(videoName.split('.')[0],blurKernel,dilationIter)
    plt.savefig(os.path.join(middleDir,datasetName))      # k=kernel i=iterations
    print(" - Figure plotted!\n")
    
    print("-----------------------")
    print("# KNOCK DOWN ANALYSIS #")
    print("-----------------------\n")
    
    print(' -> Number of iteration in Dilation filter: {}'.format(dilationIter))
    print(' -> Size of blur kernel: {}'.format(blurKernel))
    # Critical temperature analysis
    
    print(" - Running analysis...")
    
    # list containing last frame
    lastFrameMedian = []
    
    
    for column in df:
        if column == 'seconds': break
        if column == 'minutes': break
        # transforms data into list for Last Frame analysis
        s = df[column].tolist()
        p = df[column].to_numpy()
        
        # calculate last frame
        
        tempP = p[np.nonzero(p)]
        if tempP.sum == 0:
            medianTemp = 0
        else:
            medianTemp = np.nanmedian(tempP)
        #print(len(p),len(p[np.nonzero(p)]), np.mean(p[np.nonzero(p)]), meanTemp)
        lsMedian = [k for k, e in enumerate(s) if e > medianTemp]
        # if no event above criteria append zero value
        if len(lsMedian) == 0:
            lastFrameMedian.append(0)
        # if there is a list append the last element
        elif ff == False:
            lastFrameMedian.append(lsMedian[(len(lsMedian)-1)]/cuadros/60) # units in minutes
        else:
            print("in first frame mode")
            lastFrameMedian.append(lsMedian[0]/cuadros/60)
        
        
        print(' > Iteration for well {} done'.format(column))
    
    print(" - Saving dataset...")
    dataName = '{}-Dataframe-k{}-i{}.csv'.format(videoName.split('.')[0],blurKernel,dilationIter)
    df.to_csv(os.path.join(middleDir, dataName))
    #df.to_csv('BATTSI-{}\\DataFrame-{}-k{}-i{}.csv'.format(videoName[:-4],videoName[:-4],blurKernel,dilationIter))  # where to save it, usually as a csv
    
    print(" -> Saved as CSV file in folder DIME-{}\n".format(videoName[:-4]))
    #Then you can load it back using:
    #df = pd.read_pickle(file_name)
    
    print(" - Saving results table... (sample, knockdown time (seconds))")
    resultados = {'Well':nombres,
                  'LastFrameMedian':lastFrameMedian}
    dfctmax = pd.DataFrame(resultados)
    
    resultName = '{}-Results-k{}-i{}.csv'.format(videoName.split('.')[0],blurKernel,dilationIter)
    dfctmax.to_csv(os.path.join(middleDir,resultName))
    print(" -> Saved as CSV file in folder DIME-{}\n".format(videoName[:-4]))
    
    logName = "{}-Log-k{}-i{}.txt".format(videoName.split('.')[0],blurKernel, dilationIter)
    
    # print info to log file
    log = open(os.path.join(middleDir,logName), 'a')
    
    log.write("--\nDIME v1.0\n--\n - RUN begins\n")
    log.write("Video file analyzed: {}\n".format(videoName))
    log.write("Frame rate: {}\n".format(cuadros))
    log.write("Number of frame differences transformed: {}\n--\n".format(cuadrosTot))
    log.write("number of samples to be analyzed: {}\n".format(sampleSize))
    #if calibrate == "m":
    #    log.write("Calibration type selected: manual\n")
       # elif calibrate == 'd':
    #    log.write("Calibration type selected: default\n")
    log.write("Iteration number for Dilation: {}\n".format(dilationIter))
    log.write("Kernel size for Gaussian Blur: {}\n".format(blurKernel))
    #log.write("Knock down threshold criteria: {} %\n".format(criteria))
    log.write("--\nRUN ends at {}\n--\n\n".format(datetime.now()))
    log.close()
    
    print(" - Run info logged at TXT file")
    print("------\nDIME run finished\n------")

if __name__ == '__main__':
   main()