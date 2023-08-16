# -*- coding: utf-8 -*-
"""
DIME: Digital Imaging of Motion Events

This script processes a video file to analyze motion events in a set of samples (wells) using computer vision techniques.
It calculates the activity of each well by applying a series of image processing steps, including frame differencing,
grayscale conversion, Gaussian blur, thresholding, and dilation.

The results are then plotted and saved as a figure (PDF), a DataFrame (CSV), and a summary table (CSV) in a specified directory.
Additionally, a log file (TXT) is created to store run information.

Usage:
  Modify the configuration variables (e.g., video path, sample size, blur kernel size, dilation iterations, etc.)
  and run the script. Ensure that the required packages (e.g., OpenCV, NumPy, Pandas, Matplotlib) are installed.

Example:
  python dime.py

Author: Fernan Rodrigo Perez Galvez
Date: 4/5/2023

DIME
Detector of Insect Motion Endpoint
Version 1.1

REQUIREMENTS
pip install -r requirements.txt

-- quick prompt
import sys
sys.argv = ['DIME-cl.py','-v', '254-15112019.mp4', '-s', '84', '-i', '3', '-k', '1']
exec(open("BATTSI-v2.0.py").read())
--

"""
import cv2
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
import sys
import logging
import math

# not using the GStreamer library for now
# can install by sudo apt-get install gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly
os.environ["GST_DEBUG"] = "0"

# Function to extract frames 
def FirstFrame(directory_working, vidName, directory_video): 
    """
    Takes a video file and returns the first frame for further image transformation.
    Parameters:
        directory (str): The directory where the video file is located.
        file (str): The name of the video file.
        directoryMiddle (str): The directory where the output file will be saved.
    Returns:
        numpy.ndarray: The first frame of the video.
    """
    print(" - Looking for first frame of video for well detection")
    # Path to video file
    vidObj = cv2.VideoCapture(os.path.join(directory_video, vidName))
    # checks whether frames were extracted 
    SUCCESS = 1
    while SUCCESS:
        # vidObj object calls read 
        # function extract frames 
        SUCCESS, image = vidObj.read()
        # Saves the frames with frame-count
        fFrameName = "{}-firstFrame.jpg".format(vidName.split('.')[0])
        imageDir = os.path.join(directory_working, fFrameName)
        print(imageDir)
        cv2.imwrite(imageDir, image)
        print(" - captured!\n")
        print(" - First frame of video captured for well detection")
        return imageDir

# Function to select a rectangle on image
def shape_selection(event, x, y, flags, param): 
    """
    A mouse callback function that selects a ROI from a video frame.
    
    Parameters:
    -----------
    event : int
        The event type (left button down, left button up, etc.).
    x : int
        The x-coordinate of the event.
    y : int
        The y-coordinate of the event.
    flags : int
        The event flags.
    param : dict
        A dictionary that contains the following keys:
            - 'frame': the current frame being processed
            - 'refPt': a list of reference points for the selected ROI
            - 'crop': a boolean that indicates whether to crop the selected ROI
            - 'cancel': a boolean that indicates whether to cancel the selection
    
    Returns:
    --------
    None
    """
    COLOR_RECTANGLE = (0, 255, 0)
    THICKNESS = 2
    # grab references to the global variables 
    global ref_point, image
    # if the left mouse button was clicked, record the starting 
    # (x, y) coordinates and indicate that cropping is being performed 
    if event == cv2.EVENT_LBUTTONDOWN: 
        ref_point = [(x, y)]
        cropping = True
    # check to see if the left mouse button was released 
    elif event == cv2.EVENT_LBUTTONUP: 
        # record the ending (x, y) coordinates and indicate that 
        # the cropping operation is finished 
        ref_point.append((x, y))
        cropping = False
        # draw a rectangle around the region of interest 
        cv2.rectangle(image, ref_point[0], ref_point[1], COLOR_RECTANGLE, THICKNESS) 


# function to find a file
def find(name, path):
    """
    Searches for a file with a given name in a specified path.
    Parameters:
        name (str): The name of the file to be searched for.
        path (str): The directory where the file should be located.
    Returns:
        str: The full path of the file if found, or None if not found.
    """
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)

       
def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        An object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description='DIME')
    parser.add_argument('--videoName', '-v', type=str, metavar='<string>', dest='videoName', required=True,
                        help='Name of the video file to analyze')
    parser.add_argument('--sampleSize', '-s', type=int, metavar='<int>', dest='sampleSize', required=True,
                        help='Number of wells in the plate (default: 96)')
    parser.add_argument('--iterationNumber', '-i', type=int, default=2, metavar='<int>', dest='iterationNumber',
                        help='Number of iterations for the dilation filter (default: 2)')
    parser.add_argument('--kernelSize', '-k', type=int, default=3, metavar='<int>', dest='kernelSize',
                        help='Size of the kernel for the Gaussian blur (format Kernel = 2k+1, default: 2(3) +1)')
    parser.add_argument('--reverseScoring', '-r', action='store_true', default=False, dest='reverseScoring',
                        help='If set, reverse detection of the first event will be calculated (default: False)')
    parser.add_argument('--filterNoise', '-f', type=float, default=math.nan, metavar='<flt>', dest='filterNoise',
                        help='Apply percent ratio filter based on average activity in total frame (default:None')
    parser.add_argument('--numberFrames', '-n', type=int, default=1, metavar='<int>', dest='numberFrames',
                        help='Number of frames to sample for knockdown analysis (default: 1)')
    
    args = parser.parse_args()
    if not args.videoName:
        parser.error('Video name not provided. Please provide a name for the video file to analyze using the --videoName (-v) option.')
    return args


def print_progress(current_frame,total_frames):
    progress = (current_frame / total_frames) * 100
    print(f"Progress: {progress:.2f}%") # format to 2 decimals
    
def process_frame(frame1, frame2):
    """
    Processes an image frame by applying various image transformation filters.
    Parameters:
        frame1 (array): The image frame to be processed.
    Returns:
        numpy.ndarray: The processed image frame.
    """

    gray1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    diff_matrix = cv2.absdiff(gray1, gray2)
    blur = cv2.GaussianBlur(diff_matrix, (K_BLUR,K_BLUR), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    processed_frame = cv2.dilate(thresh, None, iterations=I_DILATION)
    return processed_frame

def setup_logging(log_file):
    global logger
    log_path = os.path.join(os.getcwd(), log_file)
    print(f"Logging to file: {log_path}")
    logging.basicConfig(filename=log_file,
                        filemode = 'w',
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        force = True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
def define_wells(VIDEO_NAME, sample_size, well_names, directory, videoImage):
    """
    Automatically defines regions of interest in an image frame based on a set sample size.
    Parameters:
        VIDEO_NAME (str): The name of the video file.
        SAMPLE_SIZE (int): The number of wells to be defined.
    Returns:
        numpy.ndarray: The defined wells.
    """
    
    global ref_point, image, nombres
    
    print("Well file not found, let's make a new one!")
    print('#  DEFINE WELLS  #\n------\nTO draw one Region of Interest (ROI)')
    print('1. click-and-hold TOP RIGHT corner')
    print('2. drag-while-holding BOTTOM LEFT corner')
    print('3. release')
    print('------\n')

    # now let's initialize the list of reference point
    ref_point = []
    temp_wells = []
    
    image = cv2.imread(videoImage)
    clone = image.copy()
    cv2.namedWindow("DIME")
    cv2.setMouseCallback("DIME", shape_selection)

    # run for all cases
    for i in well_names:
        print("-> Draw ROI for {} <------> Press 'c' to confirm or 'r' to reset <-".format(i))
        # keep looping until the 'q' key is pressed
        while True:
            # display the image and wait for a keypress
            cv2.imshow("DIME", image)
            key = cv2.waitKey(1) & 0xFF
            reset = False
            confirm = False
            # press 'r' to reset the window
            if key == ord("r") and not confirm:
                image = clone.copy()
                ref_point = []
                for well in temp_wells:
                    cv2.rectangle(image, well[0], well[1], (0, 0, 255), 2)
                reset = True
            # if the 'c' key is pressed, break from the loop
            elif key == ord("c"):
                confirm = True
                break
        if len(ref_point) == 2 and not reset:
            temp_wells.append(ref_point)
            cv2.rectangle(image, ref_point[0], ref_point[1], (0, 0, 255), 2)
            print("   - coordinates: {}\n\n     <-> TO continue, press 'c'\n".format(ref_point))
            cv2.waitKey(0)
    # close all open windows
    cv2.destroyAllWindows()

    # save wells list as an array
    temp_wells_save = np.asarray(temp_wells)
    temp_wells_save_name = VIDEO_NAME.split('.')[0] + "_s" + str(sample_size)
    temp_file = os.path.join(directory, temp_wells_save_name)
    np.save(temp_file, temp_wells_save)
    print(" - Well file created and saved on results directory")
    logger.info(" - Well file created and saved on results directory")
    
    # print image with rectangles and wells
    final_image = clone.copy()
    for i, well in enumerate(temp_wells):
        # draw rectangle
        cv2.rectangle(final_image, well[0], well[1], (0, 0, 255), 1)
        
        # add well number inside rectangle
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = str(i+1)
        text_size, _ = cv2.getTextSize(text, font, 1, 2)
        text_x = well[0][0] + (well[1][0] - well[0][0]) // 2 - text_size[0] // 2
        text_y = well[0][1] + (well[1][1] - well[0][1]) // 2 + text_size[1] // 2
        font_scale = (well[1][0] - well[0][0]) / 80  # adjust font size based on rectangle size
        cv2.putText(final_image, text, (text_x, text_y), font, font_scale, (0, 255, 0), 1)
    
    # save image
    image_name = temp_wells_save_name +'.jpg'
    image_path = os.path.join(directory, image_name)
    cv2.imwrite(image_path, final_image)
    
    return temp_wells
    
def process_wells(wells_list, processed_frame):
    """
    Processes regions of interest in an image frame by calculating the activity level of each well.
    Parameters:
        wells (list): A list of wells defining the regions of interest.
        processed_frame (numpy.ndarray): The processed image frame.
        names (list): A list of well names.
    Returns:
        list: The activity level of each well in the image frame.
    """
    well_activities = []
    for j in range(len(wells_list)):
        
        # run for each row of the wells list
        temp = processed_frame[wells_list[j][0][1]:wells_list[j][1][1],wells_list[j][0][0]:wells_list[j][1][0]]
        
        # compute an average for the activity
        actividad =  np.sum(temp)/(temp.shape[0]*temp.shape[1])
        
        # append to returning list
        well_activities.append(actividad)
    return well_activities

def calculate_last_frame(df, num_last_frames=1):
    """
    Calculates the last frame in which each well's activity level is above a specified threshold.
    Parameters:
        df (pandas.DataFrame): A dataframe containing the activity levels of each well over time.
        num_last_frames (int): The number of frames to be used in the calculation.
    Returns:
        list: The last frame in which each well's activity level is above the specified threshold.
        list: The thresholds used for each well.
        list: The time in minutes of the last frame for each well.
    """
    lastFrameMedian = []
    lastTimeMedian = []
    lastThresholdMedian = []
    time = df['minutes']
    for column in df:
        if column == 'seconds': 
            break
        if column == 'minutes': 
            
            break
        
        # get values for current well
        s = df[column].tolist()
        p = df[column].to_numpy()
        
        # remove zero values from array
        p = p[p != 0]
        
        # calculate median of non-zero values
        medianTemp = np.nanmedian(p)
        lastThresholdMedian.append(medianTemp)
        
        # calculate indices of values above median
        lsMedian = [k for k, e in enumerate(s) if e > medianTemp]
        
        # calculate last frames
        if len(lsMedian) == 0:
            lastFrameMedian.append(0)
            lastTimeMedian.append(0)
            
        elif len(lsMedian) < num_last_frames:
            last_frames = lsMedian
            lastFrameMedian.append(round(np.mean(last_frames)))
            lastTimeMedian.append(np.mean(time[last_frames]))
            
        else:
            last_frames = lsMedian[-num_last_frames:]
            lastFrameMedian.append(round(np.mean(last_frames)))
            lastTimeMedian.append(np.mean(time[last_frames]))
        
 
        logger.info(' > Iteration for well {} done'.format(column))
        
    return lastFrameMedian, lastThresholdMedian, lastTimeMedian

def calculate_first_frame(df, num_first_frames=1):
    """
    Calculates the first frame in which each well's activity level is above a specified threshold.
    Parameters:
        df (pandas.DataFrame): A dataframe containing the activity levels of each well over time.
        num_first_frames (int): The number of frames to be used in the calculation.
    Returns:
        list: The first frame in which each well's activity level is above the specified threshold.
        list: The thresholds used for each well.
        list: The time in minutes of the first frame for each well.
    """
    firstFrameMedian = []
    firstTimeMedian = []
    firstThresholdMedian = []
    time = df['minutes']
    for column in df:
        if column == 'seconds': 
            break
        if column == 'minutes': 
            break
        
        # get values for current well
        s = df[column].tolist()
        p = df[column].to_numpy()
        
        # remove zero values from array
        p = p[p != 0]
        
        # calculate median of non-zero values
        medianTemp = np.nanmedian(p)
        firstThresholdMedian.append(medianTemp)
        
        # calculate indices of values above median
        lsMedian = [k for k, e in enumerate(s) if e > medianTemp]
        
        # calculate last frames
        if len(lsMedian) == 0:
            firstFrameMedian.append(0)
            firstTimeMedian.append(time.tail(1).values[0])
            
        elif len(lsMedian) < num_first_frames:
            first_frames = lsMedian
            firstFrameMedian.append(round(np.mean(first_frames)))
            firstTimeMedian.append(np.mean(time[first_frames]))
        else:
            first_frames = lsMedian[:num_first_frames]
            firstFrameMedian.append(round(np.mean(first_frames)))
            firstTimeMedian.append(np.mean(time[first_frames]))
            
        logger.info(' > Iteration for well {} done'.format(column))
        
    return firstFrameMedian, firstThresholdMedian, firstTimeMedian

def process_video(video_path, wells, data_name, results_dir, filter_noise, filter_threshold, sufix):

    # open video file
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open the video file: {video_path}")
    except ValueError as e:
        logger.info(e)
        sys.exit(1)

    # get video information
    FRAMES_TOTAL = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    FRAMES = np.ceil(cap.get(cv2.CAP_PROP_FPS))

    # start reading frames for transformation
    print('#  TRANSFORMATION  #')
    print(" - 0 %")
    # initialize variables
    salida = [[]]
    total_change = []
    total_filtered = []
    empty = []
    i = 0
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    
    if filter_noise: #bool
        logger.info(f" - Filter noise above {filter_threshold}% total activity")
    filter_threshold #float
    # cycle of video analysis
    while cap.isOpened():
        if ret:
            # process frame
            processed_frame = process_frame(frame1, frame2)
            
            # ratio filter
            total_movement = np.sum(processed_frame)/(processed_frame.shape[0]*processed_frame.shape[1])
            total_change.append(total_movement)
            if filter_noise:
                if total_movement > filter_threshold:
                    # pass to 0
                    processed_frame[processed_frame != 0] = 0
                    logger.info(f" - Ratio filter at frame {i} of {total_movement} gain")
                total_movement = np.sum(processed_frame)/(processed_frame.shape[0]*processed_frame.shape[1])
                total_filtered.append(total_movement)
            # process each roi in the processed_frame
            well_activities = process_wells(wells, processed_frame)
            # fill the final dataset 'salida'
            salida.append(well_activities)

            # move to next frame
            frame1 = frame2
            ret, frame2 = cap.read()
            # advance the frame counter
            i+=1
            # print progress
            if i % 10000 == 0: print_progress(i,FRAMES_TOTAL)

        # end of video flag
        elif i > FRAMES_TOTAL-1 :
            logger.info(" - 100 %")
            break
        # flag for missing frames
        else:
            logger.info(" - Invalid frame at {} position".format(i))
            for k in range(len(wells)):
                empty.append(0)
            salida.append(empty)
            ret, frame2 = cap.read()
            i+=1

    cv2.destroyAllWindows()
    cap.release()

    # some logging
    logger.info(" - Saving dataset...")
    print(" - Saving dataset...")
    logger.info("Video file: {}".format(VIDEO_NAME))
    logger.info("Frame rate: {}".format(FRAMES))
    logger.info("Frame differences analyzed: {}".format(FRAMES_TOTAL))

    # generate time columns
    df=pd.DataFrame(salida,columns=nombres)
    df['seconds'] = df.index / FRAMES
    df['minutes'] = df.index / FRAMES/ 60
    
    # save to file
    df.to_csv(os.path.join(results_dir, data_name))
    logger.info(" -> Saved as CSV file in folder DIME-{}\n".format(VIDEO_NAME.split('.')[0]))

    
    
    sufix_nochange = sufix + '_nofilter'
    # plot total change initial
    x = range(len(total_change))/ FRAMES/ 60
    df1 = pd.DataFrame(total_change, columns=["Total"])
    total_change_array = df1['Total'].to_numpy()
    total_change_median = np.median(total_change_array[total_change_array > 0])
    plt.plot(x,total_change)
    plt.xlabel('Minutes')
    plt.ylabel('PIC')
    plt.title(f'Total activity - Median:{total_change_median:.2f}' )
    plt.savefig(os.path.join(results_dir, ("Total_trace_"+ sufix_nochange+'.png')), dpi=300)
    plt.clf()
    df1.to_csv(os.path.join(results_dir,('Total_activity_'+ sufix_nochange +'.csv')))
    # histogram
    total_change_array = total_change_array[total_change_array > 0]
    plt.yscale("log")
    plt.hist(total_change, bins=20)
    plt.xlabel('PIC')
    plt.ylabel('Frequency')
    plt.title('Histogram of Data - No filter')
    plt.savefig(os.path.join(results_dir, ("Total_histogram_" + sufix_nochange + '.png')), dpi=300)
    plt.clf()
    
    if filter_noise:
        sufix_change = sufix + f'_f{filter_threshold}'
        # plot total change filtered
        df2 = pd.DataFrame(total_filtered, columns = ['Total'])
        total_change_array = df2['Total'].to_numpy()
        total_change_median = np.median(total_change_array[total_change_array > 0])
        plt.plot(x,total_filtered)        
        plt.xlabel('Minutes')
        plt.ylabel('PIC')
        plt.title(f'Total filtered activity - Median:{total_change_median:.2f} - Filter:{filter_threshold}')
        plt.savefig(os.path.join(results_dir, ("Total_trace_filtered_" + sufix_change + '.png')), dpi=300)
        plt.clf()
        df2.to_csv(os.path.join(results_dir,("Total_activity_filtered_" + sufix_change + '.csv')))
        # histogram
        total_change_array = total_change_array[total_change_array > 0]
        plt.yscale("log")
        plt.hist(total_filtered, bins=20)
        plt.xlabel('PIC')
        plt.ylabel('Frequency')
        plt.title('Histogram of Filtered Data')
        plt.savefig(os.path.join(results_dir, ("Total_histogram_filtered_" + sufix_change + '.png')), dpi=300)
        plt.clf()

    return df

def test_wells():
    results_directory = r'C:\Users\Rodrigo Perez\OneDrive - University of Kentucky\PixelChange as Biological Activity\Final Datasets\00_Development\DIME-test_ctmax'
    directory = r'C:\Users\Rodrigo Perez\OneDrive - University of Kentucky\PixelChange as Biological Activity\Final Datasets\00_Development'
    VIDEO_NAME = 'test_ctmax.mp4'
    SAMPLE_SIZE = 4
    nombres = ['w1', 'w2', 'w3', 'w4']
    video_image = FirstFrame(results_directory, VIDEO_NAME, directory)
    wells = define_wells(VIDEO_NAME, SAMPLE_SIZE, nombres, results_directory, video_image) 
    print(wells)
    
def main():    
    global VIDEO_NAME, SAMPLE_SIZE, I_DILATION, K_BLUR, REVERSE, videoImage, new_directory, nombres
    print('DIME ----*____')    
    print('Setup...')
    
    # Initiaize some variables
    args = parse_arguments()
    FRAME_NUMBER = args.numberFrames
    SAMPLE_SIZE = args.sampleSize
    I_DILATION = args.iterationNumber
    K_BLUR = args.kernelSize*2+1
    REVERSE = args.reverseScoring
    FILTER_THRESHOLD = args.filterNoise
    FILTER = not np.isnan(FILTER_THRESHOLD)
    
    # correct for absolute - relative directory
    if os.path.sep in args.videoName:
        video_path = os.path.abspath(args.videoName)
    else:
        video_path = os.path.abspath(os.path.join(os.getcwd(), args.videoName))
    
    # directory locations
    VIDEO_NAME = os.path.basename(video_path)
    directory = os.path.dirname(video_path)
    
    # make the names for the wells
    nombres = []
    for i in range(SAMPLE_SIZE):
        a = 'w%i' %(i+1)
        nombres.append(a)
    #filter sufix
    if not FILTER:
        filter_sufix =  '_nofilter'
    else:
        filter_sufix = f'_f{FILTER_THRESHOLD}'
    
    # file names
    plotName = '{}_Activity_k{}i{}n{}size{}{}.png'.format(VIDEO_NAME.split('.')[0],K_BLUR,I_DILATION,FRAME_NUMBER, SAMPLE_SIZE, filter_sufix)# k=kernel i=iterations
    dataName = '{}_Dataframe_k{}i{}size{}{}.csv'.format(VIDEO_NAME.split('.')[0],K_BLUR,I_DILATION, SAMPLE_SIZE, filter_sufix)
    resultName = '{}_Results_k{}i{}n{}size{}{}.csv'.format(VIDEO_NAME.split('.')[0],K_BLUR,I_DILATION,FRAME_NUMBER, SAMPLE_SIZE, filter_sufix)
    log_file = "{}_Log_k{}i{}size{}{}.log".format(VIDEO_NAME.split('.')[0],K_BLUR, I_DILATION, SAMPLE_SIZE, filter_sufix)
    
    sufix = f'_k{K_BLUR}i{I_DILATION}'
    
    # find video file
    isFile = False
    while isFile != True:
        path = os.path.join(directory, VIDEO_NAME)
        print('Looking for video')
        isFile = os.path.isfile(path)
        if isFile == False:
            print(" - Video file not found")
            quit()
        else:
            print(" - Video file found!\n")
    
    # create new directory    
    print("Directory setup...")
    new_directory = "DIME-{}".format(VIDEO_NAME.split('.')[0])
    results_directory = os.path.join(directory,new_directory)
    isExist = os.path.exists(results_directory)
    if not isExist:
        try:
            os.makedirs(results_directory, exist_ok=True)
        except OSError as e:
            print(f"Cannot create/access the output directory: \n{results_directory}")
            print(e)
            sys.exit(1)
        print(f" - Results directory for {VIDEO_NAME} created in: \n{results_directory}")
    else:
        print(f" -> Directory for {VIDEO_NAME} results already exists in:\n{results_directory}")
        print(" -> Results with new parameters will be created")
    
    # create log
    log_path = os.path.join(results_directory, log_file)
    setup_logging(log_path)
    if os.path.exists(log_path):
        print("Logging file exists!")
    else:
        print("Logging file does not exist!")
    logger.info(f" - Video file found: {VIDEO_NAME}")
    logger.info(f" - Results directory for  in: {results_directory}")
    
    # load first frame for wells
    print(f'DIME will analyze {SAMPLE_SIZE} samples')
    logger.info(f'DIME will analyze {SAMPLE_SIZE} samples')
    # look for a possible array file with the same video name
    print(" - Looking for a previous well file for your video...")
    print(" -- Note: sample sizes must match")
    find_wells_file = find( VIDEO_NAME.split('.')[0] + "_s" + str(SAMPLE_SIZE) + ".npy", results_directory)

    # well creation
    if find_wells_file is None:
        print(' -> Well file will be created')
        logger.info(' -> Well file will be created')
        video_image = FirstFrame(results_directory, VIDEO_NAME, directory)
        wells = define_wells(VIDEO_NAME, SAMPLE_SIZE, nombres, results_directory, video_image) 
    else:
        print(' -> Well file found')
        logger.info(' -> Well file found')
        wellLocation = os.path.join(results_directory, VIDEO_NAME.split('.')[0] + "_s" + str(SAMPLE_SIZE) + ".npy")
        wells = np.load(wellLocation)
        wells = wells.tolist()    
    
    # transform video to movement data
    if os.path.isfile(os.path.join(results_directory, dataName)):
        logger.info(" - a Data Frame file exist in directory...")
        response = input("Data Frame file already exists. Do you want to load it? (y/n): ")
        if response.lower() == 'y':
            df = pd.read_csv(os.path.join(results_directory, dataName), index_col=0)
            print("Data Frame file loaded successfully.")
            logger.info(" - Data Frame file loaded successfully.")
            
        else:
            print("Data Frame file will be overwritten.")
            logger.info(" - Data Frame file will be overwritten.")
            df = process_video(video_path, wells, dataName, results_directory, FILTER, FILTER_THRESHOLD, sufix)
    else:
        print("Data Frame file will be created.")
        logger.info(" - Data Frame file will be created.")
        df = process_video(video_path, wells, dataName, results_directory, FILTER, FILTER_THRESHOLD, sufix)
   
    # Critical temperature analysis    
    colnames = list(df.columns)
    print("# KNOCK DOWN ANALYSIS #")
    print(f' -> Number of iteration in Dilation filter: {I_DILATION}')
    print(f' -> Size of blur kernel: {K_BLUR}')
    print(f' -> Number of frames for knock-down calculation: {FRAME_NUMBER}')
    logger.info(f' -> Number of iteration in Dilation filter: {I_DILATION}')
    logger.info(f' -> Size of blur kernel: {K_BLUR}')
    logger.info(f' -> Number of frames for knock-down calculation: {FRAME_NUMBER}')
    print(" - Running analysis...")
    # run analysis function
    if not REVERSE:
        frame_tkd, threshold_tkd, time_tkd = calculate_last_frame(df, FRAME_NUMBER)
    else:
        print(" - Analysis is for FIRST event, suitable for Chill Coma Recovery")
        logger.info(" - Analysis is for FIRST event, suitable for Chill Coma Recovery")
        frame_tkd, threshold_tkd, time_tkd  = calculate_first_frame(df, FRAME_NUMBER)
    # save results on knockdown analysis
    print(" - Saving results table... (sample, knockdown time (seconds))")
    # prepare dictionary
    resultados = {'Well':nombres,
                  'Frames':frame_tkd,
                  'Threshold': threshold_tkd,
                  'Minutes': time_tkd}
    # create a dataframe
    knockdown_results = pd.DataFrame(resultados, columns = ['Well', 'Frames', 'Threshold', 'Minutes'])
    # save to csv file
    knockdown_results.to_csv(os.path.join(results_directory,resultName))
    print(" -> Results table saved as CSV file in folder DIME-{}".format(VIDEO_NAME.split('.')[0]))
    
    # plot figure 
    print("- Plotting figure...")
    # define grid and plot objects
    fig = plt.figure(figsize=(20, 4*SAMPLE_SIZE))
    gs = gridspec.GridSpec(SAMPLE_SIZE, 1, hspace=0.2)
    
    for i, well in enumerate(colnames[:-2]):
        ax = fig.add_subplot(gs[i])
        ax.plot(df['minutes'], df[well])
        ax.axvline(x=time_tkd[i], color='black', linestyle=':')
        ax.axhline(y=threshold_tkd[i], color='black', linestyle='--')
        ax.set_title(well)

    # Add a suptitle to the figure
    fig.suptitle('{} - Kernel: {}, Iterations: {}, Wells: {}'.format(VIDEO_NAME, K_BLUR, I_DILATION, SAMPLE_SIZE))

    try:
        plt.savefig(os.path.join(results_directory, plotName), dpi=300)
        print(" - Figure plotted!\n")
        logger.info(" - Figure plotted!\n")
    except IOError as e:
        print(f"Cannot save the figure: {os.path.join(results_directory, plotName)}")
        logger.info(f"Cannot save the figure: {os.path.join(results_directory, plotName)}")
        print(e)       
    plt.close()
    
    print(" - Run info logged at TXT file")
    print("DIME end ------")
    logging.shutdown()
if __name__ == '__main__':
   main()