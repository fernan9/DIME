# DIME (Detector of Insect Motion Endpoint)
## General description
DIME is a command-line application for scoring thermal limit assays from video records.

## Before starting (Windows)
DIME will require the following dependencies:

- Python 3.0+
	+ Install Python interpreter
		* Details in https://www.python.org/downloads/
		* Required libraries
			- numpy==1.20.3
			- opencv-python==4.5.3.56
			- matplotlib==3.4.2
			- pandas==1.3.1
- OpenCV
	+ Install the PIP package-management system
	+ Install full package with:
		* `pip install opencv-contrib-python`
		* Details in https://pypi.org/project/opencv-python/

## Working directory

Your working directory must contain at least:
- A copy of `DIME-cl.py`
- The video you want to analyze

In consequence, DIME will create a new directory named as `'DIME-[name of your video]'` containing your results. The following files populate the new directory after running DIME:
+ a `DataFrame` file containing the raw motion data
+ a `Results` file providing the endpoint estimates per ROI defined
+ a `Log` file containing details of the run
+ an `Activity` PNG file with a plot of rPIC by time for each individual
+ an image of the first frame of the video with each ROI overlaid
+ a Numpy (NPY) file containing ROI coordinates data

## Running DIME

DIME is a command-line python application, it will transform the motion activity in your bioassay video to a numerical variable of relative pixel intensity change (rPIC). Then, each ROI motion data will be scored for an activity endpoint according to the default method IM (Individual Median), but alternate methods can be applied to the raw `DataFrame` data with the accompanying `R` functions and scripts provided (see below for details).

DIME can run in Terminal or PowerShell applications by calling a Python interpreter followed by the program file `dime_1_1.py` and a listing of its configuration arguments. For example, we can request the `--help` information by typing:

```
...\DIME> python .\dime_1_1.py --help
```

Some arguments are mandatory, but some have default values. The possible commands that can be used are:

+ -h, --help
	* show this help message and exit
+ --videoName <string>, -v <string>
	* Name of the video file to analyze
+ --sampleSize <int>, -s <int>
	* Number of wells in the plate
	* default: 96
+ --iterationNumber <int>, -i <int>
	* Number of iterations for the dilation filter
	* default: 2
+ --kernelSize <int>, -k <int>
	* Size of the kernel for the Gaussian blur
	* format: Kernel = 2 * `kernelSize` + 1
	* default: 7 = 2 ( *3* ) + 1
+ --reverseScoring, -r
	* If set, reverse detection of the first event will be calculated
	* default: False
+ --filterNoise <flt>, -f <flt>
	* Apply percent ratio filter based on average activity in total frame
	* (default:None)
+ --numberFrames <int>, -n <int>
	* Number of frames to sample for knockdown analysis
	* default: 1

### Standard Run

To score a last activity endpoint in a video `TestVideo.mp4` with 30 fruit flies (-s) with a small kernel of 3X3 pixels (k = 1, kernel = 2(1) + 1), in the folder DIME, we would run:

```
...\DIME> python dime_1_1.py -v .\TestVideo.mp4 -s 30 -k 1
```

Instead, to score a first activity endpoint (e.g. chill coma recovery) in the same video, we would run:

```
...\DIME> python dime_1_1.py -v .\TestVideo.mp4 -s 30 -k 1 -r
```

or

```
...\DIME> python dime_1_1.py -v .\TestVideo.mp4 -s 30 -k 1 --reverseScoring
```

The progress of the analysis will be displayed in the Terminal window.

## Determining ROI per observation


The area of the video where each individual insect performs movemenet, aka the region-of-interest (ROI), must be determined during the analysis. Once set for a `'videoName + sampleSize'` combination, the file will be automatically read in the following runs in the same video-by-size configuration. You must make a `click-hold-slide-release` movement with the mouse from the upper-left to the lower-right corner of the ROI area. You can reset the current ROI by typing 'r' for reset and then typing 'c' to confirm once the desired shape is achieved.

## Filter functionality: low-pass

A low-pass filter can be applied to the activity data to reduce the spikes of activity levels that have a technical origin, for example a camera shake caused by opening the incubator door. To apply the low-pass filter you must add the `-filterNoise` or `-f` argument to the run and specify the threshold with a number with decimal fraction (float).

1. Run DIME in the video with standard configuration.
2. Adjust `i` and `k` for the desired sensibility.	
3. Identify the `Total_trace` file in the Results folder.
4. Determine the rPIC threshold to filter. Use `Total_trace` to identify extreme peaks in the data along time and use `Total_histogram` to assess the data-set structure.
5. Re-run the program with the argument `-filterNoise` or `-f` followed by the desired threshold from step 4.
6. Compare activity trace and histograms tables between filtered and non-filtered results.

## Notes

1. The frequency of the motion detection will depend on the frame rate of your video, which is automatically detected by DIME and reported in the Log file.
2. The modification of the transformation setting (k and i argument) will depend on your specific set-up. They must be manually calibrated by running a test video with increasing values and determine the sensibility required. These values may change between species and between illumination configurations.
3. To run multiple videos in batch, run the following command in PowerShell (Windows), type:
``` 
# Get all mp4 files in the current directory
$files = Get-ChildItem -Path . -Filter *.mp4

# Loop through each file and execute the Python script
foreach ($file in $files) {
    & python dime_1_1.py -v $file.FullName -s 30
}
```
