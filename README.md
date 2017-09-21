# Vehicle Detector [![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

![car-detect](data/documentation_imgs/car-detect.gif)

![car-diagnostic-equal](data/documentation_imgs/car-diagnostic-equal.gif)

## The Project

The goals I set for this project (expanding on the basic goals set by Udacity) were to:

* Clean up an image dataset by **removing duplicate images** (or near duplicate).
* **Extract features** from that labeled training set of images to build a feature vector containing:
  * **Histogram of Oriented Gradients** (HOG) features.
  * **Color histogram** features.
  * **Spatial color** features.
* Train a **Linear SVM** classifier to identify cars vs not-cars based on .
* **Search for vehicles** using the SVM and a **sliding-window technique**.
* Estimate a bounding box for vehicles detected.

#### Video Results
You can find the full video output on [YouTube](https://www.youtube.com/watch?v=GmLn8OzekBA)
along with a [diagnostic view](https://www.youtube.com/watch?v=wKw9EWHOrDI) showing the per-frame
detections and heat map.

---

## Installation

### This Repository

Download this repository by running:

```sh
git clone https://github.com/peter-moran/vehicle-detector.git
cd vehicle-detector
```

### Software Dependencies

This project utilizes the following, easy to obtain software:

- Python 3
- OpenCV 2
- Matplotlib
- Numpy
- SciPy
- Scikit-learn
- Moviepy
- [tqdm](https://github.com/noamraph/tqdm)
- [imagehash](https://github.com/JohannesBuchner/imagehash)

An easy way to obtain these is with the [Udacity CarND-Term1-Starter-Kit](https://github.com/udacity/CarND-Term1-Starter-Kit) and Anaconda. To do so, run the following (or see the [full instructions](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/doc/configure_via_anaconda.md)):

```
git clone https://github.com/udacity/CarND-Term1-Starter-Kit.git
cd CarND-Term1-Starter-Kit
conda env create -f environment.yml
activate carnd-term1
```

And then install the rest (while still in that environment) by running:

```
pip install imagehash
```

### Data Dependencies

If you want to re-train the classifier with new feature vectors, you will need the original datasets to extract features from. They each contain over 8000 samples of 64x64 images of vehicles vs not vehicle images. These images were given by Udacity and come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the test video itself. 

* [Vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) data set.
* [Non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) data set.

If you place the top level `vehicles` and `non-vehicles` folders from these zip files in the `./data/` folder, all of the programs should be able to find them with the default arguments.

---

## Basic Usage

If all you want to do is run the pre-trained classifier on a video, all you need to do is run `find_cars.py`. There are a bunch of different options for running it, but they are all optional.

To find cars in a specific video (such as the ones in `./data/test_videos/`), run:

```
python find_cars.py -vi <input_video_path> -vo <output_video_save_path>
```

Additional options are discussed below.

## Advanced Usage

### Sample Selection

The datasets provided by Udacity have a lot of near duplicate images, something that is really easy to tell if you open the first few images in the dataset. The near-duplicates can be found consecutively, which is because the images were extracted from video, and many samples are only a short time step apart.

In order to create a dataset with a more diverse set of samples, you can use `clean_dataset.py`, which uses an image hashing algorithm to determine how unique images are compared to others and then returns a `.pkl` file containing a list of image files names for the unique images. Afterwards, it will display the first 60 samples in  the unfiltered and filtered dataset for comparison.

The level of uniqueness is set by the `MIN_HAMMING_DISTANCE` parameter. This defaults to 10, which allows good balance of allowing for a diverse dataset and multiple instances of the same objects. Increasing it would create even stricter requirement on uniqueness.

Usage can be found with the help command:

```
$ python clean_dataset.py -h
usage: clean_dataset.py [-h] [-hd MIN_HAMMING_DISTANCE] [-lm LIMIT]
                        load_path save_path

Returns an image dataset with near-duplicate images removed.

positional arguments:
  load_path             File search pattern to find images (glob style).
  save_path             Location to save the pickled list of unique files.

optional arguments:
  -h, --help            show this help message and exit
  -hd MIN_HAMMING_DISTANCE, --min_hamming_distance MIN_HAMMING_DISTANCE
                        Minimum hamming distance needed to be considered
                        unique. Increase to require greater uniqueness.
  -lm LIMIT, --limit LIMIT
                        Only run on first n files.
```

### Training

There are multiple ways to train the classifier for a particular set of samples.

**From pre-selected samples**

To train using 1000 randomly selected samples from the unique image list made by `clean_dataset.py`, run:

```
python train.py -sz 1000 -cf unique_vehicles.pkl -ncf unique_non-vehicles.pkl
```

**From pre-generated feature vectors**

When you run `train.py` it will save the feature vectors and labels to a single pickle file prior to running SVM training. After extracting these the first time, you do not need to extract them again, unless you change the characteristics of the feature vector.

To go straight to SVM training using the feature vectors, run:

```
python train.py -xylf Xy.pkl
```

**From file path**

With no parameters given, the training program will select from all the images in `./data/vehicles` and `./data/non-vehicles`. Use the `-sz` parameter to limit the number of samples selected.

**Additional Options**

```
$ python train.py -h
usage: train.py [(-cf & -ncf)? -sz | -xylf ] [additional options]

Train classifier for cars vs not-cars.

optional arguments:
  -h, --help            show this help message and exit
  -sz SAMPLE_SIZE, --sample_size SAMPLE_SIZE
                        Number of samples to use from both -cf and -ncf.
  -cf CAR_PICKLE_FILE, --car_pickle_file CAR_PICKLE_FILE
                        File path to .pkl containing a list of image files to
                        use as cars.
  -ncf NOTCAR_PICKLE_FILE, --notcar_pickle_file NOTCAR_PICKLE_FILE
                        File path to .pkl containing a list of image files to
                        use as notcars.
  -xylf XY_LOADFILE, --xy_loadfile XY_LOADFILE
                        File path to .pkl containing existing feature vectors
                        and labels as tuple (X, y).
  -ti TRAIN_ITERS, --train_iters TRAIN_ITERS
                        Number of iterations to use for the SVM parameter
                        search.
  -tj TRAIN_JOBS, --train_jobs TRAIN_JOBS
                        Number of processes to use for the SVM parameter
                        search.
  -clf CLF_SAVEFILE, --clf_savefile CLF_SAVEFILE
                        Location to save the classifier.
  -xysf XY_SAVEFILE, --xy_savefile XY_SAVEFILE
                        Location to save the extracted feature vectors and
                        labels used in training as a .pkl.
  -sr SAVE_REMAINDER, --save_remainder SAVE_REMAINDER
                        Set 'True' to also extract the unused features and
                        save them to a .pkl file.
```

### Finding cars in video

**From File**

By default `find_cars.py` will use the classifier found at `./data/trained_classifier.pkl` and feature scaler found at `./Xy_scaler.plk`. These files are included in the repository, and also happen to be the default save location of `train.py`, where these files come from.

Thus, to find cars, all you need to do is run the following, as mentioned before.

```
python find_cars.py -vi <input_video_path> -vo <output_video_save_path>
```

**Additional Options**

There are a variety of different options shown below. One particularly interesting option is `--visualization` . If you set this to `windows` you will get a diagnostic view of the tracking pipeline, which reveals per frame detections and heatmap in addition to the car bounding box selections.

```
$ python find_cars.py -h
usage: find_cars.py [ -vi & -vo | -img ]? [extra_options]

Locates cars in video and places bounding boxes around them.

optional arguments:
  -h, --help            show this help message and exit
  -vi VIDEO_IN, --video_in VIDEO_IN
                        Video to find cars in.
  -vo VIDEO_OUT, --video_out VIDEO_OUT
                        Where to save video to.
  -img IMAGES_IN, --images_in IMAGES_IN
                        Search path (glob style) to test images. Cars will be
                        found in images rather than video.
  -clf CLF_SAVEFILE, --clf_savefile CLF_SAVEFILE
                        File path to pickled trained classifier made by
                        'train.py'
  -sc SCALER_SAVEFILE, --scaler_savefile SCALER_SAVEFILE
                        File path to pickled StandardScalar made by 'train.py'
  -viz VISUALIZATION, --visualization VISUALIZATION
                        'cars' to draw bounding box around cars or 'windows'
                        to show all the detected windows.
  -st START, --start START
                        Timestamp (seconds) to start video.
  -nd END, --end END    Timestamp (seconds) to end video.
```

