from pathlib import Path
from logzero import logger, logfile
from picamera import PiCamera
from orbit import ISS
from time import sleep
from datetime import datetime, timedelta
from exif import Image
import cv2
import math

def convert(angle):
    sign, degrees, minutes, seconds = angle.signed_dms()
    exif_angle = f'{degrees:.0f}/1,{minutes:.0f}/1,{seconds*10:.0f}/10'
    return sign < 0, exif_angle

# This function takes a photo
def capture(camera, image):
    location = ISS.coordinates() #Current location of the ISS
    south, exif_latitude = convert(location.latitude)
    west, exif_longitude = convert(location.longitude)

    camera.exif_tags['GPS.GPSLatitude'] = exif_latitude
    camera.exif_tags['GPS.GPSLatitudeRef'] = "S" if south else "N"
    camera.exif_tags['GPS.GPSLongitude'] = exif_longitude
    camera.exif_tags['GPS.GPSLongitudeRef'] = "W" if west else "E"

    camera.capture(image) #Capture of the photo

# This function returns the exact time instant at that moment
def get_time(image):
    with open(image, 'rb') as image_file:
        img = Image(image_file)
        time_str = img.get("datetime_original")
        time = datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
    return time

# This function returns the difference between the times at which the photos were taken
def get_time_difference(image_1, image_2):
    time_1 = get_time(image_1)
    time_2 = get_time(image_2)
    time_difference = time_2 - time_1
    return time_difference.seconds

# This function convert the imaghes caugt in cv objects
def convert_to_cv(image_1, image_2):
    image_1_cv = cv2.imread(image_1, 0)
    image_2_cv = cv2.imread(image_2, 0)
    return image_1_cv, image_2_cv

# This function calculates the keypoins of the images
def calculate_features(image_1_cv, image_2_cv, feature_number):
    orb = cv2.ORB_create(nfeatures=feature_number)
    keypoints_1, descriptors_1 = orb.detectAndCompute(image_1_cv, None)
    keypoints_2, descriptors_2 = orb.detectAndCompute(image_2_cv, None)
    return keypoints_1, keypoints_2, descriptors_1, descriptors_2

# This function finds the matches between the keypoints of two images
def calculate_matches(descriptors_1, descriptors_2):
    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = brute_force.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def find_matching_coordinates(keypoints_1, keypoints_2, matches):
    coordinates_1 = []
    coordinates_2 = []
    for match in matches:
        image_1_idx = match.queryIdx
        image_2_idx = match.trainIdx
        (x1, y1) = keypoints_1[image_1_idx].pt
        (x2, y2) = keypoints_2[image_2_idx].pt
        coordinates_1.append((x1, y1))
        coordinates_2.append((x2, y2))
    return coordinates_1, coordinates_2

# This function calculates the average distance between the keypoints of the two images
def calculate_mean_distance(coordinates_1, coordinates_2):
    all_distances = 0
    merged_coordinates = list(zip(coordinates_1, coordinates_2))
    for coordinate in merged_coordinates:
        x_difference = coordinate[0][0] - coordinate[1][0]
        y_difference = coordinate[0][1] - coordinate[1][1]
        distance = math.hypot(x_difference, y_difference)
        all_distances += distance
    return all_distances / len(merged_coordinates)

# This function calculates the speed between two images
def calculate_speed_in_kmps(feature_distance, GSD, time_difference):
    distance = feature_distance * GSD / 100000
    speed = distance / time_difference
    return speed

# Creation of the file 'events' for error report
base_folder = Path(__file__).parent.resolve()
logfile(base_folder / "events.log")

# Initialization of the camera
cam = PiCamera()
cam.resolution = (4056, 3040)

# Initialization of the counter and call to the 'actual time' function
counter = 1
start_time = datetime.now()
now_time = datetime.now()
array_speed=[]
#array_speed1=[]

try: # The following code is written inside this try block to prevent the program from crashing if it encounters an error
    while now_time < start_time + timedelta(seconds= 540) and counter <= 30: #While loop repeats itself until exactly 9 minutes have passed and the image counter has reached 30 
    
        # Capture of the images
            image_file = f"{base_folder}/photo_{counter:03d}.jpg" # Path where the image get saved
            capture(cam, image_file) # Call of the capture function to take the photo
            logger.info(f"iteration photo {counter}")
            sleep(1) # Sleep time between an iteration and the next one
            now_time = datetime.now() # Update of the current time
            
            
            # Analisis of the images (only if there are at least 2)
            if counter >= 2:  
                previous_image = f"{base_folder}/photo_{counter - 1:03d}.jpg"
                current_image = f"{base_folder}/photo_{counter:03d}.jpg"

                # Analisis of the images in couple by calling the following functions
                time_difference = get_time_difference(previous_image, current_image)
                image_1_cv, image_2_cv = convert_to_cv(previous_image, current_image)
                keypoints_1, keypoints_2, descriptors_1, descriptors_2 = calculate_features(image_1_cv, image_2_cv, 1000)
                matches = calculate_matches(descriptors_1, descriptors_2)
                coordinates_1, coordinates_2 = find_matching_coordinates(keypoints_1, keypoints_2, matches)
                average_feature_distance = calculate_mean_distance(coordinates_1, coordinates_2)
                speed = calculate_speed_in_kmps(average_feature_distance, 12648, time_difference)
                array_speed.append(speed) # This array contains all of the speed measured during the entire duration of the program
            counter += 1 # Incrementation of the counter

                    
    sum_speed=0 # Variable that contains the sum of only the speed measured that we consider valid
    sum_speedTot=0 # Variable that contains the sum of all the speed measured
    valid=0 # Counter of the valid speeds
    counter-=1 # Decrementation of the counter
    
    for i in range(0, counter-1): # Calculation of the average speed
        if array_speed[i] > 6 and array_speed[i] < 8: # Validity condition of the speeds obtained
            sum_speed+=array_speed[i] # The speed gets added to the valid speeds sum only if it meets the requirements
            sum_speedTot+=array_speed[i] # The speed gets added to the total speeds sum in both cases
            valid= valid+1 # Incrementation of the counter
        else:
            sum_speedTot+=array_speed[i]
        if sum_speed > 0:
            speed= sum_speed/valid # If there are valid speeds, we calculate the average speed between them
        else:
            speed=sum_speedTot/(counter-1) # If there aren't valid speeds, we calculate the average speed between everyone of them

except Exception as e:
    print(f'{e.__class__.__name__}: {e}')

# Saving the velocity on the 'result.txt' file
base_folder = Path(__file__).parent.resolve()

data_file = base_folder / "result.txt" # Saving path of the 'result.txt' file

with open(data_file, "w", buffering=1) as f:
    if (speed/10>=1):
        f.write(f"{speed:5.3f} km/s")
    else:
        f.write(f"{speed:5.4f} km/s")
cam.close() # Closing of the camera    
f.close() # Closing of the file