import cv2
import mediapipe as mp
import imutils
import numpy as np
import pyaudio
import audioop
import threading

from enum import Enum

from collections import deque

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

ERODE_KERNEL_SIZE = 10

def max_rgb_filter(image):
	# split the image into its BGR components
	(B, G, R) = cv2.split(image)
	# find the maximum pixel intensity values for each
	# (x, y)-coordinate,, then set all pixel values less
	# than M to zero
	M = np.maximum(np.maximum(R, G), B)
	R[R < M] = 0
	G[G < M] = 0
	B[B < M] = 0
	# merge the channels back together and return the image
	return cv2.merge([B, G, R])


# Processing the input image
def process_image(img):
    # Converting the input to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(gray_image)
    # Returning the detected hands to calling function
    return results

def ease_in_out(t):
    return t * t * (3 - 2 * t)

def get_audio_volume():
    global audio_volume
    global volume_lock

    chunk_size = 1024
    sample_format = pyaudio.paInt16
    channels = 1 # 1 channel, index 0 for mic. 2 channels, index 1 for system audio
    rate = 48000

    p = pyaudio.PyAudio()

    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
          print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=rate,
                    input=True,
                    input_device_index=0,
                    frames_per_buffer=chunk_size)
    while True:
        data = stream.read(chunk_size)
        rms = audioop.rms(data, 2)

        # Normalize the RMS value to a scale between 0 and 1
        volume = rms / 32768.0

        with volume_lock:
            audio_volume = volume
    

    stream.stop_stream()
    stream.close()
    p.terminate()

# Drawing landmark connections
def draw_hand_connections(img, results):
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape

                # Finding the coordinates of each landmark
                cx, cy = int(lm.x * w), int(lm.y * h)

                # Printing each landmark ID and coordinates
                # on the terminal
                # print(id, cx, cy)

                # Creating a circle around each landmark
                cv2.circle(img, (cx, cy), 10, (255, 0, 0),
                           cv2.FILLED)
                # Drawing the landmark connections
                mpDraw.draw_landmarks(img, handLms,
                                      mpHands.HAND_CONNECTIONS)

        return img
    
class DistortMode(Enum):
    NONE = 0
    MAX_RGB = 1
    BLEND = 2
    FIXED_BLEND = 3


# Global variable to share audio volume across threads
audio_volume = 0.0
volume_lock = threading.Lock()  # Lock for thread safety

GHOST_LENGTH = 50
GHOST_ALPHA = 0.8

AUDIO_SCALE = 25 # for distort blend mode

ghost_frames = deque(maxlen=GHOST_LENGTH)  # Buffer to store previous frames for ghosting

def main():
   # Replace 0 with the video path to use a
   # gpre-recorded video
    cap = cv2.VideoCapture(0)

    distortion = DistortMode.NONE
    ghosting = False
    sketch = False

    audio_thread = threading.Thread(target=get_audio_volume)
    audio_thread.daemon = True  # Daemonize the thread
    audio_thread.start()

    while True:
        # Taking the input
        success, image = cap.read()
        image = imutils.resize(image, width=500, height=500)

        results = process_image(image)
        draw_hand_connections(image, results)

        kernel = np.ones((ERODE_KERNEL_SIZE, ERODE_KERNEL_SIZE), np.uint8)
        image = cv2.erode(image, kernel, iterations=1)
        kernel = np.array([[-1, -1, -1], [-1, 9.5, -1], [-1, -1, -1]])
        image = cv2.filter2D(image, -1, kernel) # sharpens
        image_distort = max_rgb_filter(image)
        image_distort = cv2.cvtColor(image_distort, cv2.COLOR_BGR2GRAY)
        image_distort = cv2.cvtColor(image_distort, cv2.COLOR_GRAY2BGR)

        if distortion == DistortMode.MAX_RGB:
            image = image_distort
        
        elif distortion == DistortMode.BLEND:
            # lerp of image and image_distort, where alpha is determined by the current computer audio output volume
            global audio_volume
            with volume_lock:
                # print(audio_volume)
                eased_volume = ease_in_out(audio_volume * AUDIO_SCALE)
                vol = audio_volume * AUDIO_SCALE
                # print(vol)
                # Linear interpolation between the two images based on audio volume
                image = cv2.addWeighted(image, 1 - vol, image_distort, vol, 0)

        elif distortion == DistortMode.FIXED_BLEND:
            FIXED_ALPHA = 0.3
            image = cv2.addWeighted(image, 1 - FIXED_ALPHA, image_distort, FIXED_ALPHA, 0)

        

        if ghosting:

            ghost_frames.appendleft(image.copy())
            ghosted_image = np.zeros_like(image, dtype=np.uint8)
            # print(len(ghost_frames))
            for idx, ghost_frame in enumerate(ghost_frames):
                alpha = GHOST_ALPHA / (idx + 1)  # Decreasing weight for ghost frames
                ghosted_image = cv2.addWeighted(ghosted_image, 1 - alpha, ghost_frame, alpha, 0)
            image = cv2.addWeighted(image, 1 - GHOST_ALPHA, ghosted_image, GHOST_ALPHA, 0)


        if sketch:
            SKETCH_ALPHA = 0.5
            image_gray, sketch_image = cv2.pencilSketch(image, sigma_s = 60, sigma_r = 0.07, shade_factor = 0.1)
            image = cv2.addWeighted(image, 1 - SKETCH_ALPHA, sketch_image, SKETCH_ALPHA, 0)

         # Displaying the output
        cv2.imshow("Hand tracker", image)

        key = cv2.waitKey(1)

        # toggle distortion
        if key == ord('d'):
            distortion = DistortMode((distortion.value + 1) % len(DistortMode))
            print(f"now set to: {distortion.name}")

        elif key == ord('g'):
          # TOGGLE GHOSTING
          ghosting = not ghosting
          print(f"ghosting set to : {ghosting}")
          ghost_frames.clear()

        elif key == ord('p'):
          # TOGGLE PENCIL SKETCH
          sketch = not sketch
          print(f"sketch set to : {sketch}")


        # Program terminates when q key is pressed
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()