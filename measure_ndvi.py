import time
import numpy as np

import cv2
import picamera
import picamera.array


def disp_single(im1=None):
    """
    Display an image.

    """
    height, width = im1.shape

    combined = np.zeros((height, width, 3), dtype=np.uint8)

    combined[0:height, 0:width, :] = cv2.cvtColor(im1, cv2.COLOR_GRAY2RGB)

    return combined


def label(image, text):
    """
    Labels the given image with the given text
    """
    return cv2.putText(image, text, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 4)


def contrast_stretch(im):
    """
    Performs a simple contrast stretch of the given image, from 5-95%.
    """
    in_min = np.percentile(im, 5)
    in_max = np.percentile(im, 95)

    out_min = 0.0
    out_max = 255.0

    out = im - in_min
    out *= ((out_min - out_max) / (in_min - in_max))
    out += in_min

    return out


def run():
    with picamera.PiCamera() as camera:
        # Set the camera resolution
        x = 400
        camera.resolution = (int(1.33 * x), x)
        # Various optional camera settings below:
        # camera.framerate = 5
        # camera.awb_mode = 'off'
        # camera.awb_gains = (0.5, 0.5)

        # Need to sleep to give the camera time to get set up properly
        time.sleep(1)

        with picamera.array.PiRGBArray(camera) as stream:
            # Loop constantly
            while True:
                # Grab data from the camera, in colour format
                # NOTE: This comes in BGR rather than RGB, which is important
                # for later!
                camera.capture(stream, format='bgr', use_video_port=True)
                image = stream.array

                # Get the individual colour components of the image
                b, g, r = cv2.split(image) # blue, green, near-IR

                # Calculate the NDVI

                # Bottom of fraction
                bottom = (r.astype(float) + b.astype(float))
                bottom[bottom == 0] = 0.01  # Make sure we don't divide by zero!

                ndvi_raw = (r.astype(float) - b.astype(float)) / bottom
                
                ndvi = contrast_stretch(ndvi_raw)
                ndvi = ndvi.astype(np.uint8)
                
                # Get average
                mean_ndvi = np.mean(ndvi_raw)

                # Do the labelling
                label(ndvi, 'NDVI: '+str(round(mean_ndvi,3)))

                # Process for display
                combined = disp_single(ndvi)

                # Display
                cv2.imshow('image', combined)

                stream.truncate(0)

                # If we press ESC then break out of the loop
                c = cv2.waitKey(7) % 0x100
                if c == 27:
                    break

    # Important cleanup here!
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run()
