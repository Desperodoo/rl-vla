import pyrealsense2 as rs
import numpy as np
import cv2

def test_realsense():
    # Configure RealSense stream
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Configure color stream
    pipeline.start(config)

    try:
        # Continuously capture and display image frames
        while True:
            frames = pipeline.wait_for_frames()  # Get frames
            color_frame = frames.get_color_frame()  # Get color frame
            if not color_frame:
                print("Failed to capture image.")
                break

            # Convert to NumPy array
            frame = np.asanyarray(color_frame.get_data())

            # Display image using OpenCV
            cv2.imshow('RealSense', frame)

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Release resources
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_realsense()
