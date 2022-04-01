import numpy as np
import cv2
import tellopy
import av

# Parameters
MAX_SPEED_AUTONOMOUS = 30


class TelloCV(object):
    """
    Base class to operate the drone via python
    """
    def __init__(self):
        self.speed = MAX_SPEED_AUTONOMOUS
        self.track_cmd = ""
        self.drone = None
        self.container = None

    def init_drone(self):
        """
        Connect, enable streaming and subscribe to events
        """
        self.drone = tellopy.Tello()  # initialize the drone object
        try:
            self.drone.log.set_level(0)
            self.drone.connect()
            self.drone.wait_for_connection(30.0)  # connession timeout
            self.drone.start_video()
            self.drone.subscribe(self.drone.EVENT_FLIGHT_DATA, self.flight_data_handler)
            self.drone.subscribe(self.drone.EVENT_FILE_RECEIVED, self.handle_flight_received)
            self.container = av.open(self.drone.get_video_stream())
        except Exception as exp:
            print(f"The following exception occoured: {exp}")
            self.drone.quit()
            return None
         
        return self.drone 

    def send_cmd(self, cmd):
        if cmd is not self.track_cmd:
            if cmd != "stop" and cmd != "other":
                getattr(self.drone, cmd)(self.speed)
                self.track_cmd = cmd
            else:
                self.track_cmd = ""

    def process_frame(self, frame):
        """
        Converts frame to cv2 image and to BGR
        """
        x = np.array(frame.to_image())
        x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        return x, frame

    def flight_data_handler(self, event, sender, data):
        """
        Listener to flight data from the drone.
        """
        pass

    def handle_flight_received(self, event, sender, data):
        """
        Receives images from the drone
        """
        pass
