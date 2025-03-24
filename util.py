import math
from ultralytics.engine.results import Boxes # type: ignore

# TODO: Change these to the actual camera values
# fov = [70, 43.75] # Arducam
fov = [59.703, 33.583] # Microsoft lifecam or other cameras with diagonal FOV of 68.5 degrees and 1280x720 resolution

def get_x_offset_deg(box: Boxes) -> float:
    #source: Limelight docs(LINK HERE)

    if len(box[0]):
        hfov = fov[0]*(math.pi/180)
        x1, y1, x2, y2 = box.xyxy[0]

        bbox_center_coord = [(x1+x2)/2,(y1+y2)/2]

        cx = bbox_center_coord[0]

        nx = (cx-320)/320

        vw = 2*math.tan((hfov/2))

        vx = vw/2*nx

        x_offset_deg = math.atan(vx/1)*(180/math.pi)

        return float(x_offset_deg)
    
    return 0

def get_y_offset_deg(box: Boxes) -> float:
    #source: Limelight docs(LINK HERE)

    if len(box[0]):
        vfov = fov[1]*(math.pi/180)
        x1, y1, x2, y2 = box.xyxy[0]

        bbox_center_coord = [(x1+x2)/2,(y1+y2)/2]

        cy = bbox_center_coord[1]

        ny = (cy-320)/320

        vh = 2*math.tan((vfov/2))

        vy = vh/2*ny

        y_offset_deg = math.atan(vy/1)*(180/math.pi)

        return float(y_offset_deg)
    
    return 0
 
    
def get_distance(box: Boxes) -> float:
    if len(box[0]) == 0:
        return 0.0
    
    x1, y1, x2, y2 = box.xyxy[0]
    ground_point_y = y2  # Bottom of bounding box
    
    # TODO need to be changed to the actual camera values
    CAMERA_HEIGHT = 0.5  
    CAMERA_TILT = 30.0  # camera tilt downwards (degs)
    IMAGE_HEIGHT = 640 #px
    
    # to radians
    camera_tilt_rad = CAMERA_TILT * (math.pi/180)
    vfov_rad = fov[1] * (math.pi/180)
    
    normalized_y = (ground_point_y - (IMAGE_HEIGHT/2)) / (IMAGE_HEIGHT/2)
    angle_in_fov = normalized_y * (vfov_rad/2)
    
    # Calculate the actual angle from camera to ground point
    ground_angle = camera_tilt_rad - angle_in_fov

    distance = CAMERA_HEIGHT / math.tan(ground_angle)
    
    return float(distance)

