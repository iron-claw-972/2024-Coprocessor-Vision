import math
from ultralytics.engine.results import Boxes # type: ignore

# TODO: Change these to the actual camera values
fov = [70, 43.75] # Arducam
RESOLUTION = [720, 1280]
# fov = [59.703, 33.583] # Microsoft lifecam or other cameras with diagonal FOV of 68.5 degrees and 1280x720 resolution

def get_fovs(box: Boxes) -> list[float]:
    x1, y1, x2, y2 = box.xyxy[0]
    center: list[int] = [(x1+x2)/2,(y1+y2)/2]
    zero_centered: list[int] = [center[0] - (RESOLUTION[0]/2), center[1] - (RESOLUTION[1]/2)]
    viewport: list[int] = [zero_centered[0], zero_centered[1], 1]
    fovs: list[int] = [math.tan(viewport[0]) * (fov[0]/2), math.tan(viewport[1]) * (fov[1]/2)]
    return fovs

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
 
    
# TODO: Add tis if we're using it, low pirority
def get_distance(box: Boxes) -> float:
    raise NotImplementedError("get_distance doesn't exist yet -- please write it!")

