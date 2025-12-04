import math
from ultralytics.engine.results import Boxes # type: ignore

# TODO: Change these to the actual camera values
FOV = [70, 43.75] # Arducam
# fov = [59.703, 33.583] # Microsoft lifecam or other cameras with diagonal FOV of 68.5 degrees and 1280x720 resolution

def get_x_offset_deg_single(xyxy: tuple, orig_shape: tuple) -> float:
    """calculates x offset (deg) of a single bounding box

    Args:
        xyxy: bounding box coordinates (x1, y1, x2, y2)
        orig_shape: original image shape (height, width)
    """
    #source: Limelight docs(LINK HERE)
    hfov = FOV[0]*(math.pi/180)
    x1, y1, x2, y2 = xyxy

    cx = (x1+x2)/2
    nx = (cx-(orig_shape[1]/2))/(orig_shape[1]/2)
    vw = 2*math.tan((hfov/2))
    vx = vw/2*nx
    x_offset_deg = math.atan(vx/1)*(180/math.pi)

    return float(x_offset_deg)

def get_y_offset_deg_single(xyxy: tuple, orig_shape: tuple) -> float:
    """calculates y offset (deg) of a single bounding box

    Args:
        xyxy: bounding box coordinates (x1, y1, x2, y2)
        orig_shape: original image shape (height, width)
    """
    #source: Limelight docs(LINK HERE)
    vfov = FOV[1]*(math.pi/180)
    x1, y1, x2, y2 = xyxy

    cy = (y1+y2)/2
    ny = (cy-(orig_shape[0]/2))/(orig_shape[0]/2)
    vh = 2*math.tan((vfov/2))
    vy = vh/2*ny
    y_offset_deg = math.atan(vy/1)*(180/math.pi)

    return float(y_offset_deg)

def get_x_offset_deg(box: Boxes) -> float:
    """calculates x offset (deg) for the first box (legacy function)"""
    #source: Limelight docs(LINK HERE)

    if len(box[0]):
        return get_x_offset_deg_single(box.xyxy[0], box.orig_shape)

    return 0

def get_y_offset_deg(box: Boxes) -> float:
    """calculates y offset (deg) for the first box (legacy function)"""
    #source: Limelight docs(LINK HERE)

    if len(box[0]):
        return get_y_offset_deg_single(box.xyxy[0], box.orig_shape)

    return 0
 
    
# TODO: Add this if we're using it, low priority
def get_distance(box: Boxes) -> float:
    raise NotImplementedError("get_distance doesn't exist yet -- please write it!")

