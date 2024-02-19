import math
import numpy as np
# TODO: Change these to the actual camera values
hfov = 139.28 
vfov = 114

def get_x_offset_deg(box):
    #source: Limelight docs(LINK HERE)

    if len(box[0]):
        hfov_rad = hfov*(math.pi/180)

        x1, y1, x2, y2 = box.xyxy[0]
        bbox_center_coord = np.array([(x1+x2)/2,(y1+y2)/2],np.float32)
        cx = bbox_center_coord[0]

        nx = (1/319.5) * (cx - 319.5)
        
        vpw = 2.0*math.tan(hfov_rad/2)

        x = vpw/2 * nx

        x_offset_deg = math.atan(x)

        return float(x_offset_deg * (180/math.pi))
    
    return float(0)

def get_y_offset_deg(box):
    #source: Limelight docs(LINK HERE)

    if len(box[0]):
        vfov_rad = vfov*(math.pi/180)

        x1, y1, x2, y2 = box.xyxy[0]
        bbox_center_coord = np.array([(x1+x2)/2,(y1+y2)/2],np.float32)
        cy = bbox_center_coord[1]

        ny = (1/239.5) * (cy - 239.5)
        
        vph = 2.0*math.tan(vfov_rad/2)

        y = vph/2 * ny

        y_offset_deg = math.atan(y)

        return float(y_offset_deg * (180/math.pi))
    
    return float(0)   
 
def print_offsets(results):
    for result in results:
        box = result.boxes
        if not box:
            continue

        x1, y1, x2, y2 = box.xyxy[0]

        x_offset = get_x_offset_deg(box)
        y_offset = get_y_offset_deg(box)

        print(f"x: {x_offset}")
        print(f"y: {y_offset}")

# TODO: Add tis if we're using it, low pirority
def get_distance(box):
    return 1
