import math

# TODO: Change these to the actual camera values
fov = [170, 163] #vertical fov calculated from: https://themetalmuncher.github.io/fov-calc/

def get_x_offset_deg(box):
    #source: Limelight docs(LINK HERE)

    if len(box):
        hfov = fov[0]*(math.pi/180)
        x1, y1, x2, y2 = box

        bbox_center_coord = [(x1+x2)/2,(y1+y2)/2]

        cx = bbox_center_coord[0]

        nx = (cx-320)/320

        vw = 2*math.tan((hfov/2))

        vx = vw/2*nx

        x_offset_deg = math.atan(vx/1)*(180/math.pi)

        return float(x_offset_deg)
    
    return 0

def get_y_offset_deg(box):
    #source: Limelight docs(LINK HERE)

    if len(box):
        vfov = fov[1]*(math.pi/180)
        x1, y1, x2, y2 = box

        bbox_center_coord = [(x1+x2)/2,(y1+y2)/2]

        cy = bbox_center_coord[1]

        ny = (cy-320)/320

        vh = 2*math.tan((vfov/2))

        vy = vh/2*ny

        y_offset_deg = math.atan(vy/1)*(180/math.pi)

        return float(y_offset_deg)
    
    return 0
 
    
def get_distance(box):
    return 3 # TODO: Add this

def get_pixel_width(box):
    x1, y1, x2, y2 = box
    width = x2-x1
    return abs(width)
        