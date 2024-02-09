import math

# TODO: Change these to the actual camera values
fov = [139.28, 114]
hres = 1280
vres = 720

def get_x_offset_deg(box):
    #source: Limelight docs(LINK HERE)

    if len(box[0]):
        hfov = fov[0]*(math.pi/180)
        x1, y1, x2, y2 = box.xyxy[0]

        bbox_center_coord = [(x1+x2)/2,(y1+y2)/2]

        cx = bbox_center_coord[0]

        nx = (cx-hres/2-0.5) * (hres/2)

        vw = 2*math.tan((hfov/2))

        vx = vw/2*nx

        x_offset_deg = math.atan(vx/1)*(180/math.pi)

        return float(x_offset_deg)
    
    return float(0)

def get_y_offset_deg(box):
    #source: Limelight docs(LINK HERE)

    if len(box[0]):
        vfov = fov[1]*(math.pi/180)
        x1, y1, x2, y2 = box.xyxy[0]

        bbox_center_coord = [(x1+x2)/2,(y1+y2)/2]

        cy = bbox_center_coord[1]

        ny = (cy-vres/2-0.5) * (vres/2)

        vh = 2*math.tan((vfov/2))

        vy = vh/2*ny

        y_offset_deg = math.atan(vy/1)*(180/math.pi)

        return float(y_offset_deg)
    
    return float(0)
 
    
# TODO: Add tis if we're using it, low pirority
def get_distance(box):
    return 1
