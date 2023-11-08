import ntcore

nt_inst = ntcore.NetworkTableInstance.getDefault()


table = nt_inst.getTable("object_detection")
distance_topic = table.getDoubleTopic("distance").publish()
x_angle_offset_topic = table.getDoubleTopic("distance").publish()
y_angle_offset_topic = table.getDoubleTopic("distance").publish()


def publish_distance(value):
    distance_topic.set(value)

def publish_x_angle_offset(value):
    x_angle_offset_topic.set(value)
    
def publish_y_angle_offset(value):
    y_angle_offset_topic.set(value)
    

