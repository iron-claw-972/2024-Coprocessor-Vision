import ntcore

nt_inst = ntcore.NetworkTableInstance.getDefault()


table = nt_inst.getTable("object_detection")
distance_topic = table.getDoubleTopic("distance").publish()
x_angle_offset_topic = table.getDoubleTopic("tx").publish()
y_angle_offset_topic = table.getDoubleTopic("ty").publish()

nt_inst.startClient4("Coprocessor")
nt_inst.setServerTeam(972)

def publish_distance(value):
    distance_topic.set(value)

def publish_x_angle_offset(value):
    x_angle_offset_topic.set(value)
    
def publish_y_angle_offset(value):
    y_angle_offset_topic.set(value)
    

