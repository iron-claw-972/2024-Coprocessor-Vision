import ntcore

nt_inst = ntcore.NetworkTableInstance.getDefault()


table = nt_inst.getTable("object_detection")
distance_topic = table.getDoubleTopic("distance").publish()


def publish_distance(value):
    distance_topic.set(value)


