#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml

# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)


# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

    objects_label = []
    objs = []
    # convert the ros message to pcl data
    cloud_filtered = ros_to_pcl(pcl_msg)

    # Statistical Outlier Filtering
    outlier_filter = cloud_filtered.make_statistical_outlier_filter()
    outlier_filter.set_mean_k(15)
    x = 0.001
    outlier_filter.set_std_dev_mul_thresh(x)
    cloud_filtered = outlier_filter.filter()

    # VoxelGrid Downsampling 
    vox = cloud_filtered.make_voxel_grid_filter()
    LEAF_SIZE = 0.006
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    cloud_filtered = vox.filter()

    # Pass Through Filter
    passthrough_z = cloud_filtered.make_passthrough_filter()
	filter_axis = 'z'
	passthrough_z.set_filter_field_name(filter_axis)
    axis_min = 0.6
    axis_max = 1.1
    passthrough_z.set_filter_limits(axis_min, axis_max)
    cloud_filtered = passthrough_z.filter()
    passthrough_y = cloud_filtered.make_passthrough_filter()
	filter_axis = 'y'
	passthrough_y.set_filter_field_name(filter_axis)
    axis_min = -0.48
    axis_max = 0.48
    passthrough_y.set_filter_limits(axis_min, axis_max)
    cloud_filtered = passthrough_y.filter()

    # RANSAC algorithm
    seg = cloud_filtered.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    max_distance = 0.01
    seg.set_distance_threshold(max_distance)
    inliers, coefficients = seg.segment()

    # Extract inliers and outliers
    objects = cloud_filtered.extract(inliers, negative=True)	
    table = cloud_filtered.extract(inliers, negative=False)


    # Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(objects)
    tree = white_cloud.make_kdtree()
    ec = white_cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(0.01)
    ec.set_MinClusterSize(1)
    ec.set_MaxClusterSize(10000)
    ec.set_SearchMethod(tree)
    cluster_indices = ec.Extract()

    # Create Cluster-Mask Point Cloud to visualize each cluster separately
    cluster_color = get_color_list(len(cluster_indices))
    color_cluster_point_list = []
    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                            white_cloud[indice][1],
                                            white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    # pcl data to ros
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)
    ros_cloud_objects = pcl_to_ros(objects)
    ros_cloud_table = pcl_to_ros(table)

    # Publish the ROS messages
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)
    pcl_cluster_pub.publish(ros_cluster_cloud)
    
    # loop through each clusters
    for idx, points in enumerate(cluster_indices):

        pcl_cluster = objects.extract(points)
        ros_cluster = pcl_to_ros(pcl_cluster)

        # Compute histogram features
        color_hists = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        normal_hists = compute_normal_histograms(normals)
        feature = np.concatenate((color_hists, normal_hists))

        # Make the prediction, retrieve the label for the result
        # and add it to objects_label list
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        objects_label.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[points[0]])
        label_pos[2] += .2
        object_markers_pub.publish(make_label(label,label_pos, idx))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        objs.append(do)

    rospy.loginfo('Detected {} objects: {}'.format(len(objects_label), objects_label))

    # Publish Ros Messages
    objs_pub.publish(objs)

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        pr2_mover(objs)

    except rospy.ROSInterruptException:
        pass


# function to load parameters and request PickPlace service
def pr2_mover(object_list):

   #  Initialize variables
    test_scene_num = Int32()
    test_scene_num = 1
    which_arm = String()
    object_name = String()
    object_name = 'soap'
    pick_pose = Pose()
    place_pose = Pose()
    labels = []
    dict_list = []
    centroids = []

    #  Get/Read parameters
    object_list_param = rospy.get_param('/object_list')
    dropbox_param = rospy.get_param('/dropbox')
    #  Parse parameters into individual variables
    for object_iterator in object_list:
        labels.append(object_iterator.label)
        points = ros_to_pcl(object_iterator.cloud).to_array()
        centroids.append(np.mean(points, axis=0)[:3])

    #  Loop through the pick list
    for object_iterator in object_list_param:
        object_name.data = object_iterator['name']
        object_group = object_iterator['group']
	if object_group == 'green':
		which_arm.data = 'right'
	else :
		which_arm.data = 'left'

    #  Create 'place_pose' for the object
	current_object = -1
	for i in range(len(labels)) :
		if object_name == labels[i]:
			current_object = i
			break
        if not current_object == -1:
		continue
	
     	target_pose = []
     	for i in range(len(dropbox_param)):
        	if dropbox_param[i]['name'] == which_arm.data:
			target_pose = dropbox_param[i]['position']
			break

        #  Assign the arm to be used for pick_place
	    place_pose.position.x = target_pose[0]
	    place_pose.position.y = target_pose[1]
	    place_pose.position.z = target_pose[2]
        pick_pose.position.x = float(centroids[current_object][0])
        pick_pose.position.y = float(centroids[current_object][1])
        pick_pose.position.z = float(centroids[current_object][2])

        #  Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
	    dict_list.append(make_yaml_dict(test_scene_num, arm, object_name, pick_pose, place_pose))
        # Wait for 'pick_place_routine' service to come up
        rospy.wait_for_service('pick_place_routine')

        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)
            #  Insert your message variables to be sent as a service request
            resp = pick_place_routine(test_scene_num, object_name, which_arm, pick_pose, place_pose)
	        print (test_scene_num, object_name, which_arm, pick_pose, place_pose)
            print ("Response: ",resp.success)

        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    # Output your request parameters into output yaml file
    send_to_yaml('output_'+str(test_scene_num.data)+'.yaml', dict_list)
    return

if __name__ == '__main__':

    # ROS node initialization
    rospy.init_node('clustering', anonymous=True)

    # Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # Create Publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)
    object_markers_pub   = rospy.Publisher("/object_markers", Marker, queue_size=1)
    objs_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)

    # Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
