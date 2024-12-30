import rclpy
from rclpy import qos
from rclpy.node import Node
from nbv_interfaces.srv import ViewPointSampling
from tf2_ros.transform_listener import TransformListener
from tf2_ros.buffer import Buffer
from tf2_ros import LookupException, ExtrapolationException, TransformException
import tf2_geometry_msgs
from geometry_msgs.msg import Transform, Pose
import numpy as np


class ViewPointSampler(Node):



    def __init__(self):
        super.__init__('view_point_sampler')
        self.create_service(ViewPointSampling, 'vi_to_nav/view_point_sampling', self.sampling_cb)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.map_frame_id = "map"
        self.base_footprint = "base_frontprint"
        self.max_degree = 0.785 # 45 degrees
        try:
            self.base_to_map = self.tf_buffer.lookup_transform(self.map_frame_id, self.base_footprint, rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().error(f"Failed to lookup transform from {self.base_footprint} to {self.map_frame_id}: {ex}")

    def sampling_cb(self, req, res):
        cam_info = req.cam_info
        try:
            cam_to_map = self.tf_buffer.lookup_transform(self.map_frame_id, cam_info.header.frame_id, rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().error(f"Failed to lookup transform from {cam_info.header.frame_id} to {self.map_frame_id}: {ex}")
            return res
        
        zero_pose = Pose()
        zero_pose.position.x = 0
        zero_pose.position.y = 0
        zero_pose.position.z = 0
        cam_pose_map = tf2_geometry_msgs.do_transform_pose(zero_pose, cam_to_map)
        base_pose_map = tf2_geometry_msgs.do_transform_pose(zero_pose, self.base_to_map)
        transform = Pose()
        transform.position.x = base_pose_map.position.x - cam_pose_map.position.x
        transform.position.y = base_pose_map.position.y - cam_pose_map.position.y
        transform.position.z = base_pose_map.position.z - cam_pose_map.position.z

        transform.orientation.w = base_pose_map.orientation.w - cam_pose_map.orientation.w
        transform.orientation.x = base_pose_map.orientation.x - cam_pose_map.orientation.x
        transform.orientation.y = base_pose_map.orientation.y - cam_pose_map.orientation.y
        transform.orientation.z = base_pose_map.orientation.z - cam_pose_map.orientation.z
        
        for centroid in req.centroids:
           print("Hello World") 

def quaternion_to_euler(w, x, y, z):
    # Roll (x-axis rotation)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1, 1))
    
    # Yaw (z-axis rotation)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    
    return roll, pitch, yaw




def main(args=None):
    rclpy.init(args=args)
    body_tilt_manager_node = ViewPointSampler()
    rclpy.spin(body_tilt_manager_node)
    body_tilt_manager_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
