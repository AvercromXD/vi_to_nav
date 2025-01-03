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
import math


class ViewPointSampler(Node):



    def __init__(self):
        super().__init__('view_point_sampler')
        self.create_service(ViewPointSampling, 'vi_to_nav/view_point_sampling', self.sampling_cb)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.map_frame_id = "map"
        self.base_footprint = "base_footprint"
        self.max_degree = 0.785 # 45 degrees
        self.num_poses = 10
        self.num_d = 3
        self.d_variation = 0.3

    def sampling_cb(self, req, res):
        cam_info = req.cam_info
        try:
            self.base_to_map = self.tf_buffer.lookup_transform(self.map_frame_id, self.base_footprint, rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().error(f"Failed to lookup transform from {self.base_footprint} to {self.map_frame_id}: {ex}")
            return res
        try:
            cam_to_map = self.tf_buffer.lookup_transform(self.map_frame_id, cam_info.header.frame_id, rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().error(f"Failed to lookup transform from {cam_info.header.frame_id} to {self.map_frame_id}: {ex}")
            return res
        
        zero_pose = Pose()
        zero_pose.position.x = 0.0
        zero_pose.position.y = 0.0
        zero_pose.position.z = 0.0
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
        roll,pitch,yaw = quaternion_to_euler(base_pose_map.orientation.w, base_pose_map.orientation.x, base_pose_map.orientation.y, base_pose_map.orientation.z)
        rpy = np.array([roll, pitch, yaw])

        for centroid in req.centroids:
            dir = np.array([centroid.x, centroid.y, centroid.z]) - np.array([base_pose_map.position.x, base_pose_map.position.y, base_pose_map.position.z]) 
            d = np.linalg.norm(np.array([centroid.x, centroid.y, centroid.z]) - np.array([base_pose_map.position.x, base_pose_map.position.y, base_pose_map.position.z]))

            for i in range(-(int) (self.num_d / 2), (int) (self.num_d/2 + 1)):
                if i == 0:
                    d_var = 0
                else:
                    d_var = d * (self.d_variation / i)
                new_d = d + d_var
                
                for j in range(self.num_poses):
                    start = dir / d - d_var # only move x and y not z! calculate start differently
                    vec = start - np.array([centroid.x , centroid.y , centroid.z])
                    theta = self.max_degree * (j + 1) / self.num_poses
                    rot1 = np.array([[np.cos(theta), -np.sin(theta), 0],
                                     [np.sin(theta), np.cos(theta), 0], 
                                     [0 , 0, 1]])

                    rot2 = np.array([[np.cos(-theta), -np.sin(-theta), 0],
                                     [np.sin(-theta), np.cos(-theta), 0], 
                                     [0 , 0, 1]])
                    p1 = np.array([centroid.x, centroid.y, centroid.z]) + np.dot(rot1, vec)
                    p2 = np.array([centroid.x, centroid.y, centroid.z]) + np.dot(rot2, vec)
                    rpy1 = rpy.copy()
                    rpy1[0] += theta
                    rpy2 = rpy.copy()
                    rpy2[0] -= theta
                    w,x,y,z = rpy_to_quaternion(rpy1[0], rpy1[1], rpy1[2])
                    pose_1 = Pose()
                    pose_1.position.x = p1[0]
                    pose_1.position.y = p1[1]
                    pose_1.position.z = p1[2]
                    pose_1.orientation.w = w
                    pose_1.orientation.x = x
                    pose_1.orientation.y = y
                    pose_1.orientation.z = z
                    # TODO: test if Pose is reachable and check if centroid is in view frustum
                    res.view_points.append(pose_1)
                    w,x,y,z = rpy_to_quaternion(rpy2[0], rpy2[1], rpy2[2])
                    pose_2 = Pose()
                    pose_2.position.x = p2[0]
                    pose_2.position.y = p2[1]
                    pose_2.position.z = p2[2]
                    pose_2.orientation.w = w
                    pose_2.orientation.x = x
                    pose_2.orientation.y = y
                    pose_2.orientation.z = z
                    res.view_points.append(pose_2)
                    # TODO: test if Pose is reachable
        return res
                    
                    




def quaternion_to_euler(w, x, y, z):
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    pitch = -math.pi * 0.5 + 2 * np.arctan2(np.sqrt(1 + 2 * (w * y - x * z)), np.sqrt(1 - 2 * (w * y - x * z)))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    return roll, pitch, yaw

def rpy_to_quaternion(roll, pitch, yaw):
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return w, x, y, z




def main(args=None):
    rclpy.init(args=args)
    node = ViewPointSampler()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
