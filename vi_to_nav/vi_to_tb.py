import rclpy
from rclpy import qos
from rclpy.node import Node
from nbv_interfaces.srv import ViewPointSampling
from nbv_interfaces.msg import CandidateView
from nbv_interfaces.action import MoveToPose
from tf2_ros.transform_listener import TransformListener
from tf2_ros.buffer import Buffer
from tf2_ros import LookupException, ExtrapolationException, TransformException
import tf2_geometry_msgs
from geometry_msgs.msg import Transform, Pose, PoseStamped, Vector3, Vector3Stamped
from std_msgs.msg import Header
import numpy as np
import math
from nav2_msgs.action import ComputePathToPose, NavigateToPose
import threading
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from functools import partial

from rclpy.action import ActionClient, ActionServer


class ViewPointSampler(Node):

    def __init__(self):
        super().__init__('view_point_sampler')
        self.cb_group = ReentrantCallbackGroup()
        self.create_service(ViewPointSampling, 'vi_to_nav/view_point_sampling', self.sampling_cb, callback_group= self.cb_group)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.map_frame_id = "map"
        self.base_footprint = "base_footprint"
        self.max_degree = 0.785 # 45 degrees
        self.num_poses = 1
        self.num_d = 3
        self.d_variation = 0.3
        self.deviation = 0.02
        self.near = 0.01
        self.far = 10
        self.action_client = ActionClient(self, ComputePathToPose, "/compute_path_to_pose")
        self.counter_lock = threading.Lock()
        self.cond = threading.Condition()


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
        self.transform = Pose()
        self.transform.position.x = cam_pose_map.position.x - base_pose_map.position.x
        self.transform.position.y = cam_pose_map.position.y - base_pose_map.position.y
        self.transform.position.z = cam_pose_map.position.z - base_pose_map.position.z
        
        roll,pitch,yaw = quaternion_to_euler(base_pose_map.orientation.w, base_pose_map.orientation.x, base_pose_map.orientation.y, base_pose_map.orientation.z)

        rpy = np.array([roll, pitch, yaw])
        self.pose_list = []
        self.future_count = 0
        try:
            optical_to_map = self.tf_buffer.lookup_transform(self.map_frame_id, req.optical_frame, rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().error(f"Failed to lookup transform from {self.base_footprint} to {self.map_frame_id}: {ex}")
            return res

        header = Header()
        header.frame_id = req.optical_frame
        header.stamp = cam_info.header.stamp
        # right = +x, up = -y, look_dir=+z
        up = Vector3(x = 0.0, y = -1.0, z = 0.0)
        up = Vector3Stamped(vector = up, header = header)
        up = tf2_geometry_msgs.do_transform_vector3(up, optical_to_map)

        look_dir = Vector3(x = 0.0, y = 0.0, z = 1.0)
        look_dir = Vector3Stamped(vector = look_dir, header = header)
        look_dir = tf2_geometry_msgs.do_transform_vector3(look_dir, optical_to_map)

        view_dir = np.array([look_dir.vector.x, look_dir.vector.y, look_dir.vector.z])
        view_dir /= np.linalg.norm(view_dir)
        up_dir = np.array([up.vector.x, up.vector.y, up.vector.z])
        up_dir /= np.linalg.norm(up_dir)

        tan_fov_x = np.tan(np.arctan2(cam_info.width/2, cam_info.k[0]))
        tan_fov_y = np.tan(np.arctan2(cam_info.height/2, cam_info.k[4]))

        self.get_logger().info("Checking centroids")

        for c in req.centroids:
            centroid = c.pos
            dir = np.array([centroid.x, centroid.y]) - np.array([base_pose_map.position.x, base_pose_map.position.y]) 
            d = np.linalg.norm(dir)

            for i in range(-(int) (self.num_d / 2), (int) (self.num_d/2 + 1)):
                if i == 0:
                    d_var = 0
                else:
                    d_var = d * (self.d_variation / i)
                
                for j in range(self.num_poses):
                    start = np.array([base_pose_map.position.x, base_pose_map.position.y]) - (dir / d) * d_var
                    vec = start - np.array([centroid.x , centroid.y])
                    theta = self.max_degree * (j + 1) / self.num_poses
                    self.handle_pose(theta, vec, base_pose_map, c, rpy, up_dir, view_dir, tan_fov_x, tan_fov_y)
                    self.handle_pose(-theta, vec, base_pose_map, c, rpy, up_dir, view_dir, tan_fov_x, tan_fov_y)

        self.get_logger().info("Waiting for futures") 
        while True:
            with self.counter_lock:
                if self.future_count <= 0:
                    break
            with self.cond:
                self.cond.wait()

        res.view_points = self.pose_list
        self.get_logger().info("Done")
        return res

    def handle_pose(self, theta, vec, base_pose_map, c, rpy, up, look_dir, tan_fov_x, tan_fov_y):
        centroid = c.pos
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)],])

        p = np.array([centroid.x, centroid.y]) + np.dot(rot, vec)
        rpy1 = rpy.copy()

        cam_pos = np.array([p[0] + self.transform.position.x, p[1] + self.transform.position.y, base_pose_map.position.z + self.transform.position.z])
        alpha = np.arccos(np.dot(np.array([centroid.x, centroid.y]) - np.array([cam_pos[0], cam_pos[1]]), np.array([look_dir[0], look_dir[1]])) / np.linalg.norm(np.array([centroid.x, centroid.y]) - np.array([cam_pos[0], cam_pos[1]])))
        if theta < 0:
            alpha = alpha * -1
        rpy1[2] += alpha 
        w,x,y,z = rpy_to_quaternion(rpy1[0], rpy1[1], rpy1[2])
        header = Header()
        header.frame_id = self.map_frame_id
        header.stamp = self.get_clock().now().to_msg()
        pose = Pose()
        pose.position.x = p[0]
        pose.position.y = p[1]
        pose.position.z = base_pose_map.position.z
        pose.orientation.w = w
        pose.orientation.x = x
        pose.orientation.y = y
        pose.orientation.z = z

        point_rel = np.array([centroid.x, centroid.y, centroid.z]) - cam_pos
        rot_alpha = np.array([[np.cos(alpha), -np.sin(alpha)],
                        [np.sin(alpha), np.cos(alpha)],])
        new_look_dir = np.dot(rot_alpha, np.array([look_dir[0], look_dir[1]]))
        new_look_dir = np.array([new_look_dir[0], new_look_dir[1], look_dir[2]])
        dist = np.dot(point_rel, new_look_dir)
        if dist < self.near or dist > self.far:
            self.get_logger().info(f"Too far or too near, {dist}")
            return

        right_proj = np.dot(point_rel, np.cross(new_look_dir, up))
        up_proj = np.dot(point_rel, up)
        width_at_dist = 2 * dist * tan_fov_x
        height_at_dist = 2 * dist * tan_fov_y
        if np.abs(up_proj) > height_at_dist / 2:
            self.get_logger().info(f"Too hight/low, {up_proj}")
            return
        if np.abs(right_proj) > width_at_dist / 2:
            self.get_logger().info(f"Too left/right, {right_proj}")
            return
        self.get_logger().info("Centroid is in frustum")
        goal = PoseStamped()    
        goal.pose= pose
        goal.header = header
        action = ComputePathToPose.Goal()
        action.goal = goal
        action.use_start = False
        action.planner_id = "GridBased"
        candidate = CandidateView()
        candidate.cam_pose = pose
        candidate.centroid = c
        with self.counter_lock:
            self.future_count += 1
        future1 = self.action_client.send_goal_async(action)
        future1.add_done_callback(partial(self.done_callback, candidate))
                    
    def done_callback(self, candidate, future):
        if future.result().accepted:
            future.result().get_result_async().add_done_callback(partial(self.get_res_callback, candidate))

    def get_res_callback(self, candidate, future):
        pose = candidate.cam_pose
        result = future.result().result
        reached = False
        reachable_pose = PoseStamped()
        d = 0
        if len(result.path.poses) != 0:
            reachable_pose = result.path.poses[len(result.path.poses) - 1]
            reached = (np.abs(reachable_pose.pose.position.x - pose.position.x) < self.deviation) and (np.abs(reachable_pose.pose.position.y - pose.position.y) < self.deviation) and (reachable_pose.pose.position.z == 0.0) and np.abs(reachable_pose.pose.orientation.w - pose.orientation.w) < self.deviation and np.abs(reachable_pose.pose.orientation.x - pose.orientation.x) < self.deviation and np.abs(reachable_pose.pose.orientation.y - pose.orientation.y) < self.deviation and np.abs(reachable_pose.pose.orientation.z - pose.orientation.z) < self.deviation
            if reached:
                prev = result.path.poses[0]
                for p in result.path.poses:
                    np_p = np.array([p.pose.position.x, p.pose.position.y, p.pose.position.z, p.pose.orientation.w, p.pose.orientation.x, p.pose.orientation.y, p.pose.orientation.z])
                    np_prev = np.array([prev.pose.position.x, prev.pose.position.y, prev.pose.position.z, prev.pose.orientation.w, prev.pose.orientation.x, prev.pose.orientation.y, prev.pose.orientation.z])
                    d += np.linalg.norm(np_p - np_prev)
                    prev = p
        with self.counter_lock:
            if reached:
                candidate.d = d
                candidate.cam_pose.position.x += self.transform.position.x
                candidate.cam_pose.position.y += self.transform.position.y
                candidate.cam_pose.position.z += self.transform.position.z
                # Orientation not important because tb cam has same orientation as base
                self.pose_list.append(candidate)
            self.future_count -= 1
            if self.future_count <= 0:
                with self.cond:
                    self.cond.notify_all()



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

def invert_quaternion(w, x, y, z):
    return w, -x, -y, -z

def quaternion_multiply(w1, x1, y1, z1, w2, x2, y2, z2):
    return w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + w2*x1 + y1*z2 - z1*y2, w1*y2 + w2*y1 - x1*z2 + x2*z1, w1*z2 + w2*z1 + x1*y2 - x2*y1




def main(args=None):
    rclpy.init(args=args)
    node = ViewPointSampler()
    executor = MultiThreadedExecutor(num_threads = 2)
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
