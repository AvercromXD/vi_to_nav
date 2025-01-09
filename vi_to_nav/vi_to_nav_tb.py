import threading
from nav2_msgs.action import ComputePathToPose, NavigateToPose
from std_msgs.msg import Header
from nbv_interfaces.action import MoveToPose
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.action import ActionClient, ActionServer
import rclpy
from rclpy import qos
from rclpy.node import Node
from tf2_ros.transform_listener import TransformListener
from tf2_ros.buffer import Buffer
from tf2_ros import LookupException, ExtrapolationException, TransformException
from geometry_msgs.msg import Pose
import tf2_geometry_msgs
from functools import partial
import numpy as np

class Navigator(Node):

    def __init__(self):
        super().__init__('vi_to_tb_navigator')
        self.action_server_group = ReentrantCallbackGroup()
        self.nav_client = ActionClient(self, NavigateToPose, "/navigate_to_pose")
        self.action_server = ActionServer(self, MoveToPose, "vi_to_nav/move_to_pose", execute_callback=self.execute_callback, cancel_callback=self.cancel_callback, callback_group=self.action_server_group)
        self.cond = threading.Condition()
        self.lock = threading.Lock()
        self.map_frame_id = "map"
        self.base_footprint = "base_footprint"
        self.deviation = 0.1
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def execute_callback(self, goal_handle):
        with self.lock:
            self.done = False
            self.canceled = False
        self.reached = False
        pose = goal_handle.request.camera_pose
        result = MoveToPose.Result()
        result.success = False

        try:
            base_to_map = self.tf_buffer.lookup_transform(self.map_frame_id, self.base_footprint, rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().error(f"Failed to lookup transform from {self.base_footprint} to {self.map_frame_id}: {ex}")
            result.message = f"Failed to lookup transform from {self.base_footprint} to {self.map_frame_id}: {ex}"
            goal_handle.abort()
            return result
        try:
            cam_to_map = self.tf_buffer.lookup_transform(self.map_frame_id, goal_handle.request.camera_frame, rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().error(f"Failed to lookup transform from {goal_handle.request.camera_frame} to {self.map_frame_id}: {ex}")
            result.message = f"Failed to lookup transform from {goal_handle.request.camera_frame} to {self.map_frame_id}: {ex}"
            goal_handle.abort()
            return result
        zero_pose = Pose()
        cam_pose_map = tf2_geometry_msgs.do_transform_pose(zero_pose, cam_to_map)
        base_pose_map = tf2_geometry_msgs.do_transform_pose(zero_pose, base_to_map)
        pose.position.x -= cam_pose_map.position.x - base_pose_map.position.x 
        pose.position.y -= cam_pose_map.position.y - base_pose_map.position.y 
        pose.position.z -= cam_pose_map.position.z - base_pose_map.position.z 
        nav_goal = NavigateToPose.Goal()
        header = Header()
        header.frame_id = self.map_frame_id
        header.stamp = self.get_clock().now().to_msg()
        nav_goal.pose.header = header
        nav_goal.pose.pose = pose
        future = self.nav_client.send_goal_async(nav_goal, feedback_callback=partial(self.feedback_cb, pose))
        future.add_done_callback(self.accepted_callback)
        while True:
            with self.lock:
                if self.canceled or self.done:
                    break
            with self.cond:
                self.cond.wait()
        if self.done:
            result.message = "Finished"
            with self.lock:
                result.success = self.reached
            goal_handle.succeed()
        if self.canceled:
            result.message = "Canceled"
            result.success = False
            future.cancel()
            goal_handle.abort()
        return result

    def cancel_callback(self):
        with self.lock:
            self.canceled = True
        with self.cond:
            self.cond.notify_all()

    def accepted_callback(self, future):
        if future.result().accepted:
            future.result().get_result_async().add_done_callback(self.done_cb)
        else:
            self.cancel_callback()

    
    def done_cb(self, future):
        with self.lock:
            self.done = True
        with self.cond:
            self.cond.notify_all()

    def feedback_cb(self, pose, msg):
        current_pose = msg.feedback.current_pose.pose
        deviation = self.deviation
        reached = (np.abs(current_pose.position.x - pose.position.x) < deviation) and (np.abs(current_pose.position.y - pose.position.y) < deviation) and (np.abs(current_pose.position.z - pose.position.z) < deviation ) and np.abs(current_pose.orientation.w - pose.orientation.w) < deviation and np.abs(current_pose.orientation.x - pose.orientation.x) < deviation and np.abs(current_pose.orientation.y - pose.orientation.y) < deviation and np.abs(current_pose.orientation.z - pose.orientation.z) < deviation
        if reached:
            with self.lock:
                self.reached = reached


def main(args=None):
    rclpy.init(args=args)
    node = Navigator()
    executor = MultiThreadedExecutor(num_threads = 2)
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()