import rclpy
from rclpy.node import Node
from nbv_interfaces.srv import ViewPointSampling
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import Point

import time

class TestNode(Node):
    def __init__(self):
        super().__init__("test")
        self.client = self.create_client(ViewPointSampling, 'vi_to_nav/view_point_sampling')
        self.sub = self.create_subscription(CameraInfo, "camera/camera_info", self.callback, 10)

    def callback(self, msg):
        req = ViewPointSampling.Request()
        req.centroids.append(Point(x = 2.0, y = 0.0, z = 0.05))
        req.cam_info = msg
        req.optical_frame = "camera_optical_frame"
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Service not available, waiting again....")
        res = self.client.call(req)
        self.get_logger().info(f"{res}")
        time.sleep(10)

def main(args=None):
    rclpy.init(args=args)

    node = TestNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()