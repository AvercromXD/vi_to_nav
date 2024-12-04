import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField, CameraInfo
from cv_bridge import CvBridge
import numpy as np
from geometry_msgs.msg import Point32
from sensor_msgs_py import point_cloud2
import image_geometry


class DepthImageToPointCloudNode(Node):
    def __init__(self):
        super().__init__('depth_to_pc')
        self.bridge =CvBridge()
        self.camera_model = None

        self.create_subscription(Image, '/depth/image', self.depth_image_callback, 10)
        self.create_subscription(CameraInfo, '/depth/camera_info', self.info_callback, 10)
        self.pointcloud_publisher = self.create_publisher(PointCloud2, 'depth/pointcloud2', 10)

        self.get_logger().info('depth_to_pc ready')

    def info_callback(self, msg):
        self.camera_model = image_geometry.PinholeCameraModel()
        self.camera_model.fromCameraInfo(msg)
        self.get_logger().info('Camera model initialized')

    def depth_image_callback(self, msg):
        if self.camera_model is None:
            self.get_logger().warn("Camera model not yet initialized")
            return
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"Failed to convert depth image: {e}")
            return
        
        height, width = cv_image.shape
        u, v = np.meshgrid(np.arange(width), np.arange(height))


def main(args=None):
    rclpy.init(args=args)
    node = DepthImageToPointCloudNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()