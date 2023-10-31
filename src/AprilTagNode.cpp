// ros
#include <apriltag_msgs/msg/april_tag_detection.hpp>
#include <apriltag_msgs/msg/april_tag_detection_array.hpp>
#ifdef cv_bridge_HPP
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif
#include "common/homography.h"// from apriltag lib
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <image_geometry/pinhole_camera_model.hpp>
#include <image_transport/camera_subscriber.hpp>
#include <image_transport/image_transport.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>
#include <tf2_ros/transform_broadcaster.h>
// apriltag
#include "tag_functions.hpp"
#include <apriltag.h>

#include <Eigen/Dense>


#define IF(N, V) \
    if(assign_check(parameter, N, V)) continue;

template<typename T>
void assign(const rclcpp::Parameter& parameter, T& var)
{
    var = parameter.get_value<T>();
}

template<typename T>
void assign(const rclcpp::Parameter& parameter, std::atomic<T>& var)
{
    var = parameter.get_value<T>();
}

template<typename T>
bool assign_check(const rclcpp::Parameter& parameter, const std::string& name, T& var)
{
    if(parameter.get_name() == name) {
        assign(parameter, var);
        return true;
    }
    return false;
}


typedef Eigen::Matrix<double, 3, 3, Eigen::RowMajor> Mat3;

rcl_interfaces::msg::ParameterDescriptor
descr(const std::string& description, const bool& read_only = false)
{
    rcl_interfaces::msg::ParameterDescriptor descr;

    descr.description = description;
    descr.read_only = read_only;

    return descr;
}

void getPose(const matd_t& H,
             const Mat3& Pinv,
             geometry_msgs::msg::Transform& t,
             const double size)
{
    // compute extrinsic camera parameter
    // https://dsp.stackexchange.com/a/2737/31703
    // H = K * T  =>  T = K^(-1) * H
    const Mat3 T = Pinv * Eigen::Map<const Mat3>(H.data);
    Mat3 R;
    R.col(0) = T.col(0).normalized();
    R.col(1) = T.col(1).normalized();
    R.col(2) = R.col(0).cross(R.col(1));

    // rotate by half rotation about x-axis to have z-axis
    // point upwards orthogonal to the tag plane
    R.col(1) *= -1;
    R.col(2) *= -1;

    // the corner coordinates of the tag in the canonical frame are (+/-1, +/-1)
    // hence the scale is half of the edge size
    const Eigen::Vector3d tt = T.rightCols<1>() / ((T.col(0).norm() + T.col(0).norm()) / 2.0) * (size / 2.0);

    const Eigen::Quaterniond q(R);

    t.translation.x = tt.x();
    t.translation.y = tt.y();
    t.translation.z = tt.z();
    t.rotation.w = q.w();
    t.rotation.x = q.x();
    t.rotation.y = q.y();
    t.rotation.z = q.z();
}

struct TagBundleMember
{
    int id;          // Payload ID
    double size;     // [m] Side length
    cv::Matx44d T_oi;// Rigid transform from tag i frame to bundle origin frame
};

class TagBundleDescription {
public:
    std::map<int, int> id2idx_;// (id2idx_[<tag ID>]=<index in tags_>) mapping

    TagBundleDescription(const std::string& name) : name_(name) {}

    void addMemberTag(int id, double size, cv::Matx44d T_oi)
    {
        TagBundleMember member;
        member.id = id;
        member.size = size;
        member.T_oi = T_oi;
        tags_.push_back(member);
        id2idx_[id] = tags_.size() - 1;
    }

    const std::string& name() const { return name_; }
    // Get IDs of bundle member tags
    std::vector<int> bundleIds()
    {
        std::vector<int> ids;
        for(unsigned int i = 0; i < tags_.size(); i++) {
            ids.push_back(tags_[i].id);
        }
        return ids;
    }
    // Get sizes of bundle member tags
    std::vector<double> bundleSizes()
    {
        std::vector<double> sizes;
        for(unsigned int i = 0; i < tags_.size(); i++) {
            sizes.push_back(tags_[i].size);
        }
        return sizes;
    }
    int memberID(int tagID) { return tags_[id2idx_[tagID]].id; }
    double memberSize(int tagID) { return tags_[id2idx_[tagID]].size; }
    cv::Matx44d memberT_oi(int tagID) { return tags_[id2idx_[tagID]].T_oi; }

private:
    // Bundle description
    std::string name_;
    std::vector<TagBundleMember> tags_;
};


class AprilTagNode : public rclcpp::Node {
public:
    AprilTagNode(const rclcpp::NodeOptions& options);

    ~AprilTagNode() override;

private:
    const OnSetParametersCallbackHandle::SharedPtr cb_parameter;

    apriltag_family_t* tf;
    apriltag_detector_t* const td;

    // parameter
    std::mutex mutex;
    double tag_edge_size;
    std::atomic<int> max_hamming;
    std::atomic<bool> profile;
    std::unordered_map<int, std::string> tag_frames;
    std::unordered_map<int, double> tag_sizes;

    std::unordered_map<int, int> bundle_ids;
    std::unordered_map<int, double> bundle_sizes;
    std::unordered_map<int, double> bundle_positions;
    std::unordered_map<int, double> bundle_orientations;
    std::shared_ptr<TagBundleDescription> bundle;

    std::function<void(apriltag_family_t*)> tf_destructor;

    const image_transport::CameraSubscriber sub_cam;
    const rclcpp::Publisher<apriltag_msgs::msg::AprilTagDetectionArray>::SharedPtr pub_detections;
    const rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_detections_image;
    tf2_ros::TransformBroadcaster tf_broadcaster;

    void onCamera(const sensor_msgs::msg::Image::ConstSharedPtr& msg_img, const sensor_msgs::msg::CameraInfo::ConstSharedPtr& msg_ci);
    void drawDetections(cv_bridge::CvImagePtr image, zarray_t* detections);
    Eigen::Isometry3d getRelativeTransform(const std::vector<cv::Point3d>& objectPoints, const std::vector<cv::Point2d>& imagePoints, double fx, double fy, double cx, double cy) const;

    void addObjectPoints(double s, cv::Matx44d T_oi, std::vector<cv::Point3d>& objectPoints) const;

    void addImagePoints(apriltag_detection_t* detection, std::vector<cv::Point2d>& imagePoints) const;

    geometry_msgs::msg::PoseWithCovarianceStamped makeTagPose(const Eigen::Isometry3d& transform, const std_msgs::msg::Header& header);

    rcl_interfaces::msg::SetParametersResult onParameter(const std::vector<rclcpp::Parameter>& parameters);
};

RCLCPP_COMPONENTS_REGISTER_NODE(AprilTagNode)


AprilTagNode::AprilTagNode(const rclcpp::NodeOptions& options)
  : Node("apriltag", options),
    // parameter
    cb_parameter(add_on_set_parameters_callback(std::bind(&AprilTagNode::onParameter, this, std::placeholders::_1))),
    td(apriltag_detector_create()),
    // topics
    sub_cam(image_transport::create_camera_subscription(this, "image_rect", std::bind(&AprilTagNode::onCamera, this, std::placeholders::_1, std::placeholders::_2), declare_parameter("image_transport", "raw", descr({}, true)), rmw_qos_profile_sensor_data)),
    pub_detections(create_publisher<apriltag_msgs::msg::AprilTagDetectionArray>("detections", rclcpp::QoS(1))),
    pub_detections_image(create_publisher<sensor_msgs::msg::Image>("detections/image", rclcpp::QoS(1))),
    tf_broadcaster(this)
{
    // read-only parameters
    const std::string tag_family = declare_parameter("family", "36h11", descr("tag family", true));
    tag_edge_size = declare_parameter("size", 1.0, descr("default tag size", true));

    // get tag names, IDs and sizes
    const auto ids = declare_parameter("tag.ids", std::vector<int64_t>{}, descr("tag ids", true));
    const auto frames = declare_parameter("tag.frames", std::vector<std::string>{}, descr("tag frame names per id", true));
    const auto sizes = declare_parameter("tag.sizes", std::vector<double>{}, descr("tag sizes per id", true));
    const auto bundle_name = declare_parameter("bundle.name", std::string{}, descr("tag bundle name", true));
    const auto bundle_ids = declare_parameter("bundle.ids", std::vector<int64_t>{}, descr("bundle tag ids", true));

    // detector parameters in "detector" namespace
    declare_parameter("detector.threads", td->nthreads, descr("number of threads"));
    declare_parameter("detector.decimate", td->quad_decimate, descr("decimate resolution for quad detection"));
    declare_parameter("detector.blur", td->quad_sigma, descr("sigma of Gaussian blur for quad detection"));
    declare_parameter("detector.refine", td->refine_edges, descr("snap to strong gradients"));
    declare_parameter("detector.sharpening", td->decode_sharpening, descr("sharpening of decoded images"));
    declare_parameter("detector.debug", td->debug, descr("write additional debugging images to working directory"));

    declare_parameter("max_hamming", 0, descr("reject detections with more corrected bits than allowed"));
    declare_parameter("profile", false, descr("print profiling information to stdout"));

    if(!frames.empty()) {
        if(ids.size() != frames.size()) {
            throw std::runtime_error("Number of tag ids (" + std::to_string(ids.size()) + ") and frames (" + std::to_string(frames.size()) + ") mismatch!");
        }
        for(size_t i = 0; i < ids.size(); i++) { tag_frames[ids[i]] = frames[i]; }
    }

    if(!sizes.empty()) {
        // use tag specific size
        if(ids.size() != sizes.size()) {
            throw std::runtime_error("Number of tag ids (" + std::to_string(ids.size()) + ") and sizes (" + std::to_string(sizes.size()) + ") mismatch!");
        }
        for(size_t i = 0; i < ids.size(); i++) { tag_sizes[ids[i]] = sizes[i]; }
    }


    if(!bundle_ids.empty()) {
        bundle = std::make_shared<TagBundleDescription>(bundle_name.c_str());

        for(size_t i = 0; i < bundle_ids.size(); i++) {
            int64_t bundle_id = bundle_ids[i];
            RCLCPP_INFO(get_logger(), "Found tag id %lo specified in bundle %s", bundle_id, bundle_name.c_str());
            std::string layout_base = "bundle.layout.tag_";
            RCLCPP_ERROR(get_logger(), layout_base.c_str());
            auto bundle_tag_data = declare_parameter(layout_base + std::to_string(bundle_id), std::vector<double>{}, descr("bundle tag ids", true));
            if(bundle_tag_data.empty()) {
                RCLCPP_ERROR(get_logger(), "no tag measurements found");
            }
            else {
                RCLCPP_INFO(get_logger(), "got tag measurements");
                double x = bundle_tag_data[0];
                double y = bundle_tag_data[1];
                double z = bundle_tag_data[2];

                double qw = bundle_tag_data[3];
                double qx = bundle_tag_data[4];
                double qy = bundle_tag_data[5];
                double qz = bundle_tag_data[6];
                RCLCPP_INFO(get_logger(), "%lo %f %f %f %f %f %f %f %f", i, tag_sizes[i], x, y, z, qw, qx, qy, qz);
                Eigen::Quaterniond q_tag(qw, qx, qy, qz);
                q_tag.normalize();
                Eigen::Matrix3d R_oi = q_tag.toRotationMatrix();

                // Build the rigid transform from tag_j to the bundle origin
                cv::Matx44d T_mj(R_oi(0, 0), R_oi(0, 1), R_oi(0, 2), x,
                                 R_oi(1, 0), R_oi(1, 1), R_oi(1, 2), y,
                                 R_oi(2, 0), R_oi(2, 1), R_oi(2, 2), z,
                                 0, 0, 0, 1);

                bundle->addMemberTag(i, tag_sizes[i], T_mj);
            }
        }
    }

    if(tag_fun.count(tag_family)) {
        tf = tag_fun.at(tag_family).first();
        tf_destructor = tag_fun.at(tag_family).second;
        apriltag_detector_add_family(td, tf);
    }
    else {
        throw std::runtime_error("Unsupported tag family: " + tag_family);
    }
}

AprilTagNode::~AprilTagNode()
{
    apriltag_detector_destroy(td);
    tf_destructor(tf);
}

void AprilTagNode::onCamera(const sensor_msgs::msg::Image::ConstSharedPtr& msg_img,
                            const sensor_msgs::msg::CameraInfo::ConstSharedPtr& msg_ci)
{
    // precompute inverse projection matrix
    const Mat3 Pinv = Eigen::Map<const Eigen::Matrix<double, 3, 4, Eigen::RowMajor>>(msg_ci->p.data()).leftCols<3>().inverse();

    // convert to 8bit monochrome image
    const cv::Mat img_uint8 = cv_bridge::toCvShare(msg_img, "mono8")->image;

    image_u8_t im{img_uint8.cols, img_uint8.rows, img_uint8.cols, img_uint8.data};

    image_geometry::PinholeCameraModel camera_model;
    camera_model.fromCameraInfo(msg_ci);

    // Get camera intrinsic properties for rectified image.
    double fx = camera_model.fx();// focal length in camera x-direction [px]
    double fy = camera_model.fy();// focal length in camera y-direction [px]
    double cx = camera_model.cx();// optical center x-coordinate [px]
    double cy = camera_model.cy();// optical center y-coordinate [px]

    // detect tags
    mutex.lock();
    zarray_t* detections = apriltag_detector_detect(td, &im);
    mutex.unlock();

    if(profile)
        timeprofile_display(td->tp);

    apriltag_msgs::msg::AprilTagDetectionArray msg_detections;
    msg_detections.header = msg_img->header;

    std::vector<geometry_msgs::msg::TransformStamped> tfs;
    std::vector<cv::Point3d> bundleObjectPoints;
    std::vector<cv::Point2d> bundleImagePoints;
    bool should_publish_bundle = false;
    for(int i = 0; i < zarray_size(detections); i++) {
        apriltag_detection_t* det;
        zarray_get(detections, i, &det);

        RCLCPP_DEBUG(get_logger(),
                     "detection %3d: id (%2dx%2d)-%-4d, hamming %d, margin %8.3f\n",
                     i, det->family->nbits, det->family->h, det->id,
                     det->hamming, det->decision_margin);

        // ignore untracked tags
        if(!tag_frames.empty() && !tag_frames.count(det->id)) { continue; }

        // reject detections with more corrected bits than allowed
        if(det->hamming > max_hamming) { continue; }

        // detection
        apriltag_msgs::msg::AprilTagDetection msg_detection;
        msg_detection.family = std::string(det->family->name);
        msg_detection.id = det->id;
        msg_detection.hamming = det->hamming;
        msg_detection.decision_margin = det->decision_margin;
        msg_detection.centre.x = det->c[0];
        msg_detection.centre.y = det->c[1];
        std::memcpy(msg_detection.corners.data(), det->p, sizeof(double) * 8);
        std::memcpy(msg_detection.homography.data(), det->H->data, sizeof(double) * 9);
        msg_detections.detections.push_back(msg_detection);

        // 3D orientation and position
        geometry_msgs::msg::TransformStamped tf;
        tf.header = msg_img->header;
        // set child frame name by generic tag name or configured tag name
        tf.child_frame_id = tag_frames.count(det->id) ? tag_frames.at(det->id) : std::string(det->family->name) + ":" + std::to_string(det->id);
        getPose(*(det->H), Pinv, tf.transform, tag_sizes.count(det->id) ? tag_sizes.at(det->id) : tag_edge_size);
        tfs.push_back(tf);
        if(bundle->id2idx_.find(det->id) != bundle->id2idx_.end()) {
            double s = bundle->memberSize(det->id) / 2;
            addObjectPoints(s, bundle->memberT_oi(det->id), bundleObjectPoints);
            //===== Corner points in the image frame coordinates
            addImagePoints(det, bundleImagePoints);
            should_publish_bundle = true;
        }
    }

    if(should_publish_bundle) {
        Eigen::Isometry3d transform = getRelativeTransform(bundleObjectPoints, bundleImagePoints, fx, fy, cx, cy);
        geometry_msgs::msg::PoseWithCovarianceStamped bundle_pose = makeTagPose(transform, msg_img->header);
        geometry_msgs::msg::TransformStamped bundle_transform;
        bundle_transform.header = bundle_pose.header;
        bundle_transform.child_frame_id = bundle->name();
        bundle_transform.transform.translation.x = bundle_pose.pose.pose.position.x;
        bundle_transform.transform.translation.y = bundle_pose.pose.pose.position.y;
        bundle_transform.transform.translation.z = bundle_pose.pose.pose.position.z;
        bundle_transform.transform.rotation.x = bundle_pose.pose.pose.orientation.x;
        bundle_transform.transform.rotation.y = bundle_pose.pose.pose.orientation.y;
        bundle_transform.transform.rotation.z = bundle_pose.pose.pose.orientation.z;
        bundle_transform.transform.rotation.w = bundle_pose.pose.pose.orientation.w;
        tf_broadcaster.sendTransform(bundle_transform);
    }

    if(!tfs.empty()) {
        pub_detections->publish(msg_detections);
        tf_broadcaster.sendTransform(tfs);
        cv_bridge::CvImagePtr detections_image = cv_bridge::toCvCopy(msg_img, msg_img->encoding);
        drawDetections(detections_image, detections);
        sensor_msgs::msg::Image detections_image_ros;
        detections_image->toImageMsg(detections_image_ros);
        pub_detections_image->publish(detections_image_ros);
    }
    apriltag_detections_destroy(detections);
}

rcl_interfaces::msg::SetParametersResult
AprilTagNode::onParameter(const std::vector<rclcpp::Parameter>& parameters)
{
    rcl_interfaces::msg::SetParametersResult result;

    mutex.lock();

    for(const rclcpp::Parameter& parameter : parameters) {
        RCLCPP_DEBUG_STREAM(get_logger(), "setting: " << parameter);

        IF("detector.threads", td->nthreads)
        IF("detector.decimate", td->quad_decimate)
        IF("detector.blur", td->quad_sigma)
        IF("detector.refine", td->refine_edges)
        IF("detector.sharpening", td->decode_sharpening)
        IF("detector.debug", td->debug)
        IF("max_hamming", max_hamming)
        IF("profile", profile)
    }

    mutex.unlock();

    result.successful = true;

    return result;
}

void AprilTagNode::addObjectPoints(
    double s, cv::Matx44d T_oi, std::vector<cv::Point3d>& objectPoints) const
{
    // Add to object point vector the tag corner coordinates in the bundle frame
    // Going counterclockwise starting from the bottom left corner
    objectPoints.push_back(T_oi.get_minor<3, 4>(0, 0) * cv::Vec4d(-s, -s, 0, 1));
    objectPoints.push_back(T_oi.get_minor<3, 4>(0, 0) * cv::Vec4d(s, -s, 0, 1));
    objectPoints.push_back(T_oi.get_minor<3, 4>(0, 0) * cv::Vec4d(s, s, 0, 1));
    objectPoints.push_back(T_oi.get_minor<3, 4>(0, 0) * cv::Vec4d(-s, s, 0, 1));
}

void AprilTagNode::addImagePoints(
    apriltag_detection_t* detection,
    std::vector<cv::Point2d>& imagePoints) const
{
    // Add to image point vector the tag corners in the image frame
    // Going counterclockwise starting from the bottom left corner
    double tag_x[4] = {-1, 1, 1, -1};
    double tag_y[4] = {1, 1, -1, -1};// Negated because AprilTag tag local
                                     // frame has y-axis pointing DOWN
                                     // while we use the tag local frame
                                     // with y-axis pointing UP
    for(int i = 0; i < 4; i++) {
        // Homography projection taking tag local frame coordinates to image pixels
        double im_x, im_y;
        homography_project(detection->H, tag_x[i], tag_y[i], &im_x, &im_y);
        imagePoints.push_back(cv::Point2d(im_x, im_y));
    }
}

Eigen::Isometry3d AprilTagNode::getRelativeTransform(
    const std::vector<cv::Point3d>& objectPoints,
    const std::vector<cv::Point2d>& imagePoints,
    double fx, double fy, double cx, double cy) const
{
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();// homogeneous transformation matrix

    // perform Perspective-n-Point camera pose estimation using the
    // above 3D-2D point correspondences
    cv::Mat rvec, tvec;
    cv::Matx33d cameraMatrix(fx, 0, cx,
                             0, fy, cy,
                             0, 0, 1);
    cv::Vec4f distCoeffs(0, 0, 0, 0);// distortion coefficients
    // TODO Perhaps something like SOLVEPNP_EPNP would be faster? Would
    // need to first check WHAT is a bottleneck in this code, and only
    // do this if PnP solution is the bottleneck.
    cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);
    cv::Matx33d R;
    cv::Rodrigues(rvec, R);

    // rotation
    T.linear() << R(0, 0), R(0, 1), R(0, 2), R(1, 0), R(1, 1), R(1, 2), R(2, 0), R(2, 1), R(2, 2);

    // translation
    T.translation() = Eigen::Vector3d::Map(reinterpret_cast<const double*>(tvec.data));

    return T;
}

geometry_msgs::msg::PoseWithCovarianceStamped AprilTagNode::makeTagPose(
    const Eigen::Isometry3d& transform,
    const std_msgs::msg::Header& header)
{
    geometry_msgs::msg::PoseWithCovarianceStamped pose;
    pose.header = header;
    Eigen::Quaterniond rot_quaternion(transform.linear());
    //===== Position and orientation
    pose.pose.pose.position.x = transform.translation().x();
    pose.pose.pose.position.y = transform.translation().y();
    pose.pose.pose.position.z = transform.translation().z();
    pose.pose.pose.orientation.x = rot_quaternion.x();
    pose.pose.pose.orientation.y = rot_quaternion.y();
    pose.pose.pose.orientation.z = rot_quaternion.z();
    pose.pose.pose.orientation.w = rot_quaternion.w();
    return pose;
}


void AprilTagNode::drawDetections(cv_bridge::CvImagePtr image, zarray_t* detections)
{
    int thickness = 3;
    for(int i = 0; i < zarray_size(detections); i++) {
        apriltag_detection_t* det;
        zarray_get(detections, i, &det);
        // Draw tag outline with edge colors green, blue, blue, red
        // (going counter-clockwise, starting from lower-left corner in
        // tag coords). cv::Scalar(Blue, Green, Red) format for the edge
        // colors!
        line(image->image, cv::Point((int) det->p[0][0], (int) det->p[0][1]),
             cv::Point((int) det->p[1][0], (int) det->p[1][1]),
             cv::Scalar(0, 0xff, 0), thickness);// green
        line(image->image, cv::Point((int) det->p[0][0], (int) det->p[0][1]),
             cv::Point((int) det->p[3][0], (int) det->p[3][1]),
             cv::Scalar(0, 0, 0xff), thickness);// red
        line(image->image, cv::Point((int) det->p[1][0], (int) det->p[1][1]),
             cv::Point((int) det->p[2][0], (int) det->p[2][1]),
             cv::Scalar(0xff, 0, 0), thickness);// blue
        line(image->image, cv::Point((int) det->p[2][0], (int) det->p[2][1]),
             cv::Point((int) det->p[3][0], (int) det->p[3][1]),
             cv::Scalar(0xff, 0, 0), thickness);// blue
        // Print tag ID in the middle of the tag
        std::stringstream ss;
        ss << det->id;
        cv::String text = ss.str();
        int fontface = cv::FONT_HERSHEY_SCRIPT_SIMPLEX;
        double fontscale = 0.5;
        int baseline;
        cv::Size textsize = cv::getTextSize(text, fontface,
                                            fontscale, 2, &baseline);
        cv::putText(image->image, text,
                    cv::Point((int) (det->c[0] - textsize.width / 2),
                              (int) (det->c[1] + textsize.height / 2)),
                    fontface, fontscale, cv::Scalar(0xff, 0x99, 0), 2);
    }
}
