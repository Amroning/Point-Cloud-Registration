#include<iostream>
#include<pcl/io/pcd_io.h>
#include<pcl/point_types.h>
#include<pcl/common/io.h>
#include<pcl/keypoints/narf_keypoint.h>
#include<pcl/features/normal_3d.h>
#include<pcl/features/fpfh_omp.h>
#include<pcl/visualization/pcl_visualizer.h>
#include<pcl/visualization/cloud_viewer.h>
#include<boost/thread/thread.hpp>
#include<pcl/common/common_headers.h>
#include<pcl/registration/ia_ransac.h>
#include<pcl/range_image/range_image.h>
#include<pcl/features/range_image_border_extractor.h>
using namespace std;

void extract_keypoint(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& keypoint)
{
	pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::CAMERA_FRAME;
	Eigen::Affine3f scene_senor_pose(Eigen::Affine3f::Identity());
	scene_senor_pose = Eigen::Affine3f(Eigen::Translation3f(cloud->sensor_origin_[0], cloud->sensor_origin_[1], cloud->sensor_origin_[2])) * Eigen::Affine3f(cloud->sensor_orientation_);

	//从点云创建深度图像
	float noise_level = 0.0;
	float min_range = 0.0f;
	int border_size = 1;
	pcl::RangeImage::Ptr range_image(new pcl::RangeImage);
	range_image->createFromPointCloud(*cloud, pcl::deg2rad(0.02f), pcl::deg2rad(0.02f), pcl::deg2rad(360.0f), pcl::deg2rad(180.0f),
		scene_senor_pose, coordinate_frame, noise_level, min_range, border_size);

	//提取NARF关键点
	pcl::RangeImage& range_image0 = *range_image;
	pcl::RangeImageBorderExtractor range_image_border_extractor;

	pcl::NarfKeypoint narf_keypoint_detector(&range_image_border_extractor);
	narf_keypoint_detector.setRangeImage(&range_image0);
	narf_keypoint_detector.getParameters().support_size = 0.2f;
	narf_keypoint_detector.getParameters().add_points_on_straight_edges = true;
	narf_keypoint_detector.getParameters().distance_for_additional_points = 0.5;
	pcl::PointCloud<int>keypoint_indices;
	narf_keypoint_detector.compute(keypoint_indices);
	keypoint->points.resize(keypoint_indices.size());
	keypoint->height = keypoint_indices.height;
	keypoint->width = keypoint_indices.width;
	for (size_t i = 0; i < keypoint_indices.points.size(); ++i)
	{
		keypoint->points[i].getVector3fMap() = range_image->points[keypoint_indices.points[i]].getVector3fMap();
	}

	//可视化
	boost::shared_ptr<pcl::visualization::PCLVisualizer>viewer1(new pcl::visualization::PCLVisualizer("关键点"));
	viewer1->setBackgroundColor(0, 0, 0);

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>cloud_color(cloud, 0.0, 255.0, 0.0);
	viewer1->addPointCloud<pcl::PointXYZ>(cloud, cloud_color, "cloud color");
	viewer1->addPointCloud<pcl::PointXYZ>(keypoint, "key color");
	viewer1->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "key color");
	viewer1->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "key color");

}

pcl::PointCloud<pcl::FPFHSignature33>::Ptr compute_fpfh_feature(pcl::PointCloud<pcl::PointXYZ>::Ptr& keypoint)
{
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree;
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
	n.setInputCloud(keypoint);
	n.setSearchMethod(tree);
	n.setKSearch(10);
	n.compute(*normals);
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh(new pcl::PointCloud<pcl::FPFHSignature33>);
	pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> f;
	f.setNumberOfThreads(8);
	f.setInputCloud(keypoint);
	f.setInputNormals(normals);
	f.setSearchMethod(tree);
	f.setRadiusSearch(50);
	f.compute(*fpfh);

	return fpfh;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr sac_align(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr s_k, pcl::PointCloud<pcl::PointXYZ>::Ptr t_k, pcl::PointCloud<pcl::FPFHSignature33>::Ptr sk_fpfh, pcl::PointCloud<pcl::FPFHSignature33>::Ptr tk_fpfh)
{
	pcl::SampleConsensusInitialAlignment<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33> scia;
	scia.setInputSource(s_k);
	scia.setInputTarget(t_k);
	scia.setSourceFeatures(sk_fpfh);
	scia.setTargetFeatures(tk_fpfh);
	scia.setMinSampleDistance(0.7);///参数：设置采样点之间的最小距离，满足的被当做采样点
	scia.setNumberOfSamples(150);////设置每次迭代设置采样点的个数(这个参数多可以增加配准精度)
	scia.setCorrespondenceRandomness(6);//设置选择随机特征对应点时要使用的邻域点个数。值越大，特征匹配的随机性就越大
	scia.setEuclideanFitnessEpsilon(0.001);
	scia.setTransformationEpsilon(1e-10);
	scia.setRANSACIterations(30);
	pcl::PointCloud<pcl::PointXYZ>::Ptr sac_result(new pcl::PointCloud<pcl::PointXYZ>);
	scia.align(*sac_result);
	pcl::transformPointCloud(*cloud, *sac_result, scia.getFinalTransformation());
	return sac_result;
}

int main(int argc, char** argv)
{
	//读取数据
	pcl::PointCloud<pcl::PointXYZ>::Ptr source(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr target(new pcl::PointCloud<pcl::PointXYZ>);
	if (pcl::io::loadPCDFile<pcl::PointXYZ>("pig_view1.pcd", *source) == -1)
	{
		PCL_ERROR("加载点云失败\n");
	}
	if (pcl::io::loadPCDFile<pcl::PointXYZ>("pig_view2.pcd", *target) == -1)
	{
		PCL_ERROR("加载点云失败\n");
	}

	//粗配准
	pcl::PointCloud<pcl::PointXYZ>::Ptr s_k(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr t_k(new pcl::PointCloud<pcl::PointXYZ>);
	extract_keypoint(source, s_k);
	extract_keypoint(target, t_k);

	std::cout << "source关键点数量" << s_k->size() << std::endl;
	std::cout << "target关键点数量" << t_k->size() << std::endl;
	
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr sk_fpfh = compute_fpfh_feature(s_k);
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr tk_fpfh = compute_fpfh_feature(t_k);

	pcl::PointCloud<pcl::PointXYZ>::Ptr result(new pcl::PointCloud<pcl::PointXYZ>);
	result = sac_align(source, s_k, t_k, sk_fpfh, tk_fpfh);

	//可视化
	boost::shared_ptr<pcl::visualization::PCLVisualizer>viewer(new pcl::visualization::PCLVisualizer("显示点云"));
	viewer->setBackgroundColor(0, 0, 0);
	// 对目标点云着色可视化 (red).
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>target_color(target, 255, 0, 0);
	viewer->addPointCloud<pcl::PointXYZ>(target, target_color, "target cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "target cloud");
	// 对源点云着色可视化 (green).
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>input_color(result, 0, 255, 0);
	viewer->addPointCloud<pcl::PointXYZ>(result, input_color, "input cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "input cloud");

	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100));
	}

	return 0;
}