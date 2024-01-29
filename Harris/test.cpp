#include<iostream>
#include<pcl/io/pcd_io.h>
#include<pcl/point_types.h>
#include<pcl/common/io.h>
#include<pcl/keypoints/harris_3d.h>
#include<pcl/features/normal_3d.h>
#include<pcl/features/fpfh_omp.h>
#include<pcl/visualization/pcl_visualizer.h>
#include<pcl/visualization/cloud_viewer.h>
#include<boost/thread/thread.hpp>
#include<pcl/filters/voxel_grid.h>
#include<pcl/common/common_headers.h>
#include<pcl/registration/ia_ransac.h>
using namespace std;

void voxel_grid(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_filt)
{
	pcl::VoxelGrid<pcl::PointXYZ> vg;
	vg.setInputCloud(cloud);
	vg.setLeafSize(0.02f, 0.02f, 0.02f);
	vg.filter(*cloud_filt);
}

void extract_keypoint(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::PointCloud<pcl::PointXYZI>::Ptr& keypoint)
{
	//提取特征点
	pcl::HarrisKeypoint3D<pcl::PointXYZ, pcl::PointXYZI> harris;
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	harris.setInputCloud(cloud);
	harris.setSearchMethod(tree);
	harris.setNumberOfThreads(8);
	harris.setRadius(4);
	harris.setRadiusSearch(4);
	harris.setNonMaxSupression(true);
	//harris.setThreshold(1e-6);
	harris.compute(*keypoint);

	//特征点可视化
	pcl::PointIndicesConstPtr keypoint2_indices = harris.getKeypointsIndices();
	pcl::PointCloud<pcl::PointXYZ>::Ptr keys(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::copyPointCloud(*cloud, *keypoint2_indices, *keys);

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer1(new pcl::visualization::PCLVisualizer("显示特征点"));
	viewer1->setBackgroundColor(255, 255, 255);
	viewer1->setWindowName("Harris keypoints");

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>cloud_color(cloud, 0.0, 255.0, 0.0);
	viewer1->addPointCloud<pcl::PointXYZ>(cloud, cloud_color, "cloud color");
	viewer1->addPointCloud<pcl::PointXYZ>(keys, "key color");
	viewer1->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "key color");
	viewer1->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "key color");

	//while (!viewer1->wasStopped())
	//{
	//	viewer1->spinOnce(100);
	//	boost::this_thread::sleep(boost::posix_time::microseconds(100));
	//}
}

pcl::PointCloud<pcl::FPFHSignature33>::Ptr compute_fpfh_feature(pcl::PointCloud<pcl::PointXYZI>::Ptr keypoint)
{
	//Normal
	pcl::search::KdTree<pcl::PointXYZI>::Ptr tree;
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud < pcl::Normal>);
	pcl::NormalEstimation<pcl::PointXYZI, pcl::Normal> n;
	n.setInputCloud(keypoint);
	n.setSearchMethod(tree);
	n.setKSearch(10);
	n.compute(*normals);

	//FPFH
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh(new pcl::PointCloud<pcl::FPFHSignature33>);
	pcl::FPFHEstimationOMP<pcl::PointXYZI, pcl::Normal, pcl::FPFHSignature33> f;
	f.setInputCloud(keypoint);
	f.setInputNormals(normals);
	f.setSearchMethod(tree);
	f.setNumberOfThreads(8);
	f.setRadiusSearch(50);
	f.compute(*fpfh);

	return fpfh;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr sac_align(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZI>::Ptr s_k, pcl::PointCloud<pcl::PointXYZI>::Ptr t_k,
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr sk_fpfh, pcl::PointCloud<pcl::FPFHSignature33>::Ptr tk_fpfh)
{
	pcl::SampleConsensusInitialAlignment<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33> scia;

	pcl::PointCloud<pcl::PointXYZ>::Ptr s_k1(new pcl::PointCloud < pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr t_k1(new pcl::PointCloud < pcl::PointXYZ>);
	pcl::copyPointCloud(*s_k, *s_k1);
	pcl::copyPointCloud(*t_k, *t_k1);

	scia.setInputSource(s_k1);
	scia.setInputTarget(t_k1);
	scia.setSourceFeatures(sk_fpfh);
	scia.setTargetFeatures(tk_fpfh);
	scia.setMinSampleDistance(3);
	scia.setNumberOfSamples(102);
	scia.setCorrespondenceRandomness(6);
	pcl::PointCloud<pcl::PointXYZ>::Ptr sac_result(new pcl::PointCloud <pcl::PointXYZ>);
	scia.align(*sac_result);
	pcl::transformPointCloud(*cloud, *sac_result, scia.getFinalTransformation());

	return sac_result;
}

int main(int argc, char** argv)
{
	//读取点云数据
	pcl::PointCloud<pcl::PointXYZ>::Ptr source(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr target(new pcl::PointCloud<pcl::PointXYZ>);
	if (pcl::io::loadPCDFile<pcl::PointXYZ>("pig_view1.pcd", *source) == -1)
	{
		PCL_ERROR("点云读取失败\n");
	}
	if (pcl::io::loadPCDFile<pcl::PointXYZ>("pig_view2.pcd", *target) == -1)
	{
		PCL_ERROR("点云读取失败\n");
	}

	//---------------------------体素滤波-----------------------------------
	//pcl::PointCloud<pcl::PointXYZ>::Ptr input_filt(new pcl::PointCloud<pcl::PointXYZ>());
	//pcl::PointCloud<pcl::PointXYZ>::Ptr target_filt(new pcl::PointCloud<pcl::PointXYZ>());

	//voxel_grid(source, input_filt);
	//voxel_grid(target, target_filt);

	//cout << "points size of input_filt:" << input_filt->points.size() << endl;
	//cout << "points size of target_filt:" << target_filt->points.size() << endl;

	//粗配准
	pcl::PointCloud<pcl::PointXYZI>::Ptr s_k(new pcl::PointCloud<pcl::PointXYZI>);
	pcl::PointCloud<pcl::PointXYZI>::Ptr t_k(new pcl::PointCloud<pcl::PointXYZI>);
	extract_keypoint(source, s_k);
	extract_keypoint(target, t_k);

	std::cout << "source特征点数量" << s_k->size() << std::endl;
	std::cout << "target特征点数量" << t_k->size() << std::endl;

	pcl::PointCloud<pcl::FPFHSignature33>::Ptr sk_fpfh = compute_fpfh_feature(s_k);
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr tk_fpfh = compute_fpfh_feature(t_k);

	pcl::PointCloud<pcl::PointXYZ>::Ptr result(new pcl::PointCloud<pcl::PointXYZ>);
	result = sac_align(source, s_k, t_k, sk_fpfh, tk_fpfh);

	//可视化
	boost::shared_ptr<pcl::visualization::PCLVisualizer>viewer(new pcl::visualization::PCLVisualizer("显示点云"));
	viewer->setBackgroundColor(255, 255, 255);
	// 对目标点云着色可视化 (green).
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>target_color(target, 0, 255, 0);
	viewer->addPointCloud<pcl::PointXYZ>(target, target_color, "target cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "target cloud");
	// 对源点云着色可视化 (red).
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>input_color(result, 255, 0, 0);
	viewer->addPointCloud<pcl::PointXYZ>(result, input_color, "input cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "input cloud");

	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100));
	}

	return 0;
}