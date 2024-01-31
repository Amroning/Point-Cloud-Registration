#include<iostream>
#include<pcl/io/pcd_io.h>
#include<pcl/point_types.h>
#include<pcl/keypoints/susan.h>
#include<pcl/features/normal_3d.h>
#include<pcl/features/fpfh_omp.h>
#include<pcl/common/io.h>
#include<pcl/common/common_headers.h>
#include<pcl/visualization/pcl_visualizer.h>
#include<pcl/visualization/cloud_viewer.h>
#include<boost/thread/thread.hpp>
#include<pcl/registration/ia_ransac.h>
using namespace std;

//SUSAN算法提取特征点
void extract_keypoint(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& keypoint)
{
	pcl::StopWatch watch;

	pcl::SUSANKeypoint<pcl::PointXYZ, pcl::PointXYZI> SUSAN;
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZI>::Ptr keypoint1(new pcl::PointCloud<pcl::PointXYZI>());

	//设置算法参数
	SUSAN.setInputCloud(cloud);
	SUSAN.setSearchMethod(tree);

	SUSAN.setRadius(3.0f);                  // 设置正常估计和非最大抑制的半径
	SUSAN.setDistanceThreshold(0.01f);      // 设置距离阈值
	SUSAN.setAngularThreshold(0.01f);       // 设置用于角点检测的角度阈值
	SUSAN.setIntensityThreshold(0.1f);      // 设置用于角点检测的强度阈值
	SUSAN.setNonMaxSupression(true);        // 对响应应用非最大值抑制，以保持最强角。

	//室内点云参数（效果不好，没找到合适的参数）
	//SUSAN.setRadius(0.007f);
	//SUSAN.setDistanceThreshold(0.01f);
	//SUSAN.setAngularThreshold(0.01f);
	//SUSAN.setIntensityThreshold(0.1f);
	//SUSAN.setNonMaxSupression(false);

	SUSAN.compute(*keypoint1);
	keypoint->points.resize(keypoint1->points.size());
	pcl::copyPointCloud(*keypoint1, *keypoint);

	cout << "提取特征点用时： " << watch.getTimeSeconds() << " 秒" << endl;
	cout << "提取特征点数：" << keypoint->points.size() << endl;
}

//计算点云FPFH特征
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

//粗配准
pcl::PointCloud<pcl::PointXYZ>::Ptr sac_align(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr s_k, pcl::PointCloud<pcl::PointXYZ>::Ptr t_k, pcl::PointCloud<pcl::FPFHSignature33>::Ptr sk_fpfh, pcl::PointCloud<pcl::FPFHSignature33>::Ptr tk_fpfh)
{
	pcl::SampleConsensusInitialAlignment<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33> scia;
	scia.setInputSource(s_k);
	scia.setInputTarget(t_k);
	scia.setSourceFeatures(sk_fpfh);
	scia.setTargetFeatures(tk_fpfh);
	scia.setMinSampleDistance(7);///参数：设置采样点之间的最小距离，满足的被当做采样点
	scia.setNumberOfSamples(100);////设置每次迭代设置采样点的个数(这个参数多可以增加配准精度)
	scia.setCorrespondenceRandomness(6);//设置选择随机特征对应点时要使用的邻域点个数。值越大，特征匹配的随机性就越大
	pcl::PointCloud<pcl::PointXYZ>::Ptr sac_result(new pcl::PointCloud<pcl::PointXYZ>);
	scia.align(*sac_result);
	pcl::transformPointCloud(*cloud, *sac_result, scia.getFinalTransformation());
	return sac_result;
}

int main(int argc, char** argv)
{
	//读取点云文件
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

	//关键点可视化
	pcl::PointCloud<pcl::PointXYZ>::Ptr s_k(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr t_k(new pcl::PointCloud<pcl::PointXYZ>);
	extract_keypoint(source, s_k);
	extract_keypoint(target, t_k);
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer1(new pcl::visualization::PCLVisualizer("SUSAN"));
	viewer1->setBackgroundColor(255, 255, 255);
	viewer1->setWindowName("SUSAN关键点提取");
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(source, 0.0, 255, 0.0);
	viewer1->addPointCloud<pcl::PointXYZ>(source, single_color, "source cloud");
	viewer1->addPointCloud<pcl::PointXYZ>(s_k, "key cloud");
	viewer1->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "key cloud");
	viewer1->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "key cloud");

	//粗配准
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr sk_fpfh = compute_fpfh_feature(s_k);
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr tk_fpfh = compute_fpfh_feature(t_k);

	pcl::PointCloud<pcl::PointXYZ>::Ptr result(new pcl::PointCloud<pcl::PointXYZ>);
	result = sac_align(source, s_k, t_k, sk_fpfh, tk_fpfh);

	//结果可视化
	boost::shared_ptr<pcl::visualization::PCLVisualizer>viewer(new pcl::visualization::PCLVisualizer("显示点云"));
	viewer->setBackgroundColor(255, 255, 255);
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