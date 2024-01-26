#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/io.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/features/fpfh_omp.h>  
#include <pcl/common/common_headers.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>
using namespace std;

void voxel_grid(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_filt)
{
	pcl::VoxelGrid<pcl::PointXYZ> vg;
	vg.setInputCloud(cloud);
	vg.setLeafSize(0.01f, 0.01f, 0.01f);
	vg.filter(*cloud_filt);
}

void extract_keypoint(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& keypoint)
{
	pcl::ISSKeypoint3D<pcl::PointXYZ, pcl::PointXYZ> iss;
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	iss.setInputCloud(cloud);
	iss.setSearchMethod(tree);
	iss.setNumberOfThreads(8);     //��ʼ��������������Ҫʹ�õ��߳���
	iss.setSalientRadius(5);  // �������ڼ���Э��������������뾶
	iss.setNonMaxRadius(5);   // ���÷Ǽ���ֵ����Ӧ���㷨�İ뾶
	iss.setThreshold21(0.95);     // �趨�ڶ����͵�һ������ֵ֮�ȵ�����
	iss.setThreshold32(0.95);     // �趨�������͵ڶ�������ֵ֮�ȵ�����
	iss.setMinNeighbors(6);       // ��Ӧ�÷Ǽ���ֵ�����㷨ʱ�����ñ����ҵ�����С�ھ���
	iss.compute(*keypoint);
}

pcl::PointCloud<pcl::FPFHSignature33>::Ptr compute_fpfh_feature(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& keypoint)
{
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree;
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
	int i = 0;
	for (auto p : cloud->points)
	{
		for (auto k : keypoint->points)
		{
			if (k.x == p.x && k.y == p.y && k.z == p.z)
			{
				inliers->indices.push_back(i);
			}
		}
		i++;
	}
	n.setInputCloud(cloud);
	n.setSearchMethod(tree);
	n.setKSearch(10);
	n.compute(*normals);
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh(new pcl::PointCloud<pcl::FPFHSignature33>);
	pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> f;
	f.setIndices(inliers);
	f.setNumberOfThreads(8);
	f.setInputCloud(cloud);
	f.setInputNormals(normals);
	f.setSearchMethod(tree);
	f.setRadiusSearch(50);
	f.compute(*fpfh);

	cout << "feature size is " << fpfh->points.size() << endl;

	return fpfh;
}
pcl::PointCloud<pcl::PointXYZ>::Ptr sac_align(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr s_k, pcl::PointCloud<pcl::PointXYZ>::Ptr t_k, pcl::PointCloud<pcl::FPFHSignature33>::Ptr sk_fpfh, pcl::PointCloud<pcl::FPFHSignature33>::Ptr tk_fpfh)
{
	pcl::SampleConsensusInitialAlignment<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33> scia;
	scia.setInputSource(s_k);
	scia.setInputTarget(t_k);
	scia.setSourceFeatures(sk_fpfh);
	scia.setTargetFeatures(tk_fpfh);
	scia.setMinSampleDistance(7);///���������ò�����֮�����С���룬����ı�����������
	scia.setNumberOfSamples(100);////����ÿ�ε������ò�����ĸ���(������������������׼����)
	scia.setCorrespondenceRandomness(6);//����ѡ�����������Ӧ��ʱҪʹ�õ�����������ֵԽ������ƥ�������Ծ�Խ��
	pcl::PointCloud<pcl::PointXYZ>::Ptr sac_result(new pcl::PointCloud<pcl::PointXYZ>);
	scia.align(*sac_result);
	pcl::transformPointCloud(*cloud, *sac_result, scia.getFinalTransformation());


	return sac_result;
}

int main(int argc, char** argv)
{
	//---------------------------��ȡ����-----------------------------------
	pcl::PointCloud<pcl::PointXYZ>::Ptr input(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr target(new pcl::PointCloud<pcl::PointXYZ>);
	if (pcl::io::loadPCDFile<pcl::PointXYZ>("pig_view1.pcd", *input) == -1)
	{
		PCL_ERROR("���ص���ʧ��\n");
	}
	if (pcl::io::loadPCDFile<pcl::PointXYZ>("pig_view2.pcd", *target) == -1)
	{
		PCL_ERROR("���ص���ʧ��\n");
	}

	//---------------------------�����˲�-----------------------------------
	//pcl::PointCloud<pcl::PointXYZ>::Ptr input_filt(new pcl::PointCloud<pcl::PointXYZ>());
	//pcl::PointCloud<pcl::PointXYZ>::Ptr target_filt(new pcl::PointCloud<pcl::PointXYZ>());

	//voxel_grid(input, input_filt);
	//voxel_grid(target, target_filt);

	//cout << "points size of input_filt:" << input_filt->points.size() << endl;
	//cout << "points size of target_filt:" << target_filt->points.size() << endl;

	//---------------------------��������ȡ-----------------------------------
	pcl::ISSKeypoint3D<pcl::PointXYZ, pcl::PointXYZ> iss;
	pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	iss.setInputCloud(input);
	iss.setSearchMethod(tree);
	iss.setNumberOfThreads(8);     //��ʼ��������������Ҫʹ�õ��߳���
	iss.setSalientRadius(5);  // �������ڼ���Э��������������뾶
	iss.setNonMaxRadius(5);   // ���÷Ǽ���ֵ����Ӧ���㷨�İ뾶
	iss.setThreshold21(0.95);     // �趨�ڶ����͵�һ������ֵ֮�ȵ�����
	iss.setThreshold32(0.95);     // �趨�������͵ڶ�������ֵ֮�ȵ�����
	iss.setMinNeighbors(6);       // ��Ӧ�÷Ǽ���ֵ�����㷨ʱ�����ñ����ҵ�����С�ھ���
	iss.compute(*keypoints);
	cout << "ISS_3D points ����ȡ���Ϊ " << keypoints->points.size() << endl;
	//��������ʾ
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer1(new pcl::visualization::PCLVisualizer("3D ISS"));
	viewer1->setBackgroundColor(255, 255, 255);
	viewer1->setWindowName("ISS�ؼ�����ȡ");
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(input, 0.0, 255, 0.0);
	viewer1->addPointCloud<pcl::PointXYZ>(input, single_color, "sample cloud");
	viewer1->addPointCloud<pcl::PointXYZ>(keypoints, "key cloud");//������
	viewer1->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "key cloud");
	viewer1->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "key cloud");

	//---------------------------����׼-----------------------------------
	pcl::PointCloud<pcl::PointXYZ>::Ptr s_k(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr t_k(new pcl::PointCloud<pcl::PointXYZ>);
	extract_keypoint(input, s_k);
	extract_keypoint(target, t_k);

	cout << "source��ȡ����������" << s_k->points.size() << endl;
	cout << "target��ȡ����������" << t_k->points.size() << endl;

	pcl::PointCloud<pcl::FPFHSignature33>::Ptr sk_fpfh = compute_fpfh_feature(input, s_k);
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr tk_fpfh = compute_fpfh_feature(target, t_k);
	pcl::PointCloud<pcl::PointXYZ>::Ptr result(new pcl::PointCloud<pcl::PointXYZ>);
	result = sac_align(input, s_k, t_k, sk_fpfh, tk_fpfh);

	//����׼
	pcl::PointCloud<pcl::PointXYZ>::Ptr icp_result(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
	icp.setInputSource(result);
	icp.setInputTarget(target);
	icp.setTransformationEpsilon(1e-10);
	icp.setMaxCorrespondenceDistance(1);
	icp.setEuclideanFitnessEpsilon(0.001);
	icp.setMaximumIterations(35);
	icp.setUseReciprocalCorrespondences(true);
	icp.align(*icp_result);
	pcl::transformPointCloud(*result, *icp_result, icp.getFinalTransformation());

	cout << "icp matrix:\n" << icp.getFinalTransformation() << endl;

	////---------------------------��׼���ӻ�-----------------------------------
	boost::shared_ptr<pcl::visualization::PCLVisualizer>viewer(new pcl::visualization::PCLVisualizer("��ʾ����"));
	viewer->setBackgroundColor(255, 255, 255);  //���ñ�����ɫΪ��ɫ
	viewer->setWindowName("����׼");
	// ��Ŀ�������ɫ���ӻ� (red).
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>target_color(target, 255, 0, 0);
	viewer->addPointCloud<pcl::PointXYZ>(target, target_color, "target cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "target cloud");
	// ��Դ������ɫ���ӻ� (green).
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>input_color(icp_result, 0, 255, 0);
	viewer->addPointCloud<pcl::PointXYZ>(icp_result, input_color, "input cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "input cloud");

	while (!viewer1->wasStopped())
	{
		viewer1->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100));
	}

	return 0;
}