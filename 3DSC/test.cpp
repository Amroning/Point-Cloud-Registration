#include<iostream>
#include<pcl/point_types.h>
#include<pcl/point_cloud.h>
#include<pcl/io/pcd_io.h>
#include<pcl/filters/voxel_grid.h>
#include<pcl/filters/random_sample.h>
#include<pcl/keypoints/iss_3d.h>
#include<pcl/features/normal_3d_omp.h>
#include<pcl/features/3dsc.h>
#include<pcl/search/kdtree.h>
#include<pcl/registration/ia_ransac.h>
#include<pcl/registration/icp.h>
#include<pcl/visualization/pcl_visualizer.h>
#include<boost/thread/thread.hpp>
#include<time.h>
using namespace std;

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;

void extract_keypoint(PointCloud::Ptr& cloud, PointCloud::Ptr& keypoint)
{
	pcl::ISSKeypoint3D<PointT, PointT> iss;
	pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);

	iss.setInputCloud(cloud);
	iss.setSearchMethod(tree);
	iss.setNumberOfThreads(8);
	iss.setSalientRadius(5);
	iss.setNonMaxRadius(5);
	iss.setThreshold21(0.95);
	iss.setThreshold32(0.95);
	iss.setMinNeighbors(6);
	iss.compute(*keypoint);
}

void computeKeypoints3DSC(PointCloud::Ptr& cloud, PointCloud::Ptr& keypoint, pcl::PointCloud<pcl::ShapeContext1980>::Ptr& dsc)
{
	//���㷨��
	pcl::NormalEstimationOMP<PointT, pcl::Normal> n;
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());

	n.setInputCloud(keypoint);
	n.setNumberOfThreads(6);
	n.setSearchMethod(tree);
	n.setSearchSurface(cloud);
	n.setKSearch(10);
	n.compute(*normals);

	//����3DSC
	pcl::ShapeContext3DEstimation<PointT, pcl::Normal, pcl::ShapeContext1980> sc;
	sc.setInputCloud(keypoint);
	sc.setInputNormals(normals);
	sc.setSearchMethod(tree);
	sc.setMinimalRadius(0.8);
	sc.setRadiusSearch(40);
	sc.setPointDensityRadius(8);
	sc.compute(*dsc);
}

void visualize_pointcloud(PointCloud::Ptr pcd_src, PointCloud::Ptr pcd_tgt, PointCloud::Ptr pcd_final)
{
	//������ʼ��Ŀ��
	pcl::visualization::PCLVisualizer viewer("registration Viewer");
	pcl::visualization::PointCloudColorHandlerCustom<PointT> src_h(pcd_src, 0, 255, 0);
	pcl::visualization::PointCloudColorHandlerCustom<PointT> tgt_h(pcd_tgt, 255, 0, 0);
	pcl::visualization::PointCloudColorHandlerCustom<PointT> final_h(pcd_final, 0, 255, 0);
	viewer.setBackgroundColor(255, 255, 255);
	viewer.addPointCloud(pcd_src, src_h, "source cloud");
	viewer.addPointCloud(pcd_tgt, tgt_h, "tgt cloud");
	viewer.addPointCloud(pcd_final, final_h, "final cloud");

	while (!viewer.wasStopped())
	{
		viewer.spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}

int main(int argc, char** argv)
{
	//���ص�������
	PointCloud::Ptr source(new PointCloud);
	PointCloud::Ptr target(new PointCloud);
	if (pcl::io::loadPCDFile<PointT>("pig_view1.pcd", *source) == -1)
	{
		PCL_ERROR("Դ���ƶ�ȡʧ��\n");
	}
	if (pcl::io::loadPCDFile<PointT>("pig_view2.pcd", *target) == -1)
	{
		PCL_ERROR("Ŀ����ƶ�ȡʧ��\n");
	}

	clock_t start = clock();

	// ��������̶��ĵ���������Ϊ������
	/*pcl::RandomSample<PointT> rs_src;
	rs_src.setInputCloud(source);
	rs_src.setSample(10000);
	PointCloud::Ptr key_src(new PointCloud);
	rs_src.filter(*key_src);
	pcl::RandomSample<PointT> rs_tgt;
	rs_tgt.setInputCloud(target);
	rs_tgt.setSample(10000);
	PointCloud::Ptr key_tgt(new PointCloud);
	rs_tgt.filter(*key_tgt);*/

	//��ȡ�����㣨ISS��
	PointCloud::Ptr key_src(new PointCloud);
	PointCloud::Ptr key_tgt(new PointCloud);
	extract_keypoint(source, key_src);
	extract_keypoint(target, key_tgt);

	cout << "Դ������ȡ����������" << key_src->points.size() << endl;
	cout << "Ŀ�������ȡ����������" << key_tgt->points.size() << endl;

	//����3DSC
	pcl::PointCloud<pcl::ShapeContext1980>::Ptr sps_src(new pcl::PointCloud<pcl::ShapeContext1980>);
	pcl::PointCloud<pcl::ShapeContext1980>::Ptr sps_tgt(new pcl::PointCloud<pcl::ShapeContext1980>);
	computeKeypoints3DSC(source, key_src, sps_src);
	computeKeypoints3DSC(target, key_tgt, sps_tgt);
	cout << *sps_src << endl;
	cout << *sps_tgt << endl;

	//SAC����׼
	pcl::SampleConsensusInitialAlignment<PointT, PointT, pcl::ShapeContext1980> scia;
	PointCloud::Ptr sac_result(new PointCloud);

	scia.setInputSource(key_src);
	scia.setInputTarget(key_tgt);
	scia.setSourceFeatures(sps_src);
	scia.setTargetFeatures(sps_tgt);
	scia.setMinSampleDistance(7);
	scia.setNumberOfSamples(100);
	scia.setCorrespondenceRandomness(6);
	scia.align(*sac_result);
	cout << "sac has converged:" << scia.hasConverged() << "  score:" << scia.getFitnessScore() << endl;
	Eigen::Matrix4f sac_trans;
	sac_trans = scia.getFinalTransformation();
	cout << "SAC Matrix:" << endl << sac_trans << endl;

	pcl::transformPointCloud(*source, *sac_result, scia.getFinalTransformation());

	clock_t sac_time = clock();

	//ICP����׼
	pcl::IterativeClosestPoint<PointT, PointT> icp;
	PointCloud::Ptr icp_result(new PointCloud);

	icp.setInputSource(key_src);
	icp.setInputTarget(key_tgt);
	icp.setMaxCorrespondenceDistance(20);
	icp.setMaximumIterations(35);
	icp.setTransformationEpsilon(1e-10);
	icp.setEuclideanFitnessEpsilon(0.01);
	icp.align(*icp_result, sac_trans);

	clock_t end = clock();

	cout << "��ʱ�䣺" << (double)(end - start) / (double)CLOCKS_PER_SEC << "s" << endl;
	cout << "SAC��׼ʱ��" << (double)(sac_time - start) / (double)CLOCKS_PER_SEC << "s" << endl;
	cout << "ICP��׼ʱ��" << (double)(end - sac_time) / (double)CLOCKS_PER_SEC << "s" << endl;

	cout << "ICP has converged:" << icp.hasConverged() << "  score:" << icp.getFitnessScore() << endl;
	Eigen::Matrix4f icp_trans;
	icp_trans = icp.getFinalTransformation();
	cout << "ICP Matrix:" << endl << icp_trans << endl;

	pcl::transformPointCloud(*source, *icp_result, icp_trans);

	//���ӻ�
	visualize_pointcloud(source, target, icp_result);

	return 0;
}

//����̫��  ��û�г������ڵ���������