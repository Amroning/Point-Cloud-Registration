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

//SUSAN�㷨��ȡ������
void extract_keypoint(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& keypoint)
{
	pcl::StopWatch watch;

	pcl::SUSANKeypoint<pcl::PointXYZ, pcl::PointXYZI> SUSAN;
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZI>::Ptr keypoint1(new pcl::PointCloud<pcl::PointXYZI>());

	//�����㷨����
	SUSAN.setInputCloud(cloud);
	SUSAN.setSearchMethod(tree);

	SUSAN.setRadius(3.0f);                  // �����������ƺͷ�������Ƶİ뾶
	SUSAN.setDistanceThreshold(0.01f);      // ���þ�����ֵ
	SUSAN.setAngularThreshold(0.01f);       // �������ڽǵ���ĽǶ���ֵ
	SUSAN.setIntensityThreshold(0.1f);      // �������ڽǵ����ǿ����ֵ
	SUSAN.setNonMaxSupression(true);        // ����ӦӦ�÷����ֵ���ƣ��Ա�����ǿ�ǡ�

	//���ڵ��Ʋ�����Ч�����ã�û�ҵ����ʵĲ�����
	//SUSAN.setRadius(0.007f);
	//SUSAN.setDistanceThreshold(0.01f);
	//SUSAN.setAngularThreshold(0.01f);
	//SUSAN.setIntensityThreshold(0.1f);
	//SUSAN.setNonMaxSupression(false);

	SUSAN.compute(*keypoint1);
	keypoint->points.resize(keypoint1->points.size());
	pcl::copyPointCloud(*keypoint1, *keypoint);

	cout << "��ȡ��������ʱ�� " << watch.getTimeSeconds() << " ��" << endl;
	cout << "��ȡ����������" << keypoint->points.size() << endl;
}

//�������FPFH����
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

//����׼
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
	//��ȡ�����ļ�
	pcl::PointCloud<pcl::PointXYZ>::Ptr source(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr target(new pcl::PointCloud<pcl::PointXYZ>);
	if (pcl::io::loadPCDFile<pcl::PointXYZ>("pig_view1.pcd", *source) == -1)
	{
		PCL_ERROR("���ص���ʧ��\n");
	}
	if (pcl::io::loadPCDFile<pcl::PointXYZ>("pig_view2.pcd", *target) == -1)
	{
		PCL_ERROR("���ص���ʧ��\n");
	}

	//�ؼ�����ӻ�
	pcl::PointCloud<pcl::PointXYZ>::Ptr s_k(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr t_k(new pcl::PointCloud<pcl::PointXYZ>);
	extract_keypoint(source, s_k);
	extract_keypoint(target, t_k);
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer1(new pcl::visualization::PCLVisualizer("SUSAN"));
	viewer1->setBackgroundColor(255, 255, 255);
	viewer1->setWindowName("SUSAN�ؼ�����ȡ");
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(source, 0.0, 255, 0.0);
	viewer1->addPointCloud<pcl::PointXYZ>(source, single_color, "source cloud");
	viewer1->addPointCloud<pcl::PointXYZ>(s_k, "key cloud");
	viewer1->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "key cloud");
	viewer1->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "key cloud");

	//����׼
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr sk_fpfh = compute_fpfh_feature(s_k);
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr tk_fpfh = compute_fpfh_feature(t_k);

	pcl::PointCloud<pcl::PointXYZ>::Ptr result(new pcl::PointCloud<pcl::PointXYZ>);
	result = sac_align(source, s_k, t_k, sk_fpfh, tk_fpfh);

	//������ӻ�
	boost::shared_ptr<pcl::visualization::PCLVisualizer>viewer(new pcl::visualization::PCLVisualizer("��ʾ����"));
	viewer->setBackgroundColor(255, 255, 255);
	// ��Ŀ�������ɫ���ӻ� (red).
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>target_color(target, 255, 0, 0);
	viewer->addPointCloud<pcl::PointXYZ>(target, target_color, "target cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "target cloud");
	// ��Դ������ɫ���ӻ� (green).
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