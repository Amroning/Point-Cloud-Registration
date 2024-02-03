#include<iostream>
#include<pcl/point_types.h>
#include<pcl/point_cloud.h>
#include<pcl/io/pcd_io.h>
#include<pcl/search/kdtree.h>
#include<pcl/features/normal_3d_omp.h>
#include<pcl/features/spin_image.h>
#include<pcl/filters/voxel_grid.h>
#include<pcl/visualization/pcl_plotter.h>
#include<boost/thread/thread.hpp>
using namespace std;

int main()
{
	//��ȡ��������
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	if (pcl::io::loadPCDFile<pcl::PointXYZ>("pig_view1.pcd", *cloud) == -1)
	{
		PCL_ERROR("���ƶ�ȡʧ��\n");
	}

	//���㷨��
	pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> n;
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);

	n.setInputCloud(cloud);
	n.setSearchMethod(tree);
	n.setNumberOfThreads(6);
	n.setRadiusSearch(0.3);
	n.compute(*normals);

	//Spin Imageͼ�����
	pcl::SpinImageEstimation<pcl::PointXYZ, pcl::Normal, pcl::Histogram<153>> spin_image_descriptor(8, 0.05, 10);
	//�������ֱַ��ʾ����תͼ��ֱ��ʣ���С��������������������֮��ķ��߼нǵ�����ֵ���Ա���֧���б����õ㣻
	//С֧�ֵ���������ȷ��������ͼ�������ĳ����֧�ְ����ĵ���٣����׳��쳣
	pcl::PointCloud<pcl::Histogram<153>>::Ptr spin_images(new pcl::PointCloud<pcl::Histogram<153>>);

	spin_image_descriptor.setInputCloud(cloud);
	spin_image_descriptor.setInputNormals(normals);
	spin_image_descriptor.setSearchMethod(tree);
	spin_image_descriptor.setRadiusSearch(40);
	spin_image_descriptor.compute(*spin_images);

	cout << "Spin Image ���������" << spin_images->points.size() << endl;

	// ��ʾ�ͼ�����һ�������ͼ��������������
	pcl::Histogram<153> first_descriptor = spin_images->points[0];
	cout << first_descriptor << endl;

	//����ͼ�����������ӻ�
	pcl::visualization::PCLPlotter plotter;
	plotter.addFeatureHistogram(*spin_images, 600);
	plotter.plot();

	return 0;
}