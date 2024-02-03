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
	//读取点云数据
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	if (pcl::io::loadPCDFile<pcl::PointXYZ>("pig_view1.pcd", *cloud) == -1)
	{
		PCL_ERROR("点云读取失败\n");
	}

	//计算法线
	pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> n;
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);

	n.setInputCloud(cloud);
	n.setSearchMethod(tree);
	n.setNumberOfThreads(6);
	n.setRadiusSearch(0.3);
	n.compute(*normals);

	//Spin Image图像计算
	pcl::SpinImageEstimation<pcl::PointXYZ, pcl::Normal, pcl::Histogram<153>> spin_image_descriptor(8, 0.05, 10);
	//三个数字分别表示：旋转图像分辨率；最小允许输入点与搜索曲面点之间的法线夹角的余弦值，以便在支撑中保留该点；
	//小支持点数，以正确估计自旋图像。如果在某个点支持包含的点较少，则抛出异常
	pcl::PointCloud<pcl::Histogram<153>>::Ptr spin_images(new pcl::PointCloud<pcl::Histogram<153>>);

	spin_image_descriptor.setInputCloud(cloud);
	spin_image_descriptor.setInputNormals(normals);
	spin_image_descriptor.setSearchMethod(tree);
	spin_image_descriptor.setRadiusSearch(40);
	spin_image_descriptor.compute(*spin_images);

	cout << "Spin Image 输出点数：" << spin_images->points.size() << endl;

	// 显示和检索第一点的自旋图像描述符向量。
	pcl::Histogram<153> first_descriptor = spin_images->points[0];
	cout << first_descriptor << endl;

	//自旋图像描述符可视化
	pcl::visualization::PCLPlotter plotter;
	plotter.addFeatureHistogram(*spin_images, 600);
	plotter.plot();

	return 0;
}