#include<iostream>
#include<pcl/io/pcd_io.h>
#include<pcl/point_types.h>
#include<pcl/point_cloud.h>
#include<pcl/features/normal_3d_omp.h>
#include<pcl/features/pfh.h>
#include<pcl/keypoints/sift_keypoint.h>
#include<pcl/registration/correspondence_estimation.h>
#include<pcl/registration/transformation_estimation_svd.h>
#include<pcl/visualization/pcl_visualizer.h>
#include<boost/thread/thread.hpp>
using namespace std;

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointCloud <pcl::PFHSignature125> pfhFeature;

namespace pcl
{
	template<>
	struct SIFTKeypointFieldSelector<PointXYZ>
	{
		inline float operator()(const PointXYZ& p) const
		{
			return p.z;
		}

	};
}

void extract_keypoint(PointCloud::Ptr& cloud, PointCloud::Ptr& keypoint)
{
	const float min_scale = 5.f;
	const int n_octaves = 3;
	const int n_scales_per_octave = 15;
	const float min_contrast = 0.01f;

	pcl::SIFTKeypoint<PointT, pcl::PointWithScale>sift;
	pcl::PointCloud<pcl::PointWithScale>result;
	pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);

	sift.setInputCloud(cloud);
	sift.setSearchMethod(tree);
	sift.setScales(min_scale, n_octaves, n_scales_per_octave);
	sift.setMinimumContrast(min_contrast);
	sift.compute(result);

	cout << "提取 " << result.size() << "  特征点" << endl;
	pcl::copyPointCloud(result, *keypoint);
}

pfhFeature::Ptr compute_pfh_feature(PointCloud::Ptr input_cloud, pcl::search::KdTree<PointT>::Ptr tree)
{
	//计算法线
	pcl::NormalEstimationOMP<PointT, pcl::Normal>n;
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	
	n.setInputCloud(input_cloud);
	n.setSearchMethod(tree);
	n.setNumberOfThreads(5);
	n.setKSearch(10);
	n.compute(*normals);

	//计算PFH
	pcl::PFHEstimation<PointT, pcl::Normal, pcl::PFHSignature125> pfh;
	pcl::PointCloud<pcl::PFHSignature125>::Ptr pfh_result(new pcl::PointCloud<pcl::PFHSignature125>());

	pfh.setInputCloud(input_cloud);
	pfh.setInputNormals(normals);
	pfh.setSearchMethod(tree);
	pfh.setRadiusSearch(10);
	pfh.compute(*pfh_result);

	return pfh_result;
}

int main(int argc, char** argv)
{
	//读取点云数据
	PointCloud::Ptr source(new PointCloud);
	PointCloud::Ptr target(new PointCloud);
	if (pcl::io::loadPCDFile<PointT>("pig_view1.pcd", *source) == -1)
	{
		PCL_ERROR("源点云读取失败\n");
	}
	if (pcl::io::loadPCDFile<PointT>("pig_view2.pcd", *target) == -1)
	{
		PCL_ERROR("目标点云读取失败\n");
	}

	//提取特征点
	PointCloud::Ptr s_k(new PointCloud);
	PointCloud::Ptr t_k(new PointCloud);
	extract_keypoint(source, s_k);
	extract_keypoint(target, t_k);

	//计算PFH特征
	pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
	pfhFeature::Ptr source_pfh = compute_pfh_feature(s_k, tree);
	pfhFeature::Ptr target_pfh = compute_pfh_feature(t_k, tree);

	pcl::registration::CorrespondenceEstimation<pcl::PFHSignature125, pcl::PFHSignature125> crude_cor_est;
	boost::shared_ptr<pcl::Correspondences> cru_correspondences(new pcl::Correspondences);
	crude_cor_est.setInputSource(source_pfh);
	crude_cor_est.setInputTarget(target_pfh);
	crude_cor_est.determineCorrespondences(*cru_correspondences);

	Eigen::Matrix4f Transform = Eigen::Matrix4f::Identity();
	pcl::registration::TransformationEstimationSVD<PointT, PointT, float>::Ptr trans(new pcl::registration::TransformationEstimationSVD<PointT, PointT, float>);

	trans->estimateRigidTransformation(*source, *target, *cru_correspondences, Transform);

	cout << Transform << endl;

	boost::shared_ptr<pcl::visualization::PCLVisualizer>viewer(new pcl::visualization::PCLVisualizer("viewer"));
	viewer->setBackgroundColor(255, 255, 255);
	
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>target_color(target, 255, 0, 0);
	viewer->addPointCloud<pcl::PointXYZ>(target, target_color, "target cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "target cloud");

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>input_color(source, 0, 255, 0);
	viewer->addPointCloud<pcl::PointXYZ>(source, input_color, "input cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "input cloud");

	viewer->addCorrespondences<pcl::PointXYZ>(s_k, t_k, *cru_correspondences, "correspondence");

	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}