#include<iostream>
#include<pcl/io/pcd_io.h>
#include<pcl/point_types.h>
#include<pcl/point_cloud.h>
#include<pcl/features/normal_3d_omp.h>
#include<pcl/features/shot_omp.h>
#include<pcl/registration/correspondence_estimation.h>
#include<pcl/registration/transformation_estimation_svd.h>
#include<pcl/visualization/pcl_visualizer.h>
#include<boost/thread/thread.hpp>
using namespace std;

typedef pcl::PointCloud<pcl::PointXYZ> pointcloud;
typedef pcl::PointCloud<pcl::Normal> pointnormal;
typedef pcl::PointCloud<pcl::SHOT352> shotFeature;

shotFeature::Ptr compute_shot_feature(pointcloud::Ptr cloud, pcl::search::KdTree<pcl::PointXYZ>::Ptr tree)
{
    pointnormal::Ptr normals(new pointnormal);
    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> n;
    n.setInputCloud(cloud);
    n.setSearchMethod(tree);
    n.setNumberOfThreads(12);
    n.setKSearch(30);
    n.compute(*normals);

    pcl::SHOTEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::SHOT352> descr_est;
    shotFeature::Ptr shot(new shotFeature);
    descr_est.setInputCloud(cloud);
    descr_est.setInputNormals(normals);
    descr_est.setSearchMethod(tree);
    descr_est.setRadiusSearch(50);
    descr_est.compute(*shot);

    return shot;
}

int main(int argc, char** argv)
{
    //读取点云数据
    pointcloud::Ptr source_cloud(new pointcloud);
    pointcloud::Ptr target_cloud(new pointcloud);
    pcl::io::loadPCDFile<pcl::PointXYZ>("pig_view1.pcd", *source_cloud);
    pcl::io::loadPCDFile<pcl::PointXYZ>("pig_view2.pcd", *target_cloud);

    //SHOT
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    shotFeature::Ptr source_shot = compute_shot_feature(source_cloud, tree);
    shotFeature::Ptr target_shot = compute_shot_feature(target_cloud, tree);

    //根据shot特征构建对应点关系
    pcl::registration::CorrespondenceEstimation<pcl::SHOT352, pcl::SHOT352> crude_cor_est;
    boost::shared_ptr<pcl::Correspondences> cru_correspondences(new pcl::Correspondences);
    crude_cor_est.setInputSource(source_shot);
    crude_cor_est.setInputTarget(target_shot);
    crude_cor_est.determineCorrespondences(*cru_correspondences, 0.1);
    Eigen::Matrix4f Transform = Eigen::Matrix4f::Identity();
    pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ, float>::Ptr trans(new pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ, float>);

    trans->estimateRigidTransformation(*source_cloud, *target_cloud, *cru_correspondences, Transform);

    cout << "变换矩阵为：\n" << Transform << endl;

    //可视化
    boost::shared_ptr<pcl::visualization::PCLVisualizer>viewer(new pcl::visualization::PCLVisualizer("显示点云"));
    viewer->setBackgroundColor(255, 255, 255);
    // 对目标点云着色可视化 (red).
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>target_color(target_cloud, 255, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ>(target_cloud, target_color, "target cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "target cloud");
    // 对源点云着色可视化 (green).
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>input_color(source_cloud, 0, 255, 0);
    viewer->addPointCloud<pcl::PointXYZ>(source_cloud, input_color, "input cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "input cloud");
    //对应关系可视化
    viewer->addCorrespondences<pcl::PointXYZ>(source_cloud, target_cloud, *cru_correspondences, "correspondence");
    //viewer->initCameraParameters();
    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }


    return 0;
}