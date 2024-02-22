#include<iostream>
#include<pcl/io/pcd_io.h>
#include<pcl/point_types.h>
#include<pcl/point_cloud.h>
#include<pcl/features/normal_3d_omp.h>
#include<pcl/features/fpfh_omp.h>
#include<pcl/registration/correspondence_estimation.h>
#include<pcl/registration/transformation_estimation_svd.h>
#include<pcl/visualization/pcl_visualizer.h>
#include<boost/thread/thread.hpp>
using namespace std;

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointCloud<pcl::Normal> pointnormal;
typedef pcl::PointCloud<pcl::FPFHSignature33> fpfhFeature;

fpfhFeature::Ptr compute_fpfh_feature(PointCloud::Ptr cloud, pcl::search::KdTree<PointT>::Ptr tree)
{
    //���㷨��
    pcl::NormalEstimationOMP<PointT, pcl::Normal> n;
    pointnormal::Ptr normals(new pointnormal);
    
    n.setInputCloud(cloud);
    n.setSearchMethod(tree);
    n.setNumberOfThreads(12);
    n.setKSearch(20);
    n.compute(*normals);

    //����FPFH
    pcl::FPFHEstimationOMP<PointT, pcl::Normal, pcl::FPFHSignature33> f;
    fpfhFeature::Ptr fpfh(new fpfhFeature);

    f.setInputCloud(cloud);
    f.setInputNormals(normals);
    f.setSearchMethod(tree);
    f.setNumberOfThreads(12);
    f.setKSearch(30);
    f.compute(*fpfh);

    return fpfh;
}

int main(int argc, char** argv)
{
    //��ȡ��������
    PointCloud::Ptr source_cloud(new PointCloud);
    PointCloud::Ptr target_cloud(new PointCloud);
    pcl::io::loadPCDFile<pcl::PointXYZ>("pig_view1.pcd", *source_cloud);
    pcl::io::loadPCDFile<pcl::PointXYZ>("pig_view2.pcd", *target_cloud);

    //����FPFH
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
    fpfhFeature::Ptr source_fpfh = compute_fpfh_feature(source_cloud, tree);
    fpfhFeature::Ptr target_fpfh = compute_fpfh_feature(target_cloud, tree);

    //��FPFH����������Ӧ���ϵ
    pcl::registration::CorrespondenceEstimation<pcl::FPFHSignature33, pcl::FPFHSignature33> crude_cor_est;
    boost::shared_ptr<pcl::Correspondences> cru_correspondences(new pcl::Correspondences);
    crude_cor_est.setInputSource(source_fpfh);
    crude_cor_est.setInputTarget(target_fpfh);
    crude_cor_est.determineCorrespondences(*cru_correspondences, 0.4);

    Eigen::Matrix4f Transform = Eigen::Matrix4f::Identity();
    pcl::registration::TransformationEstimationSVD<PointT, PointT, float>::Ptr trans(new pcl::registration::TransformationEstimationSVD<PointT, PointT, float>);

    trans->estimateRigidTransformation(*source_cloud, *target_cloud, *cru_correspondences, Transform);
    cout << "�任����" << trans << endl;

    //���ӻ�
    boost::shared_ptr<pcl::visualization::PCLVisualizer>viewer(new pcl::visualization::PCLVisualizer("��ʾ����"));
    viewer->setBackgroundColor(255, 255, 255);
    // ��Ŀ�������ɫ���ӻ� (red).
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>target_color(target_cloud, 255, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ>(target_cloud, target_color, "target cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "target cloud");
    // ��Դ������ɫ���ӻ� (green).
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>input_color(source_cloud, 0, 255, 0);
    viewer->addPointCloud<pcl::PointXYZ>(source_cloud, input_color, "input cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "input cloud");
    //��Ӧ��ϵ���ӻ�
    viewer->addCorrespondences<pcl::PointXYZ>(source_cloud, target_cloud, *cru_correspondences, "correspondence");
    //viewer->initCameraParameters();
    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }

    return 0;
}