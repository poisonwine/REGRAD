#include <iostream>              //标准C++库中的输入输出的头文件
#include <pcl/io/pcd_io.h>           //PCD读写类相关的头文件
#include <pcl/point_types.h>      //PCL中支持的点类型的头文件

 
int main (int argc, char** argv)
{
    pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBL>);
    if (pcl::io::loadPCDFile<pcl::PointXYZRGBL>("/home/lucky/gym/bullet3/my_pubullet_code/point/00_noisy00000.pcd", *cloud) == -1) {
        return(-1);
    }
    //for (size_t i = 0; i < cloud->points.size(); ++i) {
    //    std::cout << cloud->points[i].r << cloud->points[i].g << cloud->points[i].b<<" "<<cloud->points[i].label<<" ";//<< std::endl;
    //}



    return 0;
}

