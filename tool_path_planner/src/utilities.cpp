/*
 * Copyright (c) 2018, Southwest Research Institute
 * All rights reserved.*
 * utilities.cpp
 *
 *  Created on: Nov 16, 2018
 *      Author: Jorge Nicho
 */


#include <limits>
#include <cmath>
#include <numeric>

#include <Eigen/Core>
#include <eigen_conversions/eigen_msg.h>

#include <vtkParametricFunctionSource.h>
#include <vtkOBBTree.h>
#include <vtkIntersectionPolyDataFilter.h>
#include <vtkDelaunay2D.h>
#include <vtkMath.h>
#include <vtkSpline.h>
#include <vtkPolyDataNormals.h>
#include <vtkKdTreePointLocator.h>
#include <vtkDoubleArray.h>
#include <vtkPointData.h>
#include <vtkCellData.h>
#include <vtkTriangle.h>
#include <vtkDoubleArray.h>
#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtk_viewer/vtk_utils.h>
#include <vtkReverseSense.h>
#include <vtkImplicitDataSet.h>
#include <vtkCutter.h>
#include <vtkCellLocator.h>
#include <vtkGenericCell.h>
#include <vtkTriangleFilter.h>
#include <tool_path_planner/utilities.h>
#include <console_bridge/console.h>

namespace tool_path_planner
{

//void flipPointOrder(ToolPath &path)
//{
//  vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
//  vtkSmartPointer<vtkPoints> points2 = vtkSmartPointer<vtkPoints>::New();
//  points = path.line->GetPoints();

//  // flip point order
//  for(int i = points->GetNumberOfPoints() - 1; i >= 0; --i)
//  {
//    points2->InsertNextPoint(points->GetPoint(i));
//  }
//  path.line->SetPoints(points2);

//  // flip normal order
//  vtkSmartPointer<vtkDataArray> norms = path.line->GetPointData()->GetNormals();
//  vtkSmartPointer<vtkDoubleArray> new_norms = vtkSmartPointer<vtkDoubleArray>::New();
//  new_norms->SetNumberOfComponents(3);

//  for(int i = norms->GetNumberOfTuples() - 1; i >= 0; --i)
//  {
//    double* ptr = norms->GetTuple(i);
//    new_norms->InsertNextTuple(ptr);
//  }
//  path.line->GetPointData()->SetNormals(new_norms);

//  // flip point order
//  points = path.derivatives->GetPoints();
//  vtkSmartPointer<vtkPoints> dpoints2 = vtkSmartPointer<vtkPoints>::New();
//  for(int i = points->GetNumberOfPoints() - 1; i >= 0; --i)
//  {
//    dpoints2->InsertNextPoint(points->GetPoint(i));
//  }
//  path.derivatives->SetPoints(dpoints2);

//  // flip derivative directions
//  vtkDataArray* ders = path.derivatives->GetPointData()->GetNormals();
//  vtkSmartPointer<vtkDoubleArray> new_ders = vtkSmartPointer<vtkDoubleArray>::New();
//  new_ders->SetNumberOfComponents(3);
//  for(int i = ders->GetNumberOfTuples() -1; i >= 0; --i)
//  {
//    double* pt = ders->GetTuple(i);
//    pt[0] *= -1;
//    pt[1] *= -1;
//    pt[2] *= -1;
//    new_ders->InsertNextTuple(pt);
//  }
//  path.derivatives->GetPointData()->SetNormals(new_ders);

//  // reset points in spline
//  path.spline->SetPoints(points);
//}

tool_path_planner::ToolPathsData toToolPathsData(const tool_path_planner::ToolPaths& paths)
{
  using namespace Eigen;

  tool_path_planner::ToolPathsData results;
  for (const auto& path : paths)
  {
    tool_path_planner::ToolPathData tp_data;
    for(const auto& segment : path)
    {

      tool_path_planner::ToolPathSegmentData tps_data;
      tps_data.line = vtkSmartPointer<vtkPolyData>::New();
      tps_data.derivatives = vtkSmartPointer<vtkPolyData>::New();

      //set vertex (cell) normals
      vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
      vtkSmartPointer<vtkDoubleArray> line_normals = vtkSmartPointer<vtkDoubleArray>::New();
      line_normals->SetNumberOfComponents(3); //3d normals (ie x,y,z)
      line_normals->SetNumberOfTuples(static_cast<long>(segment.size()));
      vtkSmartPointer<vtkDoubleArray> der_normals = vtkSmartPointer<vtkDoubleArray>::New();;
      der_normals->SetNumberOfComponents(3); //3d normals (ie x,y,z)
      der_normals->SetNumberOfTuples(static_cast<long>(segment.size()));

      int idx = 0;
      for(auto& pose : segment)
      {
        Vector3d point, vx, vy, vz;
        point = pose.translation();
        vx = pose.linear().col(0);
        vx *= -1.0;
        vy = pose.linear().col(1);
        vz = pose.linear().col(2);
        points->InsertNextPoint(point.data());
        line_normals->SetTuple(idx,vz.data());
        der_normals->SetTuple(idx,vx.data());

        idx++;
      }
      tps_data.line->SetPoints(points);
      tps_data.line->GetPointData()->SetNormals(line_normals);
      tps_data.derivatives->GetPointData()->SetNormals(der_normals);
      tps_data.derivatives->SetPoints(points);

      tp_data.push_back(tps_data);
    }
    results.push_back(tp_data);
  }
  return results;
}

Eigen::Matrix3d toRotationMatrix(const Eigen::Vector3d& vx, const Eigen::Vector3d& vy,
                                                    const Eigen::Vector3d& vz)
{
  using namespace Eigen;
  Matrix3d rot;
  rot.block(0,0,1,3) = Vector3d(vx.x(), vy.x(), vz.x()).array().transpose();
  rot.block(1,0,1,3) = Vector3d(vx.y(), vy.y(), vz.y()).array().transpose();
  rot.block(2,0,1,3) = Vector3d(vx.z(), vy.z(), vz.z()).array().transpose();
  return rot;
}


bool createPoseArray(const pcl::PointCloud<pcl::PointNormal>& cloud_normals, const std::vector<int>& indices,
                     geometry_msgs::PoseArray& poses)
{
  using namespace pcl;
  using namespace Eigen;

  geometry_msgs::Pose pose_msg;
  Isometry3d pose;
  Vector3d x_dir, z_dir, y_dir;
  std::vector<int> cloud_indices;
  if(indices.empty())
  {
    cloud_indices.resize(cloud_normals.size());
    std::iota(cloud_indices.begin(), cloud_indices.end(), 0);
  }
  else
  {
    cloud_indices.assign(indices.begin(), indices.end());
  }

  for(std::size_t i = 0; i < cloud_indices.size() - 1; i++)
  {
    std::size_t idx_current = cloud_indices[i];
    std::size_t idx_next = cloud_indices[i+1];
    if(idx_current >= cloud_normals.size() || idx_next >= cloud_normals.size())
    {
      CONSOLE_BRIDGE_logError("Invalid indices (current: %lu, next: %lu) for point cloud were passed",
                              idx_current, idx_next);
      return false;
    }
    const PointNormal& p1 = cloud_normals[idx_current];
    const PointNormal& p2 = cloud_normals[idx_next];
    x_dir = (p2.getVector3fMap() - p1.getVector3fMap()).normalized().cast<double>();
    z_dir = Vector3d(p1.normal_x, p1.normal_y, p1.normal_z).normalized();
    y_dir = z_dir.cross(x_dir).normalized();

    pose = Translation3d(p1.getVector3fMap().cast<double>());
    pose.matrix().block<3,3>(0,0) = tool_path_planner::toRotationMatrix(x_dir, y_dir, z_dir);
    tf::poseEigenToMsg(pose,pose_msg);

    poses.poses.push_back(pose_msg);
  }

  // last pose
  pose_msg = poses.poses.back();
  pose_msg.position.x = cloud_normals[cloud_indices.back()].x;
  pose_msg.position.y = cloud_normals[cloud_indices.back()].y;
  pose_msg.position.z = cloud_normals[cloud_indices.back()].z;
  poses.poses.push_back(pose_msg);

  return true;
}

}

