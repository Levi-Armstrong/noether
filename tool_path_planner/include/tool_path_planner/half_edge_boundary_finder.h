/**
 * @author Jorge Nicho
 * @file mesh_boundary_finder.h
 * @date Dec 5, 2019
 * @copyright Copyright (c) 2019, Southwest Research Institute
 *
 * @par License
 * Software License Agreement (Apache License)
 * @par
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * @par
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef INCLUDE_TOOL_PATH_PLANNER_HALF_EDGE_BOUNDARY_FINDER_H_
#define INCLUDE_TOOL_PATH_PLANNER_HALF_EDGE_BOUNDARY_FINDER_H_

#include <boost/optional.hpp>
#include <pcl/PolygonMesh.h>
#include <shape_msgs/Mesh.h>
#include <geometry_msgs/PoseArray.h>

namespace tool_path_planner
{


/**
 * @class tool_path_planner::HalfEdgeBoundaryFinder
 * @details Computes the edges of a mesh by extracting the mesh half edges, needs a mesh that does not have duplicate points
 */
class HalfEdgeBoundaryFinder
{
public:

  enum PointSpacingMethod: int
  {
    NONE = 0,
    MIN_DISTANCE = 1,
    EQUAL_SPACING,
    PARAMETRIC_SPLINE
  };

  struct Config
  {
    std::size_t min_num_points = 200;     /**@brief only edge segments with more than this many points will be returned*/
    bool normal_averaging = true;         /**@brief True in order set the normal of each point as the average of the normal vectors
                                                  of the points within a specified radius*/
    double normal_search_radius = 0.02;   /**@brief The search radius used for normal averaging */
    double normal_influence_weight = 0.5; /**@brief A value [0, 1] that influences the normal averaged based on its distance,
                                                    set to 0 to disable */
    PointSpacingMethod point_spacing_method = PointSpacingMethod::EQUAL_SPACING; /**@brief the method used for spacing the points,
                                                   NONE = 0, MIN_DISTANCE = 1, EQUAL_SPACING = 2, PARAMETRIC_SPLINE = 3 */
    double point_dist = 0.01;         /**@brief point distance parameter used in conjunction with the spacing method */
  };

  HalfEdgeBoundaryFinder();
  virtual ~HalfEdgeBoundaryFinder();


  /**
   * @brief sets the input mesh from which edges are to be generated
   * @param mesh The mesh input
   */
  void setInput(pcl::PolygonMesh::ConstPtr mesh);

  /**
   * @brief sets the input mesh from which edges are to be generated
   * @param mesh The mesh input
   */
  void setInput(const shape_msgs::Mesh& mesh);

  /**
   * @brief Generate the edge poses that follow the contour of the mesh
   * @param config The configuration
   * @return  An array of edge poses or boost::none when it fails.
   */
  boost::optional< std::vector<geometry_msgs::PoseArray> > generate(const HalfEdgeBoundaryFinder::Config& config);

  /**
   * @brief Generate the edge poses that follow the contour of the mesh
   * @param mesh  The input mesh from which edges will be generated
   * @param config The configuration
   * @return  An array of edge poses or boost::none when it fails.
   */
  boost::optional< std::vector<geometry_msgs::PoseArray> > generate(const shape_msgs::Mesh& mesh,
                                                                    const HalfEdgeBoundaryFinder::Config& config);

  /**
   * @brief Generate the edge poses that follow the contour of the mesh
   * @param mesh  The input mesh from which edges will be generated
   * @param config The configuration
   * @return  An array of edge poses or boost::none when it fails.
   */
  boost::optional< std::vector<geometry_msgs::PoseArray> > generate(pcl::PolygonMesh::ConstPtr mesh,
                                                                    const HalfEdgeBoundaryFinder::Config& config);

protected:
  pcl::PolygonMesh::ConstPtr mesh_;
};

} /* namespace tool_path_planner */

#endif /* INCLUDE_TOOL_PATH_PLANNER_HALF_EDGE_BOUNDARY_FINDER_H_ */
