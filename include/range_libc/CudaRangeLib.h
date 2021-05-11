#include <vector>

#ifndef WORLD_TO_GRID_CONVERSION
#define WORLD_TO_GRID_CONVERSION 1
#endif

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502
#endif

class RayMarchingCUDA
{
public:
	RayMarchingCUDA(std::vector<std::vector<float> > grid, int w, int h, float mr);
	~RayMarchingCUDA();
	void calc_range_many(float *ins, float *outs, int num_casts);
	void numpy_calc_range(float *ins, float *outs, int num_casts);
	void numpy_calc_range_angles(float * ins, float * angles, float * outs, int num_particles, int num_angles);
	void calc_range_repeat_angles_eval_sensor_model(float * ins, float * angles, float * obs, double * weights, int num_particles, int num_angles);
	void set_sensor_table(double *sensor_table, int table_width);
	void set_conversion_params(float x_origin_w, float y_origin_w,
                             float th_w, float sin_th_w, float cos_th_w,
                             float scale_w
                            ) {
			scale_world = scale_w;
			inv_scale_world = 1.0 / scale_w;
			th_world = th_w;
			x_origin_world = x_origin_w;
			y_origin_world = y_origin_w;
			sin_th_world = sin_th_w;
			cos_th_world = cos_th_w;
			rotation_const = th_w - 1.5 * M_PI;
			constants_set = true;
	}

private:
	float *d_ins;
	float *d_outs;
	float *d_distMap;
	double *d_sensorTable;
	double *d_weights;
	int width;
	int height;
	int table_width;
	float max_range;

  // World to grid conversion parameters (usually for ROS use)
  float x_origin_world;   // X translation of origin (cell 0,0) relative to world frame (meters)
  float y_origin_world;   // Y translation of origin (cell 0,0) relative to world frame (meters)
  float th_world;         // Angle relative to world frame
  float sin_th_world;     // Sin of angle relative to world frame
  float cos_th_world;     // Cos of angle relative to world frame
  float scale_world;      // Scale relative to world frame (meters per pixel)
  float inv_scale_world;  // Scale of world relative to map frame (pixels per meter)
  float rotation_const;   // Rotation conversion constant
  bool constants_set = false;

	bool allocated_weights = false;
};