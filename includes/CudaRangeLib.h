#include <vector>

#ifndef ROS_WORLD_TO_GRID_CONVERSION
#define ROS_WORLD_TO_GRID_CONVERSION 1
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

	bool allocated_weights = false;
};