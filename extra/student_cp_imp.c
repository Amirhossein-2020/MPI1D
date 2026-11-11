#include "utils_1d.h"
#include <stdlib.h>
#include <math.h>

/* Implement the MPI solver here. You may add includes and helper functions.
 * Keep the explicit finite difference scheme and the public signature.
 * MPI is initialized/finalized by the driver. The full result and cumulative
 * error must be available on rank 0 when this function returns.
 */

 /**
 * @brief Solve the heat equation using an explicit finite difference scheme.
 *
 * This functions solves the heat equation iterating over ITER time steps.
 *
 * @param heat_array Array of `double` values representing the temperatures of the points.
 * @param err Pointer to a `double` value representing the error in the simulation.
 * @param num_pts Number of points in the simulation.
 *
 * @return 0 on success, -1 on error.
 */
int heat_eqn_solver_1d(double *heat_array, double *err, size_t num_pts){
  double dx = 1.0/(num_pts+1);
  double dt = 0.5;
  int32_t beg = 1;
  int32_t end = beg + num_pts - 1;

  double *aux = (double *) malloc((num_pts + 2) * sizeof(double));
  double mom_err = *err;

  for (int32_t i = 0; i < ITER; i++){
    aux[beg - 1] = 0.0;
    aux[end + 1] = 0.0;
    for (int32_t j = beg; j <= end; j++){
      aux[j] = dt*(heat_array[j-1] + heat_array[j+1]);
      mom_err += dx*fabs(heat_array[j] - aux[j]);
    }
    void *mom_ptr = (void*)aux;
    aux = heat_array;
    heat_array = (double*)mom_ptr;
  }
  *err = mom_err;
  free(aux);
  return 0;
}
