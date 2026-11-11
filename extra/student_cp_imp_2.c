#include "utils_1d.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#define SIZE_TART 7
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
  double dt = 0.015625;
  int32_t beg = 1;
  int32_t end = beg + num_pts - 1;

  double *aux = (double *) malloc((num_pts + 2) * sizeof(double));
  //double *aux1 = (double *) malloc((num_pts + 2) * sizeof(double));
  //memcpy( aux1, heat_array, num_pts * sizeof(double) );


  double mom_err = *err;

  int32_t i;
  for (i = 0; i < ITER; i += 6){
    aux[beg - 1] = 0.0;
    aux[end + 1] = 0.0;
    aux[ 1 ] = 0.015625 * ( 20.0 * heat_array[ 1 ] + 15.0 * heat_array[ 3 ] + 6.0 * heat_array[ 5 ] + heat_array[ 7 ] );
    mom_err += dx*fabs(heat_array[1] - aux[1]);
    aux[ 2 ] = 0.015625 * ( 20.0 * heat_array[ 2 ] + 15.0 * heat_array[ 4 ] + 6.0 * heat_array[ 6 ] + heat_array[ 8 ] );
    mom_err += dx*fabs(heat_array[2] - aux[2]);
    aux[ 3 ] = 0.015625 * ( 15.0 * heat_array[ 1 ] + 20.0 * heat_array[ 3 ] + 15.0 * heat_array[ 5 ] + 6.0 * heat_array[ 7 ] + heat_array[ 9 ] );
    mom_err += dx*fabs(heat_array[3] - aux[3]);
    aux[ 4 ] = 0.015625 * ( 15.0 * heat_array[ 2 ] + 20.0 * heat_array[ 4 ] + 15.0 * heat_array[ 6 ] + 6.0 * heat_array[ 8 ] + heat_array[ 10 ] );
    mom_err += dx*fabs(heat_array[4] - aux[4]);
    aux[ 5 ] = 0.015625 * ( 6.0 * heat_array[ 1 ] + 15.0 * heat_array[ 3 ] + 20.0 * heat_array[ 5 ] + 15.0 * heat_array[ 7 ] + 6.0 * heat_array[ 9 ] + heat_array[ 11 ] );
    mom_err += dx*fabs(heat_array[5] - aux[5]);
    for (int32_t j = 6; j < end-4; j++){
      aux[ j ] = 0.015625 * ( heat_array[ j - 6 ] + 6.0 * heat_array[ j - 4 ] + 15.0 * heat_array[ j - 2 ] + 20.0 * heat_array[ j ] + 15.0 * heat_array[ j + 2 ] + 6.0 * heat_array[ j + 4 ] + heat_array[ j + 6 ] );
      mom_err += dx*fabs(heat_array[j] - aux[j]);
    }
    aux[ end - 4 ] = 0.015625 * ( heat_array[ end - 10 ] + 6.0 * heat_array[ end - 8 ] + 15.0 * heat_array[ end - 6 ] + 20.0 * heat_array[ end - 4 ] + 15.0 * heat_array[ end - 2 ] + 6.0 * heat_array[ end ] );
    mom_err += dx*fabs(heat_array[end - 4 ] - aux[end - 4 ]);
    aux[ end - 3 ] = 0.015625 * ( heat_array[ end - 9 ] + 6.0 * heat_array[ end - 7 ] + 15.0 * heat_array[ end - 5 ] + 20.0 * heat_array[ end - 3 ] + 15.0 * heat_array[ end - 1 ] );
    mom_err += dx*fabs(heat_array[end - 3] - aux[end - 3]);
    aux[ end - 2 ] = 0.015625 * ( heat_array[ end - 8 ] + 6.0 * heat_array[ end - 6 ] + 15.0 * heat_array[ end - 4 ] + 20.0 * heat_array[ end - 2 ] + 15.0 * heat_array[ end ] );
    mom_err += dx*fabs(heat_array[end - 2] - aux[end - 2]);
    aux[ end - 1 ] = 0.015625 * ( heat_array[ end - 7 ] + 6.0 * heat_array[ end - 5 ] + 15.0 * heat_array[ end - 3 ] + 20.0 * heat_array[ end - 1 ] );
    mom_err += dx*fabs(heat_array[end - 1] - aux[end - 1]);
    aux[ end ] = 0.015625 * ( heat_array[ end - 6 ] + 6.0 * heat_array[ end - 4 ] + 15.0 * heat_array[ end - 2 ] + 20.0 * heat_array[ end ] );
    mom_err += dx*fabs(heat_array[end] - aux[end]);
    void *mom_ptr = (void*)aux;
    aux = heat_array;
    heat_array = (double*)mom_ptr;
  }
  i -= 6;
  for (; i < ITER; i++){
    aux[beg - 1] = 0.0;
    aux[end + 1] = 0.0;
    for (int32_t j = beg; j <= end; j++){
      aux[j] = 0.5*(heat_array[j-1] + heat_array[j+1]);
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
