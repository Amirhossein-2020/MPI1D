#include "utils_1d.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <sched.h>
#include <string.h>
#include <immintrin.h>
#include <mpi.h>
#include <math.h>
#define MAX( a , b ) ( (a) > (b) ? (a) : (b) )
#define MIN( a , b ) ( (a) < (b) ? (a) : (b) )
#define HALO 32

#if defined(__AVX512F__)
    #define ELEM_D 8
    #define TYPE_VEC_D __m512d
    #define SET1_D( a )  ( _mm512_set1_pd( (a) ) )
    #define LOADU_D( a ) ( _mm512_loadu_pd( (a) ) )
    #define STOREU_D( a, b ) ( _mm512_storeu_pd( (a), (b) ) )
    #define ADD_D( a, b ) ( _mm512_add_pd( (a), (b) ) )
    #define SUB_D( a, b ) ( _mm512_sub_pd( (a), (b) ) )
    #define MUL_D( a, b ) ( _mm512_mul_pd( (a), (b) ) )
    #define ABS_D( a ) ( _mm512_abs_pd( (a) ) )
#elif defined(__AVX2__) || defined(__AVX__)
    #define ELEM_D 4
    #define TYPE_VEC_D __m256d
    #define SET1_D( a )  ( _mm256_set1_pd( (a) ) )
    #define LOADU_D( a ) ( _mm256_loadu_pd( (a) ) )
    #define STOREU_D( a, b ) ( _mm256_storeu_pd( (a), (b) ) )
    #define ADD_D( a, b ) ( _mm256_add_pd( (a), (b) ) )
    #define SUB_D( a, b ) ( _mm256_sub_pd( (a), (b) ) )
    #define MUL_D( a, b ) ( _mm256_mul_pd( (a), (b) ) )
    #define ABS_D( a ) ( _mm256_andnot_pd( _mm256_set1_pd( -0.0 ), (a) ) )
#elif defined(__SSE2__)
    #define ELEM_D 2
    #define TYPE_VEC_D __m128d
    #define SET1_D( a )  ( _mm_set1_pd( (a) ) )
    #define LOADU_D( a ) ( _mm_loadu_pd( (a) ) )
    #define STOREU_D( a, b ) ( _mm_storeu_pd( (a), (b) ) )
    #define ADD_D( a, b ) ( _mm_add_pd( (a), (b) ) )
    #define SUB_D( a, b ) ( _mm_sub_pd( (a), (b) ) )
    #define MUL_D( a, b ) ( _mm_mul_pd( (a), (b) ) )
    #define ABS_D( a ) ( _mm_andnot_pd( _mm_set1_pd( -0.0 ), (a) ) )
#else
    #define ELEM_D 1
    #define TYPE_VEC_D double
    #define SET1_D( a )  ( (a) )
    #define LOADU_D( a ) ( *(a) )
    #define STOREU_D( a, b ) ( *(a) = (b) )
    #define ADD_D( a, b ) ( (a) + (b) )
    #define SUB_D( a, b ) ( (a) - (b) )
    #define MUL_D( a, b ) ( (a) * (b) )
    #define ABS_D( a ) ( fabs( (a) ) )
#endif

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

int heat_eqn_solver_1d( double *heat_array, double *err, size_t num_pts ){
    int32_t size_mpi, rank_mpi;
    double *chunk, *chunk_cp;
    MPI_Comm_size( MPI_COMM_WORLD, &size_mpi );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank_mpi );
    int32_t min_chunk = num_pts + 1 > HALO * size_mpi ? size_mpi : MAX( 1, num_pts / HALO );
    int32_t check_entrance = rank_mpi < min_chunk;
    int32_t check = min_chunk < size_mpi;
    MPI_Comm work_channel = MPI_COMM_WORLD;
    if ( check ) {
        MPI_Comm_split( MPI_COMM_WORLD, rank_mpi < min_chunk, rank_mpi, &work_channel );
        MPI_Comm_size( work_channel, &size_mpi );
        MPI_Comm_rank( work_channel, &rank_mpi );
    }

    if ( check_entrance ) {
        MPI_Request req[ 4 ];
        int32_t check_left  = rank_mpi > 0 ? rank_mpi - 1 : MPI_PROC_NULL;
        int32_t check_right = rank_mpi < size_mpi - 1 ? rank_mpi + 1 : MPI_PROC_NULL;
        int32_t check_left_tag = check_left > - 1 ? check_left : MPI_ANY_TAG;
        int32_t check_right_tag = check_right > - 1 ? check_right : MPI_ANY_TAG;
        int32_t size_chunk = num_pts / size_mpi;
        int32_t extra_chunk = num_pts % size_mpi;
        int32_t global_start = size_chunk * rank_mpi + MIN( rank_mpi, extra_chunk ) + 1;
        int32_t global_end = global_start + size_chunk + ( rank_mpi < extra_chunk ) - 1;
        int32_t size = global_end - global_start + 1;
        int32_t cpy_check = global_start > HALO - 1 ? global_start - HALO : 0;
        int32_t size_check = global_end + HALO < num_pts + 2 ? global_end + HALO - cpy_check + 1 : num_pts + 2 - cpy_check ;
        int32_t local_start = global_start - cpy_check;
        int32_t local_end = local_start + size - 1;
        int32_t recvcounts[ size_mpi ], displs[ size_mpi ];
        if ( rank_mpi == 0 ) {
            recvcounts[ 0 ] = size;
            displs[0] = 1;
            for ( int32_t i = 1; i < size_mpi; i++ ) {
                recvcounts[ i ] = size_chunk + ( i < extra_chunk );
                displs[ i ] = displs[ i - 1 ] + recvcounts[ i - 1 ];
            }
        }

        double dx = 1.0 / ( num_pts + 1 );
        int32_t init = posix_memalign( ( void** )&chunk, CACHE_LINE_SIZE, ( sizeof( double ) * ( size_check ) + CACHE_LINE_SIZE - 1 ) & ~( CACHE_LINE_SIZE - 1 ) );
        if ( init != 0 ) {
            fprintf( stderr, "Error: posix_memalign for chunk failed with code %s. Rank %d\n", strerror( init ), rank_mpi );
            MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
            exit( EXIT_FAILURE );
        }

        init = posix_memalign( ( void** )&chunk_cp, CACHE_LINE_SIZE, ( sizeof( double ) * ( size_check ) + CACHE_LINE_SIZE - 1 ) & ~( CACHE_LINE_SIZE - 1 ) );
        if ( init != 0 ) {
            fprintf( stderr, "Error: posix_memalign for chunk_cp failed with code %s. Rank %d\n", strerror( init ), rank_mpi );
            MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
            exit( EXIT_FAILURE );
        }

        memcpy( ( void* ) chunk, ( void* ) &heat_array[ cpy_check ], sizeof( double ) * ( size_check ) );
        chunk_cp[ 0 ] = 0;
        chunk_cp[ size_check - 1 ] = 0;

        int32_t idx_halo_start = 1;
        int32_t idx_halo_end = size_check - 2;
        TYPE_VEC_D err_vec = SET1_D( 0.0 );
        double err_var = 0;
        int32_t i;
        for ( i = 0; i < ITER - HALO; i += HALO ) {
            int32_t j;
            for ( j = local_start + 1; j < local_end + 1 - ELEM_D; j += ELEM_D ){
                STOREU_D( &chunk_cp[ j ], MUL_D( SET1_D( 0.5 ), ADD_D( LOADU_D( &chunk[ j - 1 ] ), LOADU_D( &chunk[ j + 1 ] ) ) ) );
            }
            for ( ; j < local_end; j++ ) {
                chunk_cp[ j ] = 0.5 * ( chunk[ j - 1 ] + chunk[ j + 1 ] );
            }

            if ( i > 0 ) MPI_Waitall( 4, req, MPI_STATUSES_IGNORE );

            for ( j = idx_halo_start; j < local_start + 2 - ELEM_D; j += ELEM_D ){
                STOREU_D( &chunk_cp[ j ], MUL_D( SET1_D( 0.5 ), ADD_D( LOADU_D( &chunk[ j - 1 ] ), LOADU_D( &chunk[ j + 1 ] ) ) ) );
            }
            for ( ; j < local_start + 1; j++ ) {
                chunk_cp[ j ] = 0.5 * ( chunk[ j - 1 ] + chunk[ j + 1 ] );
            }

            for ( j = local_end; j < idx_halo_end + 2 - ELEM_D; j += ELEM_D ){
                STOREU_D( &chunk_cp[ j ], MUL_D( SET1_D( 0.5 ), ADD_D( LOADU_D( &chunk[ j - 1 ] ), LOADU_D( &chunk[ j + 1 ] ) ) ) );
            }
            for ( ; j < idx_halo_end + 1; j++ ) {
                chunk_cp[ j ] = 0.5 * ( chunk[ j - 1 ] + chunk[ j + 1 ] );
            }

            for ( j = local_start; j < local_end + 2 - ELEM_D; j += ELEM_D ) {
                err_vec = ADD_D( err_vec, ABS_D( SUB_D( LOADU_D( &chunk[ j ] ), LOADU_D( &chunk_cp[ j ] ) ) ) );
            }
            for ( ; j < local_end + 1; j++ ){
                err_var += fabs( chunk[ j ] - chunk_cp[ j ] );
            }

            void *mom_ptr = ( void* )chunk;
            chunk = chunk_cp;
            chunk_cp = ( double* )mom_ptr;
            idx_halo_start = rank_mpi > 0 ? idx_halo_start + 1 : 1;
            idx_halo_end = rank_mpi < size_mpi - 1 ? idx_halo_end - 1 : size_check - 2;

            for ( int32_t z = 0; z < HALO - 1; z++ ) {
                int32_t u;
                for ( u = idx_halo_start; u < idx_halo_end + 2 - ELEM_D; u += ELEM_D ){
                    STOREU_D( &chunk_cp[ u ], MUL_D( SET1_D( 0.5 ), ADD_D( LOADU_D( &chunk[ u - 1 ] ), LOADU_D( &chunk[ u + 1 ] ) ) ) );
                }
                for ( ; u < idx_halo_end + 1; u++ ) {
                    chunk_cp[ u ] = 0.5 * ( chunk[ u - 1 ] + chunk[ u + 1 ] );
                }
                for ( u = local_start; u < local_end + 2 - ELEM_D; u += ELEM_D ) {
                    err_vec = ADD_D( err_vec, ABS_D( SUB_D( LOADU_D( &chunk[ u ] ), LOADU_D( &chunk_cp[ u ] ) ) ) );
                }
                for ( ; u < local_end + 1; u++ ){
                    err_var += fabs( chunk[ u ] - chunk_cp[ u ] );
                }
                void *mom_ptr = ( void* )chunk;
                chunk = chunk_cp;
                chunk_cp = ( double* )mom_ptr;
                idx_halo_start = rank_mpi > 0 ? idx_halo_start + 1 : 1;
                idx_halo_end = rank_mpi < size_mpi - 1 ? idx_halo_end - 1 : size_check - 2;
            }

            MPI_Isend( &chunk[ local_start ], HALO, MPI_DOUBLE, check_left, rank_mpi, work_channel, &req[ 0 ] );
            MPI_Isend( &chunk[ local_end - HALO + 1 ], HALO, MPI_DOUBLE, check_right, rank_mpi, work_channel, &req[ 1 ] );
            MPI_Irecv( &chunk[ 0 ], HALO, MPI_DOUBLE, check_left, check_left_tag, work_channel, &req[ 2 ] );
            MPI_Irecv( &chunk[ local_end + 1 ], HALO, MPI_DOUBLE, check_right, check_right_tag, work_channel, &req[ 3 ] );

            idx_halo_start = 1;
            idx_halo_end = size_check - 2;
        }

        idx_halo_start = rank_mpi > 0 ? local_start - ( ITER - i ) + 1 : 1;
        idx_halo_end = rank_mpi < size_mpi - 1 ? local_end + ( ITER - i ) - 1 : size_check - 2;
        int32_t j;
        for ( j = local_start + 1; j < local_end + 1 - ELEM_D; j += ELEM_D ){
            STOREU_D( &chunk_cp[ j ], MUL_D( SET1_D( 0.5 ), ADD_D( LOADU_D( &chunk[ j - 1 ] ), LOADU_D( &chunk[ j + 1 ] ) ) ) );
        }
        for ( ; j < local_end; j++ ) {
            chunk_cp[ j ] = 0.5 * ( chunk[ j - 1 ] + chunk[ j + 1 ] );
        }

        if ( i > 0 ) MPI_Waitall( 4, req, MPI_STATUSES_IGNORE );

        for ( j = idx_halo_start; j < local_start + 2 - ELEM_D; j += ELEM_D ){
            STOREU_D( &chunk_cp[ j ], MUL_D( SET1_D( 0.5 ), ADD_D( LOADU_D( &chunk[ j - 1 ] ), LOADU_D( &chunk[ j + 1 ] ) ) ) );
        }
        for ( ; j < local_start + 1; j++ ) {
            chunk_cp[ j ] = 0.5 * ( chunk[ j - 1 ] + chunk[ j + 1 ] );
        }

        for ( j = local_end; j < idx_halo_end + 2 - ELEM_D; j += ELEM_D ){
            STOREU_D( &chunk_cp[ j ], MUL_D( SET1_D( 0.5 ), ADD_D( LOADU_D( &chunk[ j - 1 ] ), LOADU_D( &chunk[ j + 1 ] ) ) ) );
        }
        for ( ; j < idx_halo_end + 1; j++ ) {
            chunk_cp[ j ] = 0.5 * ( chunk[ j - 1 ] + chunk[ j + 1 ] );
        }

        for ( j = local_start; j < local_end + 2 - ELEM_D; j += ELEM_D ) {
            err_vec = ADD_D( err_vec, ABS_D( SUB_D( LOADU_D( &chunk[ j ] ), LOADU_D( &chunk_cp[ j ] ) ) ) );
        }
        for ( ; j < local_end + 1; j++ ){
            err_var += fabs( chunk[ j ] - chunk_cp[ j ] );
        }

        void *mom_ptr = ( void* )chunk;
        chunk = chunk_cp;
        chunk_cp = ( double* )mom_ptr;
        idx_halo_start = rank_mpi > 0 ? idx_halo_start + 1 : 1;
        idx_halo_end = rank_mpi < size_mpi - 1 ? idx_halo_end - 1 : size_check - 2;

        int32_t end = ITER - i - 1;
        for ( int32_t z = 0; z < end; z++ ) {
            int32_t u;
            for ( u = idx_halo_start; u < idx_halo_end + 2 - ELEM_D; u += ELEM_D ){
                STOREU_D( &chunk_cp[ u ], MUL_D( SET1_D( 0.5 ), ADD_D( LOADU_D( &chunk[ u - 1 ] ), LOADU_D( &chunk[ u + 1 ] ) ) ) );
            }
            for ( ; u < idx_halo_end + 1; u++ ) {
                chunk_cp[ u ] = 0.5 * ( chunk[ u - 1 ] + chunk[ u + 1 ] );
            }
            for ( u = local_start; u < local_end + 2 - ELEM_D; u += ELEM_D ) {
                err_vec = ADD_D( err_vec, ABS_D( SUB_D( LOADU_D( &chunk[ u ] ), LOADU_D( &chunk_cp[ u ] ) ) ) );
            }
            for ( ; u < local_end + 1; u++ ){
                err_var += fabs( chunk[ u ] - chunk_cp[ u ] );
            }
            void *mom_ptr = ( void* )chunk;
            chunk = chunk_cp;
            chunk_cp = ( double* )mom_ptr;
            idx_halo_start = rank_mpi > 0 ? idx_halo_start + 1 : 1;
            idx_halo_end = rank_mpi < size_mpi - 1 ? idx_halo_end - 1 : size_check - 2;
        }

        double buff[ ELEM_D ];
        STOREU_D( buff, err_vec );
        for ( j = 0; j < ELEM_D; j++ ) {
            err_var += buff[ j ];
        }
        if ( rank_mpi == 0 ) MPI_Reduce( MPI_IN_PLACE, &err_var, 1, MPI_DOUBLE, MPI_SUM, 0, work_channel );
        else MPI_Reduce( &err_var, NULL, 1, MPI_DOUBLE, MPI_SUM, 0, work_channel );
        MPI_Gatherv( ( void* ) &chunk[ local_start ], size, MPI_DOUBLE, ( void* ) heat_array, recvcounts, displs, MPI_DOUBLE, 0, work_channel );

        *err = dx * err_var;
        free( chunk );
        free( chunk_cp );
    }
    if ( check ) MPI_Comm_free( &work_channel );
    return 0;
}
