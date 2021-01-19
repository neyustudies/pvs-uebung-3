#include <assert.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define MAT_SIZE 1000

#define MASTER_ID 0

#define EPSILON 0.0001f

// ---------------------------------------------------------------------------
// allocate space for empty matrix A[row][col]
// access to matrix elements possible with:
// - A[row][col]
// - A[0][row*col]

float** alloc_mat(int row, int col) {
    float **A1, *A2;

    A1 = (float**)calloc(row, sizeof(float*));      // pointer on rows
    A2 = (float*)calloc(row * col, sizeof(float));  // all matrix elements
    for (int i = 0; i < row; i++)
        A1[i] = A2 + i * col;

    return A1;
}

// ---------------------------------------------------------------------------
// random initialisation of matrix with values [0..9]

void init_mat(float** A, int row, int col) {
    for (int i = 0; i < row * col; i++)
        A[0][i] = (float)(rand() % 10);
}

// ---------------------------------------------------------------------------
// DEBUG FUNCTION: printout of all matrix elements

void print_mat(float** A, int row, int col, char const* tag) {
    int i, j;

    printf("Matrix %s:\n", tag);
    for (i = 0; i < row; i++) {
        for (j = 0; j < col; j++)
            printf("%6.1f   ", A[i][j]);
        printf("\n");
    }
}

// ---------------------------------------------------------------------------
// free dynamically allocated memory, which was used to store a 2D matrix
void free_mat(float** A, int num_rows) {
    free(A[0]);  // free contiguous block of float elements (row*col floats)
    free(A);  // free memory for pointers pointing to the beginning of each row
}

// ---------------------------------------------------------------------------
// Check two matrices for equality
bool mat_equal(float** mat1, float** mat2, int m, int n) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (abs(mat1[i][j] - mat2[i][j]) > EPSILON) {
                return false;
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Number of rows of A that will be transferred to the worker.
// Will not always be even, e.g. 333, 333 and 334 for 1000x1000
// matrices and 3 workers.
int calc_num_rows_part(int worker_id, int num_workers) {
    int num_rows_part = MAT_SIZE / num_workers;
    if (worker_id == num_workers) {
        num_rows_part += MAT_SIZE % num_workers;
    }
    return num_rows_part;
}

// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    int node_id, num_nodes, num_workers;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_nodes);
    num_workers = num_nodes - 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &node_id);
    MPI_Status status;

    if (node_id == 0) {
        float **A, **B, **C_ser, **C_dist;  // matrices
        int d1, d2, d3;                     // dimensions of matrices
        int i, j, k;                        // loop variables

        d1 = MAT_SIZE;
        d2 = MAT_SIZE;
        d3 = MAT_SIZE;

        printf("Matrix sizes C[%d][%d] = A[%d][%d] x B[%d][%d]\n", d1, d3, d1,
               d2, d2, d3);

        /* prepare matrices */
        A = alloc_mat(d1, d2);
        init_mat(A, d1, d2);
        B = alloc_mat(d2, d3);
        init_mat(B, d2, d3);

        // no initialisation of C, because it gets filled by matmult
        C_ser = alloc_mat(d1, d3);
        C_dist = alloc_mat(d1, d3);

        /* serial version of matmult */
        printf("Performing serial matrix multiplication...\n");
        double start = MPI_Wtime();
        for (i = 0; i < d1; i++)
            for (j = 0; j < d3; j++)
                for (k = 0; k < d2; k++)
                    C_ser[i][j] += A[i][k] * B[k][j];
        printf("[Serial]\tDone in %.5f seconds!\n", MPI_Wtime() - start);

        printf("Performing parallel matrix multiplication...\n");

        double start_dist = MPI_Wtime();
        // TODO
        printf("[Distributed]\tDone sending data in %.5f seconds!\n",
               MPI_Wtime() - start_dist);

        double start_dist_collect = MPI_Wtime();
        // TODO
        printf("[Distributed]\tDone collecting data in %.5f seconds!\n",
               MPI_Wtime() - start_dist_collect);
        printf("[Distributed]\tDone in a total of %.5f seconds!\n",
               MPI_Wtime() - start_dist);

        assert(mat_equal(C_ser, C_dist, MAT_SIZE, MAT_SIZE));

        printf("\nDone :)\n");

        /* free dynamic memory */
        free_mat(A, d1);
        free_mat(B, d2);
        free_mat(C_ser, d1);
        free_mat(C_dist, d1);

    } else {
        double start = MPI_Wtime();
        // TODO
        printf("[Distributed#%d]\tDone recieving in %.5f seconds!\n", node_id,
               MPI_Wtime() - start);

        double start_calc = MPI_Wtime();
        // TODO
        printf("[Distributed#%d]\tDone calculating in %.5f seconds!\n", node_id,
               MPI_Wtime() - start_calc);

        double start_send = MPI_Wtime();
        // TODO
        printf("[Distributed#%d]\tDone sending in %.5f seconds!\n", node_id,
               MPI_Wtime() - start_send);

        // TODO free dynamic memory
    }

    MPI_Finalize();
    return 0;
}
