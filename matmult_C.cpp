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
        // A[0][i] = (float)(rand() % 10);
        A[0][i] = (float)i;
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

    int* sendcounts = (int*)calloc(num_nodes, sizeof(int));
    int* displs = (int*)calloc(num_nodes, sizeof(int));
    int* sendcounts_row = (int*)calloc(num_nodes, sizeof(int));
    int* displs_row = (int*)calloc(num_nodes, sizeof(int));

    for (int i = 0; i < num_nodes; ++i) {
        int num_rows_part = MAT_SIZE / num_workers;
        displs[i] = (i - 1) * (num_rows_part * MAT_SIZE);
        if (i == num_workers) {
            num_rows_part += MAT_SIZE % num_workers;
        }
        sendcounts[i] = MAT_SIZE * num_rows_part;
    }
    sendcounts[num_workers] += (MAT_SIZE * MAT_SIZE) % num_workers;

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

        // Broadcast B to all workers
        double start_dist = MPI_Wtime();
        MPI_Bcast(B[0], MAT_SIZE * MAT_SIZE, MPI_FLOAT, MASTER_ID,
                  MPI_COMM_WORLD);

        // Scatter A
        MPI_Scatterv(A[0], sendcounts, displs, MPI_FLOAT, MPI_IN_PLACE, 0,
                     MPI_FLOAT, MASTER_ID, MPI_COMM_WORLD);

        printf("[Distributed]\tDone sending data in %.5f seconds!\n",
               MPI_Wtime() - start_dist);

        // Gather C
        double start_dist_collect = MPI_Wtime();
        MPI_Gatherv(MPI_IN_PLACE, 0, MPI_FLOAT, C_dist[0], sendcounts, displs,
                    MPI_FLOAT, MASTER_ID, MPI_COMM_WORLD);

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

        int num_elements = sendcounts[node_id];
        int displ = displs[node_id];

        // Recieve broadcast of B
        float** B = alloc_mat(MAT_SIZE, MAT_SIZE);
        MPI_Bcast(*B, MAT_SIZE * MAT_SIZE, MPI_FLOAT, MASTER_ID,
                  MPI_COMM_WORLD);

        // Recieve scattered A
        float* part_A = (float*)calloc(num_elements, sizeof(float));
        MPI_Scatterv(MPI_IN_PLACE, sendcounts, displs, MPI_FLOAT, part_A,
                     num_elements, MPI_FLOAT, MASTER_ID, MPI_COMM_WORLD);

        printf("[Distributed#%d]\tDone recieving in %.5f seconds!\n", node_id,
               MPI_Wtime() - start);

        double start_calc = MPI_Wtime();

        // Initialize part of C
        float* part_C = (float*)calloc(num_elements, sizeof(float));
        for (int i = 0; i < num_elements; ++i) {
            part_C[i] = 0;
        }

        // Calculate
        for (int pos_c = 0; pos_c < num_elements; ++pos_c) {
            int i = pos_c / MAT_SIZE;  // row in C
            int j = pos_c % MAT_SIZE;  // col in C
            for (int k = 0; k < MAT_SIZE; ++k) {
                int pos_a = (i * MAT_SIZE) + k;
                part_C[pos_c] += part_A[pos_a] * B[k][j];
            }
        }

        printf("[Serial]\tDone in %.5f seconds!\n", MPI_Wtime() - start);

        printf("[Distributed#%d]\tDone calculating in %.5f seconds!\n", node_id,
               MPI_Wtime() - start_calc);

        // Send C (Gather)
        double start_send = MPI_Wtime();
        MPI_Gatherv(part_C, num_elements, MPI_FLOAT, MPI_IN_PLACE, sendcounts,
                    displs, MPI_FLOAT, MASTER_ID, MPI_COMM_WORLD);
        printf("[Distributed#%d]\tDone sending in %.5f seconds!\n", node_id,
               MPI_Wtime() - start_send);

        // free dynamic memory
        free_mat(B, MAT_SIZE);
        free(part_A);
        free(part_C);
    }

    MPI_Finalize();
    return 0;
}
