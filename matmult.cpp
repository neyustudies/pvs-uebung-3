#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define MAT_SIZE 100

#define MASTER_ID 0

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

int main(int argc, char* argv[]) {
    int node_id, num_nodes, num_workers;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_nodes);
    num_workers = num_nodes - 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &node_id);
    MPI_Status status;

    if (node_id == 0) {
        float **A, **B, **C_ser, **C_par;  // matrices
        int d1, d2, d3;       // dimensions of matrices
        int i, j, k;          // loop variables

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
        printf("Master:  B[4][2] is:  %.f\n", B[4][2]);
        // no initialisation of C, because it gets filled by matmult
        C_ser = alloc_mat(d1, d3);
        C_par = alloc_mat(d1, d3);

        /* serial version of matmult */
        printf("Perform serial matrix multiplication...\n");
        double start = MPI_Wtime();
        for (i = 0; i < d1; i++)
            for (j = 0; j < d3; j++)
                for (k = 0; k < d2; k++)
                    C_ser[i][j] += A[i][k] * B[k][j];
        printf("Done in %.3f seconds!\n", MPI_Wtime() - start);

        // Send B and relevant parts of A to workers.
        for (int worker_id = 1; worker_id < num_nodes; ++worker_id) {
            // Send B.
            MPI_Send(*B, MAT_SIZE * MAT_SIZE, MPI_FLOAT, worker_id, 0,
                     MPI_COMM_WORLD);

            // Number of rows of A that will be transferred to the worker.
            // Will not always be even, e.g. 333, 333 and 334 for 1000x1000
            // matrices and 3 workers.
            int n_rows_a = MAT_SIZE / num_workers;
            if (worker_id == num_workers) {
                n_rows_a += MAT_SIZE % num_workers;
            }
            MPI_Send(&n_rows_a, 1, MPI_FLOAT, worker_id, 0, MPI_COMM_WORLD);

            float** part_A = alloc_mat(n_rows_a, MAT_SIZE);
            int row_offset = (worker_id - 1) * (MAT_SIZE / num_workers);
            for (int row = 0; row < n_rows_a; ++row) {
                for (int col = 0; col < MAT_SIZE; ++col) {
                    part_A[row][col] = A[row + row_offset][col];
                }
            }
            // Send part_A.
            MPI_Send(*part_A, n_rows_a * MAT_SIZE, MPI_FLOAT, worker_id, 0,
                     MPI_COMM_WORLD);

            // float* buf_A = (float*)calloc(n_rows_a, sizeof(float));
            //*buf = 42;
            // MPI_Send(buf, 1, MPI_FLOAT, worker_id, 0, MPI_COMM_WORLD);
        }

        /* test output */
        // print_mat(A, d1, d2, "A");
        // print_mat(B, d2, d3, "B");
        // print_mat(C, d1, d3, "C");

        printf("\nDone.\n");

        /* free dynamic memory */
        free_mat(A, d1);
        free_mat(B, d2);
        free_mat(C_ser, d1);
        free_mat(C_par, d1);

    } else {
        printf("Waiting for stuff to do (%d/%d)...\n", node_id, num_nodes);

        float** B = alloc_mat(MAT_SIZE, MAT_SIZE);
        MPI_Recv(*B, MAT_SIZE * MAT_SIZE, MPI_FLOAT, MASTER_ID, 0,
                 MPI_COMM_WORLD, &status);

        printf("Worker: B[4][2] is:  %.f\n", B[4][2]);

        int n_rows_a;
        MPI_Recv(&n_rows_a, 1, MPI_INT, MASTER_ID, 0, MPI_COMM_WORLD, &status);

        printf("Will get %d rows of A (%d/%d)...\n", n_rows_a, node_id,
               num_nodes);

        float** part_A = alloc_mat(n_rows_a, MAT_SIZE);
        MPI_Recv(*part_A, n_rows_a * MAT_SIZE, MPI_FLOAT, MASTER_ID, 0,
                 MPI_COMM_WORLD, &status);

        float** part_C = alloc_mat(n_rows_a, MAT_SIZE);

        for (int i = 0; i < n_rows_a; ++i) {
            for (int j = 0; j < MAT_SIZE; ++j) {
                for (int k = 0; k < MAT_SIZE; ++k) {
                    part_C[i][j] += part_A[i][k] * B[k][j];
                }
            }
        }

        // printf("Is this the answer? %.f\n", *buf);
    }

    MPI_Finalize();
    return 0;
}
