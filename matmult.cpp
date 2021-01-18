#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define MAT_SIZE 1000

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
    int nodeID, numNodes;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numNodes);
    MPI_Comm_rank(MPI_COMM_WORLD, &nodeID);
    MPI_Status status;

    if (nodeID == 0) {
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

        for (int worker_id = 1; worker_id < numNodes; ++worker_id) {
            float* buf = (float*)calloc(1, sizeof(float));
            *buf = 42;
            MPI_Send(buf, 1, MPI_FLOAT, worker_id, 0, MPI_COMM_WORLD);
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
        printf("Waiting for stuff to do (%d/%d)...\n", nodeID, numNodes);

        float* buf = (float*)calloc(1, sizeof(float));
        MPI_Recv(buf, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);

        printf("Is this the answer? %.f\n", *buf);
    }

    MPI_Finalize();
    return 0;
}
