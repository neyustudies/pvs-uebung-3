#include <stdio.h>
#include <stdlib.h>

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
    float **A, **B, **C;  // matrices
    int d1, d2, d3;       // dimensions of matrices
    int i, j, k;          // loop variables

    /* print user instruction */
    if (argc != 4) {
        printf("Matrix multiplication: C = A x B\n");
        printf("Usage: %s <NumRowA> <NumColA> <NumColB>\n", argv[0]);
        return 0;
    }

    /* read user input */
    d1 = atoi(argv[1]);  // rows of A and C
    d2 = atoi(argv[2]);  // cols of A and rows of B
    d3 = atoi(argv[3]);  // cols of B and C

    printf("Matrix sizes C[%d][%d] = A[%d][%d] x B[%d][%d]\n", d1, d3, d1, d2,
           d2, d3);

    /* prepare matrices */
    A = alloc_mat(d1, d2);
    init_mat(A, d1, d2);
    B = alloc_mat(d2, d3);
    init_mat(B, d2, d3);
    C = alloc_mat(
        d1, d3);  // no initialisation of C, because it gets filled by matmult

    /* serial version of matmult */
    printf("Perform matrix multiplication...\n");
    for (i = 0; i < d1; i++)
        for (j = 0; j < d3; j++)
            for (k = 0; k < d2; k++)
                C[i][j] += A[i][k] * B[k][j];

    /* test output */
    print_mat(A, d1, d2, "A");
    print_mat(B, d2, d3, "B");
    print_mat(C, d1, d3, "C");

    printf("\nDone.\n");

    /* free dynamic memory */
    free_mat(A, d1);
    free_mat(B, d2);
    free_mat(C, d1);

    return 0;
}
