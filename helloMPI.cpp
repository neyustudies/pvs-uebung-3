#include<stdio.h>
#include<mpi.h>	// include MPI header file
 
int main(int argc, char** argv) {
	int nodeID, numNodes;

	/* MPI environment */
	MPI_Init(&argc, &argv);										// initializes MPI library, first MPI function call
	MPI_Comm_size(MPI_COMM_WORLD, &numNodes);	// returns the total number of processes
	MPI_Comm_rank(MPI_COMM_WORLD, &nodeID);		// returns the rank (ID) of the calling MPI process

	/* writes formatted output to stdout */
	printf("Hello world from process %d of %d\n", nodeID, numNodes);

	/* terminates MPI environment, last MPI function call */
	MPI_Finalize();

	return 0;

}
