.ONESHELL:
GCC_FLAGS = -fopenmp -Wall

ASSIGNMENT_GROUP=B
ASSIGNMENT_NUMBER=03
ASSIGNMENT_TITLE=pvs$(ASSIGNMENT_NUMBER)-grp$(ASSIGNMENT_GROUP)

.PHONY: build
build: helloMPI matmult

.PHONY: debug
debug: GCC_FLAGS += -g
debug: build

.PHONY: helloMPI
helloMPI:
	mpic++ helloMPI.cpp -o helloMPI

.PHONY: matmult
matmult:
	mpic++ matmult_A.cpp -o matmult_A

.PHONY: test
test: build
	mpirun -np 4 helloMPI
	mpirun -np 4 matmult_A

.PHONY: clean
clean:
	rm helloMPI matmult_A

.PHONY: codeformat
codeformat:
	clang-format -i *.[ch]pp

PDF_FILENAME=$(ASSIGNMENT_TITLE).pdf
.PHONY: pdf
pdf:
	pandoc pvs.md -o $(PDF_FILENAME) --from markdown --template ~/.pandoc/eisvogel.latex --listings

FILES=Makefile pvs.md *.[ch]pp $(PDF_FILENAME)

ASSIGNMENT_DIR=$(ASSIGNMENT_TITLE)
TARBALL_NAME=$(ASSIGNMENT_TITLE)-piekarski-wichmann-ruckel.tar.gz
.PHONY: tarball
tarball: pdf
	[ -z "$(TARBALL_NAME)" ] || rm $(TARBALL_NAME)
	mkdir $(ASSIGNMENT_DIR)
	for f in $(FILES); do cp $$f $(ASSIGNMENT_DIR); done
	tar zcvf $(TARBALL_NAME) $(ASSIGNMENT_DIR)
	rm -fr $(ASSIGNMENT_DIR)
