#nvcc matrix_square_test.cu -arch=sm_35 -O3 -o matrix.out -run
F77=nvcc
SRC = LJ_data.cu
OFLAG = -O3
OUTPUT = out
#LIBS = -lm -lcufft   
#LIBS = -L /usr/local/cuda/lib64 -lgsl -lgslcblas -lm -lcufft -lcufftw
LIBS = -L  
INCLUDES=-I  
Xcompiler = -Xcompiler " "  
${OUTPUT} : ${SRC}
	${F77} ${SRC} -arch=sm_30 -o  ${OUTPUT} ${OFLAG} ${LIBS} ${INCLUDES} ${Xcompiler}

