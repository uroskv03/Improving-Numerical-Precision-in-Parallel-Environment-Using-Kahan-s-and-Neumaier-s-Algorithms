# C compiler
CC = gcc
CC_FLAGS = -fopenmp -O3
LD_FLAGS = -lm -lquadmath

TARGET = matMul_test

all: $(TARGET)

$(TARGET): $(TARGET).c
	$(CC) $(CC_FLAGS) $(TARGET).c -o $(TARGET) $(LD_FLAGS)

clean:
	rm -f $(TARGET)