# Define variables
CC = gcc
CFLAGS = -W -lm -fsanitize=address -static-libasan -g
TEST_SRC = tests/tests.c
OUTPUT = output

# Default target
all: $(OUTPUT)

# Compile and run the tests
$(OUTPUT): $(TEST_SRC)
	$(CC) -o $(OUTPUT) $(TEST_SRC) $(CFLAGS)
	./$(OUTPUT)

# Clean up the output file
clean:
	rm -f $(OUTPUT)
