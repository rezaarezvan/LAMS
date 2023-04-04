# Define variables
CC = gcc
CFLAGS = -W -lm -fsanitize=address -static-libasan -g
TEST_SRC = tests/tests.c
OUTPUT = output

# Default target
all: $(OUTPUT)

# Compile the tests
$(OUTPUT): $(TEST_SRC) src/linear_algebra.c src/stats.c
	$(CC) $(CFLAGS) -o $(OUTPUT) $(TEST_SRC) src/linear_algebra.c src/stats.c

# Run the tests
test: $(OUTPUT)
	./$(OUTPUT)

# Clean up the output file
clean:
	rm -f $(OUTPUT)
