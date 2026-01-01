CXX := g++
CXXFLAGS := -std=c++20 -Wall -Wextra -O3 -march=native -mtune=native -fopenmp -DNDEBUG -pipe

TARGET := gen
SRC := src/gen.cpp

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)