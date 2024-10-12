CXX = g++
CXXFLAGS = -I /usr/include/eigen3 -Wall -O2
TARGET = main
SRC = main.cpp

all: $(TARGET)

$(TARGET):$(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)
