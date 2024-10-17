CXX = g++
CXXFLAGS = -I /usr/include/eigen3 -Wall -O3

# Targets for each executable
TARGETS = main test_main new_test

all: $(TARGETS)

# Rule for compiling main.cpp into main executable
main: main.o
	$(CXX) $(CXXFLAGS) -o main main.o

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c main.cpp

test_main: test_main.o
	$(CXX) $(CXXFLAGS) -o test_main test_main.o

test_main.o: test_main.cpp
	$(CXX) $(CXXFLAGS) -c test_main.cpp

new_test: new_test.o
	$(CXX) $(CXXFLAGS) -o new_test new_test.o

new_test.o: new_test.cpp
	$(CXX) $(CXXFLAGS) -c new_test.cpp

clean:
	rm -f $(TARGETS) *.o

