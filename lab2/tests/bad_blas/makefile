CC = gcc
CFLAGS = -Wall -Wextra -O2 -I.
LDFLAGS = -L. -lbadblas -lm

all: test_bad.exe

test_bad.exe: test_level2.c test_utils.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

run: test_bad.exe
	./test_bad.exe

clean:
	rm -f *.exe *.dll