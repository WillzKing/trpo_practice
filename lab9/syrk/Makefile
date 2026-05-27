CC = gcc
CFLAGS = -Wall -Wextra -O2
OPENBLAS_PATH ?= ../../openblas_install
CFLAGS += -I"$(OPENBLAS_PATH)/include"
LDFLAGS = -L"$(OPENBLAS_PATH)/lib" -lopenblas -lm

all: test_syrk.exe benchmark_syrk.exe

test_syrk.exe: test_syrk.c syrk.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

benchmark_syrk.exe: benchmark_syrk.c syrk.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

run_test: test_syrk.exe
	.\test_syrk.exe

run_bench: benchmark_syrk.exe
	.\benchmark_syrk.exe

clean:
	rm -f *.exe

.PHONY: all run_test run_bench clean