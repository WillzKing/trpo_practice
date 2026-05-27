# Компилятор и флаги
CC = gcc
CFLAGS = -Wall -Wextra -O2
OPENBLAS_PATH = ../openblas_install
CFLAGS += -I"$(OPENBLAS_PATH)/include"
LDFLAGS = -L"$(OPENBLAS_PATH)/lib" -lopenblas -lm

# ============================================================
# Все цели
# ============================================================
all: test_level2.exe test_multithread.exe

# ============================================================
# Основной тест Level 2
# ============================================================
test_level2.exe: test_level2.c test_utils.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# ============================================================
# Тест многопоточности
# ============================================================
test_multithread.exe: test_multithread.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# ============================================================
# Запуск тестов
# ============================================================
run: test_level2.exe
	@echo "========================================"
	@echo "Running Level 2 tests with 2 threads..."
	@echo "========================================"
	@export PATH="../openblas_install/bin:$$PATH" && export OPENBLAS_NUM_THREADS=2 && ./test_level2.exe
	@echo ""
	@echo "========================================"
	@echo "Running Level 2 tests with 4 threads..."
	@echo "========================================"
	@export PATH="../openblas_install/bin:$$PATH" && export OPENBLAS_NUM_THREADS=4 && ./test_level2.exe

run_mt: test_multithread.exe
	@echo "========================================"
	@echo "Running multithreading performance test"
	@echo "========================================"
	@export PATH="../openblas_install/bin:$$PATH" && ./test_multithread.exe

run_all: run run_mt

# ============================================================
# Покрытие кода (gcov)
# ============================================================
cov: CFLAGS += -fprofile-arcs -ftest-coverage -O0
cov: LDFLAGS += -lgcov --coverage
cov: clean test_level2.exe
	@echo "========================================"
	@echo "Running tests for coverage analysis..."
	@echo "========================================"
	@export PATH="../openblas_install/bin:$$PATH" && export OPENBLAS_NUM_THREADS=2 && ./test_level2.exe
	@gcov test_level2.c
	@gcov test_utils.c
	@echo ""
	@echo "Coverage reports generated:"
	@echo "  - test_level2.c.gcov"
	@echo "  - test_utils.c.gcov"

# ============================================================
# Очистка
# ============================================================
clean:
	rm -f *.exe
	rm -f *.gcda *.gcno *.gcov
	rm -f *.o

# ============================================================
# Полная пересборка
# ============================================================
rebuild: clean all

# ============================================================
# Справка
# ============================================================
help:
	@echo "Available targets:"
	@echo "  all        - Build test_level2.exe and test_multithread.exe"
	@echo "  run        - Run Level 2 tests with 2 and 4 threads"
	@echo "  run_mt     - Run multithreading performance test"
	@echo "  run_all    - Run all tests"
	@echo "  cov        - Generate code coverage report"
	@echo "  clean      - Remove all generated files"
	@echo "  rebuild    - Clean and rebuild all"
	@echo "  help       - Show this help"

.PHONY: all run run_mt run_all cov clean rebuild help