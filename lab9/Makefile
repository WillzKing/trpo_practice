.PHONY: all test clean

all:
	$(MAKE) -C syrk
	$(MAKE) -C blas_level2

test:
	@echo "=== Running SYRK correctness tests ==="
	$(MAKE) -C syrk run_test
	@echo "=== Running BLAS Level 2 tests ==="
	$(MAKE) -C blas_level2 run
	@echo "=== All tests passed ==="

clean:
	$(MAKE) -C syrk clean
	$(MAKE) -C blas_level2 clean