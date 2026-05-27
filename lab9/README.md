# BLAS Implementation and Tests

## Структура проекта

- `syrk/` - реализация SYRK (BLAS Level 3) и тесты производительности
- `blas_level2/` - интерфейсные тесты для CBLAS Level 2
- `bad_blas/` - "кривая" библиотека для демонстрации падения тестов

## Требования

- GCC (MinGW)
- OpenBLAS (установлена в `../openblas_install`)

## Сборка и запуск

\`\`\`bash
make all
make test
\`\`\`