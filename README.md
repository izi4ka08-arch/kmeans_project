# k-means (K-средних) на C — с нуля

Реализация **k-means** (алгоритм Ллойда) **на чистом C**: библиотечная функция для кластеризации, CLI-демо для работы с CSV, генерация синтетических данных и визуализация результатов.

![Визуализация кластеров](assets/clusters_example.png)

## Что это делает

Алгоритм k-means разбивает набор `d`-мерных точек на `k` кластеров так, чтобы каждая точка относилась к кластеру с ближайшим центроидом (средним).

## Когда это использовать

- Когда у вас **числовые данные** (векторы признаков).
- Когда **нет меток** классов и нужно разбить данные на группы.
- Когда вы **знаете `k`** (количество кластеров) или можете подобрать его экспериментально.

## Быстрый старт

1) Сгенерировать данные:

```bash
python tools/generate_blobs.py --out data.csv --samples 600 --centers 3
```

Чтобы получить повторяемый датасет (одинаковый на разных запусках), укажите `--seed`, например:

```bash
python tools/generate_blobs.py --out data.csv --samples 600 --centers 3 --seed 42
```

2) Собрать (CMake):

```bash
cmake -S . -B build
cmake --build build --config Release
```

3) Запустить демо:

Windows (PowerShell), если собирали с `--config Release`:

```powershell
.\build\Release\kmeans_demo.exe --in data.csv --k 3 --out pred.csv --seed 42
```

Linux/macOS (обычно без `--config Release`):

```bash
./build/kmeans_demo --in data.csv --k 3 --out pred.csv --seed 42
```

4) Визуализировать:

```bash
python tools/plot_results.py --data data.csv --pred pred.csv
```

Важно: папка `build/` привязана к пути на диске и конкретной машине. Если переносите проект на другой ПК (или переместили папку проекта), просто удалите `build/` и пересоздайте:

```powershell
Remove-Item -Recurse -Force .\build
cmake -S . -B build
cmake --build build --config Release
```

## Пример использования как библиотеки (C API)

Ниже — минимальная схема вызова: задаём параметры, передаём матрицу `X` размера `n_samples × n_features`, получаем `labels` и `centroids`.

```c
#include "kmeans.h"

// X: плоский массив длиной n_samples*n_features (строчно-ориентированно)
// labels_out: длина n_samples
// centroids_out: длина k*n_features
int rc = kmeans_fit(X, n_samples, n_features, &params, labels_out, centroids_out, &inertia);
```

См. заголовок `src/kmeans.h` и демо `demo/kmeans_demo.c` для полного примера.

## Сложность

Если `k` и `d` фиксированы, время работы одной итерации — `O(n*k*d)`, а всего — `O(n*k*d*i)`, где:

- `n` — число объектов (samples)
- `k` — число кластеров
- `d` — размерность (features)
- `i` — число итераций до сходимости

На данных с выраженной кластерной структурой обычно сходится за небольшое число итераций.

## Параметры и опции

Параметры алгоритма задаются через `kmeans_params_t` (см. `src/kmeans.h`):

- `k` — количество кластеров
- `max_iters` — максимум итераций
- `tol` — порог сходимости (0 — отключить проверку по сдвигу центроидов)
- `seed` — seed для генератора случайных чисел
- `init_method` — инициализация центроидов (случайная или k-means++)

## Структура проекта

- `src/kmeans.h`, `src/kmeans.c` — реализация k-means
- `src/csv.h`, `src/csv.c` — чтение `data.csv` и запись `pred.csv`
- `demo/kmeans_demo.c` — CLI-демо (читает CSV → кластеризует → пишет CSV)
- `tools/generate_blobs.py` — генерация `data.csv`
- `tools/plot_results.py` — построение `clusters.png` и расчёт метрик

## Сборка без CMake (MSVC)

Откройте **x64 Native Tools Command Prompt for VS** и выполните:

```bat
cl /O2 /I src demo\kmeans_demo.c src\kmeans.c src\csv.c /Fe:kmeans_demo.exe
kmeans_demo.exe --in data.csv --k 3 --out pred.csv
```

