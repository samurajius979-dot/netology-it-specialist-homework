# Задание «Обучение нейросети» (MLP на PyTorch)

Модуль «Машинное обучение и нейросети», программа «Специалист по искусственному интеллекту».

**Файл:** `01_PyTorch_MLP_task_solution.ipynb` — заполненный и исполненный шаблон `PyTorch_mlp_task.ipynb`.

## Задача

Построить с нуля полносвязную нейросеть (Multi-Layer Perceptron) на PyTorch и обучить её на датасете
Breast Cancer из scikit-learn: реализовать класс Dataset, архитектуру сети, цикл обучения (train loop) и
цикл оценки (validation loop), построить графики Loss и Accuracy, провести минимум 2 эксперимента по улучшению
модели и написать аналитический вывод.

## Что сделано

| Часть шаблона | Реализация |
|---------------|------------|
| PyTorch Dataset | `CancerDataset` (X → `torch.float32`, y → `torch.float32` с `view(-1, 1)`) |
| DataLoader | `train_loader` — `batch_size=32, shuffle=True`; `test_loader` — `batch_size=32, shuffle=False` |
| Архитектура | `BinaryClassificationMLP(nn.Module)` — `input → 64 → 32 → 1`, ReLU между слоями, выход — логиты |
| Loss / optimizer | `nn.BCEWithLogitsLoss()` + `optim.Adam(lr=0.001)` |
| Train loop | `zero_grad → forward → loss → backward → step`, усреднение loss по батчам |
| Val loop | `model.eval()` + `torch.no_grad()`, расчёт val loss и Accuracy через `accuracy_score` |
| Графики | Loss (train/val) и Accuracy (train/val) |
| Эксперимент 1 | 3 скрытых слоя (128→64→32) + `BatchNorm1d` + ReLU, Adam lr=0.002 |
| Эксперимент 2 | `Dropout 0.3`, SGD + momentum (lr=0.05) |
| Выводы | Итоговая таблица, разбор каждого эксперимента, рекомендация по финальной конфигурации |

## Результаты (тестовая выборка, 114 наблюдений)

| Модель | Train Loss | Val Loss | Accuracy | Precision | Recall | F1 | ROC-AUC |
|--------|-----------|----------|----------|-----------|--------|-----|---------|
| Baseline (64-32, Adam) | 0.1155 | 2.7751 | 0.9737 | 0.9722 | 0.9859 | 0.9790 | 0.9944 |
| **Эксперимент 1 (BatchNorm, 3 слоя)** | 0.5200 | 2.8403 | 0.9737 | 0.9595 | **1.0000** | **0.9793** | 0.9908 |
| Эксперимент 2 (Dropout + SGD) | 1.3029 | 5.9783 | 0.9649 | 0.9718 | 0.9718 | 0.9718 | 0.9895 |

**Вывод:** финальной выбрана конфигурация эксперимента 1 — при равной accuracy она даёт **Recall = 1.0**
(ни один злокачественный случай не пропущен, что критично для медицинской задачи), лучший F1 и меньший разрыв
между train и val loss по сравнению с baseline. Эксперимент 2 показал, что Dropout действительно уменьшает
переобучение, но неоптимальный SGD с momentum съел этот выигрыш — вывод: оставить Dropout, заменить оптимизатор
на Adam.

## Как запустить

```bash
pip install torch numpy pandas matplotlib scikit-learn jupyter
jupyter lab 01_PyTorch_MLP_task_solution.ipynb
```

Ноутбук воспроизводится целиком (`Kernel → Restart and Run All`): данные берутся из
`sklearn.datasets.load_breast_cancer()`, скачивать ничего не нужно.
