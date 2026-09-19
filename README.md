# Мониторинг природных пожаров

Воспроизводимый baseline для кейса «КосмоХакатон 2026»: детектор активного горения (AF), картирование гарей и степени поражения (BS), формирование RLE-submission и REST API для подготовленного геокаталога.

## Что реализовано

- Быстрый baseline: контекстный признак I4-I5 для AF, dNBR с порогами по land cover и SCL/SAR-проверкой для BS.
- Обучаемый вариант: `HistGradientBoostingClassifier` по спектральным, индексным, SAR и вспомогательным признакам.
- RLE-кодировщик/декодировщик, проверка полного submission и EDA, которые запускаются из командной строки.
- FastAPI: фильтрация подготовленных GeoJSON-объектов по датам, bbox или полигону, контуры и аналитическая справка.

Не используются геопривязка, даты или внешние продукты для тестовых чипов.

## Установка

```powershell
python -m pip install -r requirements.txt
```

Требуется Python 3.12+. Данные должны оставаться в исходной структуре. Для примеров ниже пути на Windows заключены в кавычки.

## EDA и обучение

```powershell
python scripts/eda.py --data-dir "Мониторинг DATA/train" --output report/EDA.md
python train_models.py --data-dir "Мониторинг DATA/train" --models-dir artifacts/models
```

`train_models.py` разделяет AF по годам съёмки, BS — по `fire_event_id`, исключает no-data и не передаёт модели target-derived поля из `meta.csv`. Для строгой оценки на всех пикселях отложенных чипов используйте `--full-chip-validation`. Последний строгий прогон: F1 AF 0,8735, IoU гари 0,5155, mIoU степени 0,5252, Score 0,6437.

## Инференс и проверка

Быстрый пороговый режим:

```powershell
python inference.py --data-dir "Мониторинг DATA/test" --output output/submission_baseline.csv
```

Обучаемый режим:

```powershell
python inference.py --data-dir "Мониторинг DATA/test" --models-dir artifacts/models --output output/submission_model.csv
python scripts/validate_submission.py --submission output/submission_model.csv --template "Мониторинг DATA/test/sample_submission.csv" --meta "Мониторинг DATA/test/meta.csv"
```

Команда создаёт все 447 строк шаблона; пустые маски записываются как пустой RLE. Валидатор проверяет пары `(chip_id, class_id)`, границы RLE и отсутствие пересечений классов BS.

### Docker

```powershell
docker build -t fire-monitoring .
docker run --rm -v "${PWD}\Мониторинг DATA:/data" -v "${PWD}\output:/output" fire-monitoring --data-dir /data/test --output /output/submission.csv
```

Обучаемые веса в Docker передаются отдельным volume в `artifacts/models` и указываются через `--models-dir`.

## Сервис

```powershell
python scripts/build_demo_catalogue.py --data-dir "Мониторинг DATA/train" --output service/catalogue.geojson
uvicorn service.app:app --host 0.0.0.0 --port 8000
```

Генератор создаёт демонстрационный GeoJSON только из разрешённой train-части. Карта доступна на `/`, интерактивная схема — на `/docs`. API: `POST /api/v1/query`, `/api/v1/fire-points`, `/api/v1/burn-polygons`, `/api/v1/report`.

## Ограничения текущей версии

Пороговый baseline работает менее чем за 30 секунд на этой машине. Модельный CPU-инференс занимает около 93 секунд; перед финальной сдачей его следует ускорить и выбрать вариант по публичной метрике.
