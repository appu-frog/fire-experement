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

`train_models.py` группирует обучающую и проверочную части по `fire_event_id`, исключает пиксели no-data и не передаёт модели target-derived поля из `meta.csv`. Метрики в `artifacts/models/validation_metrics.json` являются оценкой на сэмплированных пикселях отложенных пожаров, поэтому для окончательного выбора модели нужен отдельный тест на полных чипах/регионах.

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

## Сервис

```powershell
uvicorn service.app:app --host 0.0.0.0 --port 8000
```

Для демонстрации укажите `FIRE_CATALOGUE=service/catalogue.geojson`. Это GeoJSON FeatureCollection, где свойства объекта содержат `kind` (`fire_point` или `burn_polygon`), `observed_at`, `severity_class` и `area_ha`. API: `POST /api/v1/query`, `/api/v1/fire-points`, `/api/v1/burn-polygons`, `/api/v1/report`; интерактивная схема доступна в `/docs`.

## Ограничения текущей версии

Пороговый baseline работает менее чем за 30 секунд на этой машине. Модельный вариант точнее на event-holdout сэмплах, но CPU-инференс занимает около 93 секунд; перед финальной сдачей его следует ускорить (ONNX/уменьшение числа итераций) и выбрать вариант по публичной метрике.
