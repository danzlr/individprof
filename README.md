# Print+Brain

## Состав команды

* Мартемьянов Илья Тимофеевич — капитан, программист, алгоритмист
* Смирнов Михаил — программист, аналитик, алгоритмист
* Романов Артем Андреевич — программист, аналитик, тестировщик

---

## Итоги 1 дня

Команда успешно выполнила все задания первого дня, получив максимальные баллы на приватных тестах.

---

## Задание 1: Детекция событий на поле

### Описание

Разработка алгоритма для анализа видеопотока с дрона/камеры:

* Детекция изменений на поле
* Классификация событий (пожар, наводнение, авария и др.)
* Определение адреса дома
* Отображение информации на кадре

### Результаты

* Получен видеопоток: 10/10
* Детекция событий: 20/20
* Классификация: 30/30
* Геолокация: 50/50

Итог: 100/100

### Трудности и решения

Проблема: низкое качество исходного видео

Решение:

* Создание собственного датасета из кадров
* Использование многопоточности для обработки

### Основная логика

```python
if threading.active_count() <= 2:
    thread = threading.Thread(target=recognition_worker, args=(rois_to_process,))
    thread.daemon = True
    thread.start()

cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
display_text = f"Дом {address}: {event_label}"
cv2.putText(frame, display_text, (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
```

---

## Задание 2: Движение робота по линии

### Описание

Анализ видеопотока с камеры робота:

* Детекция линии
* Расчет скоростей моторов
* Сохранение логов
* Проверка точности

### Результаты

* Получен видеопоток: 10/10
* Детекция линии: 20/20
* Логи: 30/30
* Проверка: 50/50

Итог: 100/100

### Основная логика

```python
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

M = cv2.moments(thresh)
if M["m00"] > 0:
    cx = int(M["m10"] / M["m00"])
    error = cx - (frame.shape[1] // 2)

    wl_speed = base_speed + (kp * error)
    wr_speed = base_speed - (kp * error)

    log_file.write(f"{wl_speed:.2f} {wr_speed:.2f}\n")
```

---

## Задание 3: Распознавание сигналов светофора

### Описание

Анализ видеопотока при подъезде к перекрестку:

* Детекция светофора
* Распознавание сигнала
* Поиск стоп-линии
* Вывод команд STOP / GO

### Результаты

* Получен видеопоток: 10/10
* Детекция: 20/20
* Распознавание: 30/30
* Своевременность: 50/50

Итог: 100/100

### Трудности и решения

Проблема: блики и плохое освещение

Решение:

* HSV-фильтрация
* Морфологические операции

### Основная логика

```python
hsv_frame = cv2.cvtColor(roi_traffic_light, cv2.COLOR_BGR2HSV)
mask_red = cv2.inRange(hsv_frame, lower_red, upper_red)
mask_green = cv2.inRange(hsv_frame, lower_green, upper_green)

mask_yellow = cv2.inRange(hsv_road, lower_yellow, upper_yellow)

if is_near_stop_line(mask_yellow):
    if cv2.countNonZero(mask_red) > threshold:
        cv2.putText(frame, "STOP", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    elif cv2.countNonZero(mask_green) > threshold:
        cv2.putText(frame, "GO!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
```

---

## Использованные технологии

* Python 3
* OpenCV (cv2)
* NumPy
* Threading
* SIFT / MatchTemplate
