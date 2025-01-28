# Learnable_PF

## Общая информация

Существуют несколько эвристических алгоритмов для нахождения оптимального (или близкого к опптимельному) пути в гриде. Нас интересуют алгоритмы `A*` и `WA*`. Данный код изучает возможность предсказания эвристик для данных алгоритмов с помощью глубоких нейронных сетей, в частности TransPath и U-Net.

## Генератор датасета

***TODO***

## Эвристики

Данный код предназначен для обучения и использования двух эвристик: абсолютного оптимального пути от каждой проходимой клетки грида, достижимой из клетки `goal`, до клетки `goal` и корректирующий фактор для каждой клетки — число от 0 до 1, на которое нужно поделить быстро вычислимую эвристику `octtile distance`, чтобы получился абсолютный оптимальный путь от данной проходимой клетки, достижимой и клетки `goal`, до клетки `goal`. Для точного подсчёта двух данных эвристик используется алгоритм Дейкстры, запущенный из клетки `goal`.

## Обучение моделей

Директория `./weights` содержит параметры для некоторых предобученных моделей.

Используйте `Workflow.ipynb` для обучения (или дообучения) своей собственной модели. Для этого установите в ячейке ноутбука под номером 2 параметр `dataset_dir` — путь до своего датасета, датасет должен иметь абсолютно тот же формат, что и TMP датасет (можно скачать TMP датасет, запустив файл `download.py`, датасет весит порядка 20ГБ!), параметр `mode` — тип изучаемой эвристики (доступны `cf` и `h` эвристики), `batch_size` — размер одного батча, `max_epochs` — количество эпох, `learning_rate`, weight_decay, `limit_train_batches` и `limit_val_batches`, если вы хотите ограничить количество батчей, `proj_name` — директория, куда сохраняются логи (используется wandb логгер), `accelerator` — на чём обучать (cuda, cpu, что-то ещё), `devices` — номер устройства на котором обучать (если появится желание обучать на более чем одном устройстве, могут потребоваться дополнительные изменения в коде). По-умолчанию стоит обучение модели `TransPath` с нуля, если вы желаете обчить другую модель или дообучить уже имеющуюся модель, стоит в ячейке ноутбука под номером 5 изменить слеудующий код:
```
#model_path = './weights/alex_100_h_model'
model = TransPathModel()
#model.load_state_dict(torch.load(model_path, weights_only=True))
lit_module = TransPathLit(
        model=model,
        mode=mode,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )
```

## Предсказание эвристик и сбор статистики по датасету

Запустив `Statistics_collector.ipynb`, можно предсказать необходимые эвристики, а также использовать их для подсчёта таких статистик, как `number of expansions` и `path length`, также в качестве бэйзлайна подсчитываются статистики для алгоритма `A*` с эвристикой `octile distance` и `tie-break`, что всегда даёт нам оптимальную длину пути в данном сценарии (упадёт с `assert`, если в тестовых экземплярах будет присутствовать сценарий с несуществующим путём). Параметры здесь аналогичны параметрам при обучении.


