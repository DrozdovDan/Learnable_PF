# Learnable_PF
## Dataset generator

***TODO***

## Pretrained Models

Директория `./weights` содержит параметры для некоторых предобученных моделей.

Используйте `Workflow.ipynb` для обучения (или дообучения) своей собственной модели. Для этого установите в ячейке ноутбука под номером 2 параметр `dataset_dir` — путь до своего датасета, параметр `mode` — тип изучаемой эвристики (доступны cf и h эвристики), `batch_size` — размер одного батча, `max_epochs` — количество эпох, `learning_rate`, weight_decay, `limit_train_batches` и `limit_val_batches`, если вы хотите ограничить количество батчей, `proj_name` — директория, куда сохраняются логи (используется wandb логгер), `accelerator` — на чём обучать (cuda, cpu, что-то ещё), `devices` — номер устройства на котором обучать (если появится желание обучать на более чем одном устройстве, могут потребоваться дополнительные изменения в коде). По-умолчанию стоит обучение модели `TransPath` с нуля, если вы желаете обчить другую модель или дообучить уже имеющуюся модель, стоит в ячейке ноутбука под номером 5 изменить слеудующий код:
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


