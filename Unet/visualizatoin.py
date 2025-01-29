import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_map(T, ind, preds, save_path=None):
    """
    Визуализация {i}-ой карты с начальной и конечной точками, а также результатом Дейкстры с прозрачными областями вне карты.

    Параметры:
        T (np.ndarray): Тензор данных размером (N, 64, 64, 4).
        i (int): Индекс карты для отображения.
        save_path (str): Путь для сохранения изображения (если указан).
    """
    raw_map = T[0]
    end_point = T[1]
    start_point = T[2]

    # Копируем массив расстояний и заменяем нули на NaN для прозрачных областей
    distances = np.copy(raw_map)
    distances[np.isclose(distances, 1)] = np.nan

    # Создаем цветовую карту с прозрачным фоном
    cmap = plt.cm.viridis
    cmap.set_bad(color=(0, 0, 0, 0))  # Прозрачный цвет для NaN

    plt.figure(figsize=(6, 6), dpi=100)
    plt.title(f"Map {ind + 1}: Landscape with Points")

    # Визуализация карты с прозрачным фоном для областей вне карты
    img = plt.imshow(((1 - distances) * preds)[0], cmap=cmap, vmin=0, vmax=np.nanmax((1 - raw_map) * preds[0]))

    # Наносим начальную и конечную точки
    end_point = np.argwhere(np.isclose(end_point, 1))[:, 1:]
    start_point  = np.argwhere(np.isclose(start_point, 1))[:, 1:]
    print(start_point, end_point)
    start_point = start_point[np.random.randint(0, len(start_point))].reshape(1, 2)

    if len(start_point) > 0:
        plt.scatter(start_point[0, 1], start_point[0, 0], color='blue', label='Start', s=100)
    if len(end_point) > 0:
        plt.scatter(end_point[0, 1], end_point[0, 0], color='red', label='End', s=100)

    plt.legend()
    plt.colorbar(img, label='Distance to End Point')

    # Сохраняем изображение, если указан путь
    if save_path:
        plt.savefig(save_path, transparent=True, bbox_inches='tight')

    plt.show()
    

def plot_multiple_maps(T, num_maps=5, save_dir="maps"):
    """
    Отображение нескольких карт с сохранением изображений.

    Параметры:
        T (np.ndarray): Тензор данных размером (N, 64, 64, 4).
        num_maps (int): Количество карт для отображения.
        save_dir (str): Директория для сохранения изображений.
    """
    import os

    # Создаем директорию для сохранения изображений, если её нет
    os.makedirs(save_dir, exist_ok=True)

    for i in range(min(num_maps, T.shape[0])):
        save_path = os.path.join(save_dir, f"map_{i + 1}.pdf")
        plot_map(T, i, save_path=save_path)