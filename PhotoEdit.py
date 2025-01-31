import numpy as np
from PIL import Image


def inversion(image):
    image_array = np.array(image)
    if len(image_array.shape) == 3:
        print("Зображення має формат RGB.")
    else:
        raise ValueError("Зображення не є RGB.")

    image_array = image_array[:, :, :3]
    print(f"Розмір зображення: {image_array.shape[:2]}")

    processed_image_array = 255 - image_array

    return Image.fromarray(processed_image_array)


def gray(image):
    image_array = np.array(image)
    if len(image_array.shape) == 3:
        print("Зображення має формат RGB.")
    else:
        raise ValueError("Зображення не є RGB.")

    image_array = image_array[:, :, :3]
    print(f"Розмір зображення: {image_array.shape[:2]}")

    new_array = np.mean(image_array, axis=2, dtype='int16', out=None, keepdims=True)
    processed_image_array = np.concatenate((new_array, new_array, new_array), axis=2)

    return Image.fromarray(processed_image_array.astype(np.uint8))


def edges(image):
    def reform(arr, int_value):
        out_arr = int_value / arr.max() * arr
        return out_arr.astype('uint8')

    image_array = np.array(image)
    if len(image_array.shape) == 3:
        print("Зображення має формат RGB.")
    else:
        raise ValueError("Зображення не є RGB.")

    image_array = image_array[:, :, :3]
    print(f"Розмір зображення: {image_array.shape[:2]}")

    image_array = image_array.astype('int16')
    arr_shape = image_array.shape
    out_arr = np.zeros(arr_shape)

    first_line = image_array[:, 0]
    for i in range(arr_shape[1] - 1):
        second_line = image_array[:, i + 1]
        out_arr[:, i + 1] = abs(first_line - second_line)
        first_line = second_line.copy()

    first_line = image_array[0, :]
    for i in range(arr_shape[0] - 1):
        second_line = image_array[i + 1, :]
        out_arr[i + 1, :] = out_arr[i + 1, :] + abs(first_line - second_line)
        first_line = second_line.copy()

    return Image.fromarray(reform(out_arr, 255))


def noise(img_arr: np.ndarray):
    noise_arr = np.random.rand(1024, 1024, 3)
    noise_arr[noise_arr > 0.9] = 100
    noise_arr[noise_arr <= 0.1] = -100
    noise_arr = noise_arr.astype('int32')

    return np.clip(noise_arr + img_arr, a_min=0, a_max=255).astype('uint8')