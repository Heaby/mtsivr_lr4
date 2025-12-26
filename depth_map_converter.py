import struct
import os
import json
import math
import sys

try:
    from OpenGL.GL import *
    from OpenGL.GLUT import *
    from OpenGL.GLU import *

    OPENGL_AVAILABLE = True
    glut_initialized = False  # Флаг инициализации GLUT
except ImportError as e:
    OPENGL_AVAILABLE = False
    glut_initialized = False
    print(f"OpenGL библиотеки не установлены: {e}")
    print("Для установки выполните:")
    print("  pip install PyOpenGL PyOpenGL-GLUT")
except Exception as e:
    OPENGL_AVAILABLE = False
    glut_initialized = False
    print(f"Ошибка при загрузке OpenGL: {e}")

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("NumPy не установлен, некоторые функции могут работать медленнее")


# ==================== ЧТЕНИЕ КОНФИГУРАЦИИ ====================

def load_config(config_file="config.json"):
    """Загрузка конфигурации из JSON файла"""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Файл конфигурации {config_file} не найден")

    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)

    return config


# ==================== ЧТЕНИЕ КАРТЫ ГЛУБИНЫ ====================

def read_depth_map(filename):
    """Чтение карты глубины"""
    file_size = os.path.getsize(filename)
    print(f"Размер файла: {file_size} байт")

    with open(filename, 'rb') as f:
        # Чтение размеров как double
        height_raw = struct.unpack('d', f.read(8))[0]
        width_raw = struct.unpack('d', f.read(8))[0]

        height = int(round(height_raw))
        width = int(round(width_raw))

        print(f"Размеры из заголовка: {width} x {height}")

        if height <= 0 or width <= 0:
            raise ValueError(f"Некорректные размеры: {width} x {height}")

        # Чтение данных глубины
        data_size = width * height
        data = struct.unpack(f'{data_size}d', f.read(data_size * 8))

        print(f"Прочитано значений: {len(data)}")
        print(f"Диапазон глубин: {min(data):.3f} - {max(data):.3f}")

        return width, height, data


# ==================== СОЗДАНИЕ 3D МОДЕЛИ ====================

def create_3d_model_from_depth_map(depth_data, width, height):
    """Создание 3D модели"""
    vertices = []
    vertex_index_map = [-1] * (width * height)

    # Создание вершин
    for y in range(height):
        for x in range(width):
            grid_index = y * width + x
            depth = depth_data[grid_index]

            # Пропускаем точки с нулевой глубиной (фон)
            if depth > 0.0:
                point_x = float(x)
                point_y = float(height - y - 1)  # Инвертируем Y
                point_z = depth

                vertex_index_map[grid_index] = len(vertices)
                vertices.append((point_x, point_y, point_z))

    # Создание треугольных граней
    faces = []
    for y in range(height - 1):
        for x in range(width - 1):
            idx1 = y * width + x
            idx2 = y * width + (x + 1)
            idx3 = (y + 1) * width + x
            idx4 = (y + 1) * width + (x + 1)

            v1 = vertex_index_map[idx1]
            v2 = vertex_index_map[idx2]
            v3 = vertex_index_map[idx3]
            v4 = vertex_index_map[idx4]

            # Проверяем, что все вершины существуют
            if v1 != -1 and v2 != -1 and v3 != -1 and v4 != -1:
                # Первый треугольник (v1, v2, v3)
                faces.append((v1, v2, v3))
                # Второй треугольник (v2, v4, v3)
                faces.append((v2, v4, v3))

    print(f"Создано вершин: {len(vertices)}")
    print(f"Создано треугольников: {len(faces)}")

    return vertices, faces


def compute_vertex_normals(vertices, faces):
    """Вычисление нормалей для вершин"""
    normals = [[0.0, 0.0, 0.0] for _ in range(len(vertices))]

    for face in faces:
        if len(face) == 3:
            v0, v1, v2 = face
            if all(0 <= idx < len(vertices) for idx in [v0, v1, v2]):
                p0 = vertices[v0]
                p1 = vertices[v1]
                p2 = vertices[v2]

                # Векторы сторон треугольника
                u = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]]
                v = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]]

                # Векторное произведение для нормали
                normal = [
                    u[1] * v[2] - u[2] * v[1],
                    u[2] * v[0] - u[0] * v[2],
                    u[0] * v[1] - u[1] * v[0]
                ]

                # Нормализация
                length = (normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2) ** 0.5
                if length > 0:
                    normal = [n / length for n in normal]

                    # Добавляем нормаль ко всем вершинам треугольника
                    for vertex_idx in face:
                        normals[vertex_idx][0] += normal[0]
                        normals[vertex_idx][1] += normal[1]
                        normals[vertex_idx][2] += normal[2]

    # Нормализуем итоговые нормали
    for i in range(len(normals)):
        length = (normals[i][0] ** 2 + normals[i][1] ** 2 + normals[i][2] ** 2) ** 0.5
        if length > 0:
            normals[i] = [n / length for n in normals[i]]
        else:
            normals[i] = [0.0, 0.0, 1.0]

    return normals


# ==================== МОДЕЛИ ОТРАЖЕНИЯ ====================

def dot_product(v1, v2):
    """Скалярное произведение векторов"""
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]


def normalize_vector(v):
    """Нормализация вектора"""
    length = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    if length > 0:
        return [v[0] / length, v[1] / length, v[2] / length]
    return [0.0, 0.0, 1.0]


def reflect_vector(incident, normal):
    """Отражение вектора от нормали"""
    dot = dot_product(incident, normal)
    return [
        incident[0] - 2 * dot * normal[0],
        incident[1] - 2 * dot * normal[1],
        incident[2] - 2 * dot * normal[2]
    ]


def lambert_reflection(normal, light_dir, material_diffuse):
    """Модель отражения Ламберта"""
    n_dot_l = max(0.0, dot_product(normal, light_dir))
    return [
        material_diffuse[0] * n_dot_l,
        material_diffuse[1] * n_dot_l,
        material_diffuse[2] * n_dot_l
    ]


def phong_blinn_reflection(normal, light_dir, view_dir, material_diffuse, material_specular, shininess):
    """Модель отражения Фонга-Блинна"""
    # Диффузная составляющая (Ламберт)
    n_dot_l = max(0.0, dot_product(normal, light_dir))
    diffuse = [
        material_diffuse[0] * n_dot_l,
        material_diffuse[1] * n_dot_l,
        material_diffuse[2] * n_dot_l
    ]

    # Спекулярная составляющая (Блинн)
    half_vector = normalize_vector([
        light_dir[0] + view_dir[0],
        light_dir[1] + view_dir[1],
        light_dir[2] + view_dir[2]
    ])
    n_dot_h = max(0.0, dot_product(normal, half_vector))

    # Вычисляем интенсивность блика
    # При больших значениях shininess используем более заметную формулу
    if n_dot_l > 0:
        # Нормализуем shininess для более предсказуемого результата
        normalized_shininess = max(1.0, min(128.0, shininess))

        # Используем более мягкую формулу для лучшей видимости бликов
        # Вместо прямого возведения в степень, используем более плавную кривую
        specular_intensity = pow(max(0.0, n_dot_h), normalized_shininess)

        # Агрессивное усиление бликов для лучшей видимости
        if normalized_shininess > 50:
            specular_intensity *= 3.0  # Сильное усиление для высокого блеска
        elif normalized_shininess > 20:
            specular_intensity *= 2.0  # Усиление для среднего блеска
        else:
            specular_intensity *= 1.5  # Небольшое усиление для низкого блеска

        # Дополнительно усиливаем яркие блики (когда n_dot_h близок к 1)
        if n_dot_h > 0.8:
            specular_intensity *= 1.5
    else:
        specular_intensity = 0.0

    specular = [
        material_specular[0] * specular_intensity,
        material_specular[1] * specular_intensity,
        material_specular[2] * specular_intensity
    ]

    return [
        diffuse[0] + specular[0],
        diffuse[1] + specular[1],
        diffuse[2] + specular[2]
    ]


def oren_nayar_reflection(normal, light_dir, view_dir, material_diffuse, roughness):
    """Модель отражения Орена-Найара"""
    n_dot_l = max(0.0, dot_product(normal, light_dir))
    n_dot_v = max(0.0, dot_product(normal, view_dir))

    if n_dot_l <= 0.0 or n_dot_v <= 0.0:
        return [0.0, 0.0, 0.0]

    # Углы
    theta_r = math.acos(n_dot_v)
    theta_i = math.acos(n_dot_l)

    # Проекции на плоскость
    cos_phi_diff = dot_product(
        normalize_vector(
            [view_dir[0] - normal[0] * n_dot_v, view_dir[1] - normal[1] * n_dot_v, view_dir[2] - normal[2] * n_dot_v]),
        normalize_vector([light_dir[0] - normal[0] * n_dot_l, light_dir[1] - normal[1] * n_dot_l,
                          light_dir[2] - normal[2] * n_dot_l])
    )

    # Параметры модели
    sigma2 = roughness * roughness
    A = 1.0 - 0.5 * (sigma2 / (sigma2 + 0.33))
    B = 0.45 * (sigma2 / (sigma2 + 0.09))

    alpha = max(theta_i, theta_r)
    beta = min(theta_i, theta_r)

    # Формула Орена-Найара
    C1 = A
    C2 = B * max(0.0, cos_phi_diff) * math.sin(alpha) * math.tan(beta)

    intensity = n_dot_l * (C1 + C2)

    return [
        material_diffuse[0] * intensity,
        material_diffuse[1] * intensity,
        material_diffuse[2] * intensity
    ]


def calculate_vertex_color(vertex, normal, light_pos, viewer_pos, material, reflection_model):
    """Вычисление цвета вершины с учетом модели отражения"""
    # Направление к источнику света
    light_dir = normalize_vector([
        light_pos[0] - vertex[0],
        light_pos[1] - vertex[1],
        light_pos[2] - vertex[2]
    ])

    # Направление к наблюдателю
    view_dir = normalize_vector([
        viewer_pos[0] - vertex[0],
        viewer_pos[1] - vertex[1],
        viewer_pos[2] - vertex[2]
    ])

    # Выбор модели отражения
    if reflection_model == "lambert":
        color = lambert_reflection(normal, light_dir, material["diffuse"])
    elif reflection_model == "phong_blinn":
        color = phong_blinn_reflection(
            normal, light_dir, view_dir,
            material["diffuse"], material["specular"], material["shininess"]
        )
    elif reflection_model == "oren_nayar":
        color = oren_nayar_reflection(
            normal, light_dir, view_dir,
            material["diffuse"], material["roughness"]
        )
    else:
        # По умолчанию Ламберт
        color = lambert_reflection(normal, light_dir, material["diffuse"])

    # Добавляем ambient (для Phong-Blinn уменьшаем влияние ambient, чтобы блики были заметнее)
    ambient = material.get("ambient", [0.1, 0.1, 0.1])
    if reflection_model == "phong_blinn":
        # Для Phong-Blinn используем меньше ambient, чтобы блики выделялись
        ambient_factor = 0.5  # Уменьшаем ambient наполовину
        color = [
            min(1.0, color[0] + ambient[0] * ambient_factor),
            min(1.0, color[1] + ambient[1] * ambient_factor),
            min(1.0, color[2] + ambient[2] * ambient_factor)
        ]
    else:
        # Для других моделей используем полный ambient
        color = [
            min(1.0, color[0] + ambient[0]),
            min(1.0, color[1] + ambient[1]),
            min(1.0, color[2] + ambient[2])
        ]

    return color


# ==================== ЭКСПОРТ В РАЗЛИЧНЫЕ ФОРМАТЫ ====================

def export_to_vrml(vertices, faces, output_filename, material=None):
    """Экспорт в VRML формат"""
    with open(output_filename, 'w', encoding='utf-8') as f:
        # Заголовок VRML
        f.write("#VRML V2.0 utf8\n")
        f.write("Shape {\n")
        f.write("  appearance Appearance {\n")
        f.write("    material Material {\n")

        if material:
            f.write(
                f"      diffuseColor {material['diffuse'][0]:.3f} {material['diffuse'][1]:.3f} {material['diffuse'][2]:.3f}\n")
            f.write(
                f"      specularColor {material['specular'][0]:.3f} {material['specular'][1]:.3f} {material['specular'][2]:.3f}\n")
            f.write(f"      shininess {material['shininess']:.3f}\n")
        else:
            f.write("      diffuseColor 0.8 0.8 0.8\n")
            f.write("      specularColor 0.5 0.5 0.5\n")
            f.write("      shininess 0.8\n")

        f.write("    }\n")
        f.write("  }\n")
        f.write("  geometry IndexedFaceSet {\n")
        f.write("    solid FALSE\n")
        f.write("    creaseAngle 0.5\n")

        # Запись координат вершин
        f.write("    coord Coordinate {\n")
        f.write("      point [\n")
        for x, y, z in vertices:
            f.write(f"        {x:.6f} {y:.6f} {z:.6f},\n")
        f.write("      ]\n")
        f.write("    }\n")

        # Запись индексов граней
        f.write("    coordIndex [\n")
        for face in faces:
            f.write(f"        {face[0]}, {face[1]}, {face[2]}, -1,\n")
        f.write("    ]\n")

        f.write("  }\n")
        f.write("}\n")


def export_to_obj(vertices, faces, output_filename):
    """Экспорт в OBJ формат"""
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write("# OBJ file generated from depth map\n")

        # Запись вершин
        for x, y, z in vertices:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")

        # Запись граней (индексы в OBJ начинаются с 1)
        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


def export_to_ply(vertices, faces, output_filename):
    """Экспорт в PLY формат"""
    with open(output_filename, 'w', encoding='utf-8') as f:
        # Заголовок PLY
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        # Запись вершин
        for x, y, z in vertices:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

        # Запись граней
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


# ==================== ЭКСПОРТ В BMP ====================

def export_to_bmp(vertices, faces, normals, output_filename, config):
    """Экспорт в BMP с учетом модели отражения"""
    if not OPENGL_AVAILABLE:
        print("OpenGL не доступен, экспорт в BMP невозможен")
        return

    vis_config = config["visualization"]
    material_config = config["material"]

    width = vis_config["bmp_resolution"]["width"]
    height = vis_config["bmp_resolution"]["height"]

    # Вычисление центра и масштаба модели
    if not vertices:
        print("Нет вершин для экспорта")
        return

    min_x = min(v[0] for v in vertices)
    max_x = max(v[0] for v in vertices)
    min_y = min(v[1] for v in vertices)
    max_y = max(v[1] for v in vertices)
    min_z = min(v[2] for v in vertices)
    max_z = max(v[2] for v in vertices)

    center = [
        (min_x + max_x) * 0.5,
        (min_y + max_y) * 0.5,
        (min_z + max_z) * 0.5
    ]

    span_x = max(1e-6, max_x - min_x)
    span_y = max(1e-6, max_y - min_y)
    span_z = max(1e-6, max_z - min_z)
    max_span = max(span_x, span_y, span_z)
    scale = 2.0 / max_span

    # Инициализация OpenGL для рендеринга
    global glut_initialized
    if not glut_initialized:
        glutInit(sys.argv)
        glut_initialized = True

    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(width, height)
    window_id = glutCreateWindow(b"BMP Export")

    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)
    glClearColor(0.1, 0.1, 0.1, 1.0)  # Светло-серый фон вместо черного

    # Устанавливаем viewport
    glViewport(0, 0, width, height)

    # Настройка камеры
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    # Увеличиваем дальнюю плоскость отсечения для больших моделей
    gluPerspective(45.0, float(width) / float(height), 0.1, 1000.0)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # ВАЖНО: Сначала трансформируем модель (центрируем и масштабируем)
    # чтобы она была в начале координат и нормализованного размера
    glScaled(scale, scale, scale)
    glTranslated(-center[0], -center[1], -center[2])

    # Поворот модели для правильной ориентации (из конфигурации)
    # Применяем повороты в правильном порядке: сначала Z, потом Y, потом X
    rotation = vis_config.get("model_rotation", {"x": 0.0, "y": 0.0, "z": 0.0})
    if rotation.get("z", 0.0) != 0.0:
        glRotatef(rotation["z"], 0.0, 0.0, 1.0)
    if rotation.get("y", 0.0) != 0.0:
        glRotatef(rotation["y"], 0.0, 1.0, 0.0)
    if rotation.get("x", 0.0) != 0.0:
        glRotatef(rotation["x"], 1.0, 0.0, 0.0)

    # Теперь модель находится в начале координат (0,0,0) с нормализованным размером
    # Вычисляем расстояние до камеры на основе нормализованного размера
    normalized_size = 2.0  # После масштабирования модель имеет размер ~2 единицы
    camera_distance = 3.0  # Расстояние от начала координат (достаточно для обзора)

    # Позиция наблюдателя из конфига (используем как направление)
    viewer_pos = vis_config["viewer_position"]
    # Нормализуем направление от центра к наблюдателю
    viewer_dir = [
        viewer_pos["x"] - center[0],
        viewer_pos["y"] - center[1],
        viewer_pos["z"] - center[2]
    ]
    viewer_len = math.sqrt(viewer_dir[0] ** 2 + viewer_dir[1] ** 2 + viewer_dir[2] ** 2)
    if viewer_len > 0:
        viewer_dir = [d / viewer_len for d in viewer_dir]
    else:
        viewer_dir = [0.0, 0.0, 1.0]  # По умолчанию смотрим по Z

    # Позиция камеры (теперь относительно начала координат, т.к. модель уже центрирована)
    camera_pos = [
        viewer_dir[0] * camera_distance,
        viewer_dir[1] * camera_distance,
        viewer_dir[2] * camera_distance
    ]

    # Поворот камеры вокруг модели (если указан в конфиге)
    camera_rotation = vis_config.get("camera_rotation", {"y": 0.0})  # Поворот вокруг Y (горизонтальный)
    cam_rot_y = camera_rotation.get("y", 0.0)

    if abs(cam_rot_y) > 0.1:
        # Поворачиваем позицию камеры вокруг модели (вокруг оси Y)
        angle_rad = math.radians(cam_rot_y)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        # Поворот вокруг оси Y (горизонтальный поворот)
        new_camera_x = camera_pos[0] * cos_a - camera_pos[2] * sin_a
        new_camera_z = camera_pos[0] * sin_a + camera_pos[2] * cos_a
        camera_pos = [new_camera_x, camera_pos[1], new_camera_z]

    # Устанавливаем камеру (смотрим на начало координат, где теперь находится модель)
    # Вектор "up" зависит от поворота модели
    rotation = vis_config.get("model_rotation", {"x": 0.0, "y": 0.0, "z": 0.0})
    rot_x = rotation.get("x", 0.0)

    # Определяем вектор "up" в зависимости от поворота
    if abs(rot_x - 90.0) < 1.0 or abs(rot_x + 90.0) < 1.0:
        # Если повернуто на 90 градусов вокруг X, меняем вектор "up"
        up_vector = [0.0, 0.0, 1.0]
    else:
        # Стандартный вектор "up"
        up_vector = [0.0, 1.0, 0.0]

    gluLookAt(
        camera_pos[0], camera_pos[1], camera_pos[2],
        0.0, 0.0, 0.0,  # Смотрим на начало координат (центр модели)
        up_vector[0], up_vector[1], up_vector[2]
    )

    # Настройка освещения
    light_pos = vis_config["light_position"]
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION, [light_pos["x"], light_pos["y"], light_pos["z"], 1.0])
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
    glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])

    # Отрисовка с моделью отражения
    reflection_model = vis_config["reflection_model"]
    material = {
        "ambient": material_config["ambient"],
        "diffuse": material_config["diffuse"],
        "specular": material_config["specular"],
        "shininess": material_config["shininess"],
        "roughness": material_config.get("roughness", 0.5)
    }

    # Вычисление цветов вершин
    # ВАЖНО: Используем исходные координаты вершин (до трансформации) для вычисления освещения
    # но учитываем, что модель будет трансформирована
    vertex_colors = []
    for i, vertex in enumerate(vertices):
        # Трансформируем вершину так же, как она будет отрисована
        transformed_vertex = [
            (vertex[0] - center[0]) * scale,
            (vertex[1] - center[1]) * scale,
            (vertex[2] - center[2]) * scale
        ]

        # Трансформируем позицию света относительно центра и масштаба
        transformed_light = [
            (light_pos["x"] - center[0]) * scale,
            (light_pos["y"] - center[1]) * scale,
            (light_pos["z"] - center[2]) * scale
        ]

        color = calculate_vertex_color(
            transformed_vertex, normals[i],
            transformed_light,
            camera_pos,  # Позиция камеры уже в нормализованных координатах
            material, reflection_model
        )
        # Убеждаемся, что цвета не слишком темные
        color = [max(0.1, min(1.0, c)) for c in color]  # Минимум 0.1, максимум 1.0
        vertex_colors.append(color)

    # Отрисовка
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glDisable(GL_LIGHTING)  # Используем предвычисленные цвета вершин

    # Отладочная информация
    print(f"  Центр модели (исходный): ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
    print(f"  Размер модели: {max_span:.2f}")
    print(f"  Масштаб: {scale:.6f}")
    print(f"  Позиция камеры (нормализованная): ({camera_pos[0]:.2f}, {camera_pos[1]:.2f}, {camera_pos[2]:.2f})")
    print(f"  Расстояние камеры: {camera_distance:.2f}")

    glBegin(GL_TRIANGLES)
    for face in faces:
        if len(face) == 3:
            for vertex_index in face:
                if 0 <= vertex_index < len(vertices):
                    x, y, z = vertices[vertex_index]
                    nx, ny, nz = normals[vertex_index]
                    r, g, b = vertex_colors[vertex_index]
                    glColor3f(r, g, b)
                    glNormal3f(nx, ny, nz)
                    glVertex3f(x, y, z)
    glEnd()

    # Принудительно обновляем буфер
    glFlush()
    glFinish()  # Ждем завершения всех операций

    # Убеждаемся, что мы читаем из правильного буфера
    glReadBuffer(GL_FRONT)

    # Чтение пикселей
    pixels = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)

    # Проверка, что пиксели не пустые
    if pixels:
        # Проверяем, есть ли нечерные пиксели
        non_black_count = sum(1 for i in range(0, len(pixels), 3)
                              if pixels[i] > 10 or pixels[i + 1] > 10 or pixels[i + 2] > 10)
        print(f"  Нечерных пикселей: {non_black_count} из {width * height}")
    else:
        print("  ВНИМАНИЕ: Пиксели не прочитаны!")

    # Сохранение в BMP
    save_bmp(output_filename, width, height, pixels)

    # Закрываем окно BMP перед созданием нового окна для визуализации
    try:
        glutSetWindow(window_id)
        glutHideWindow()
        glutDestroyWindow(window_id)
    except:
        pass

    print(f"BMP файл сохранен: {output_filename}")


def save_bmp(filename, width, height, pixels):
    """Сохранение изображения в формат BMP"""
    # BMP заголовок
    file_size = 54 + width * height * 3
    bmp_header = bytearray(54)
    bmp_header[0:2] = b'BM'  # Сигнатура
    bmp_header[2:6] = struct.pack('<I', file_size)
    bmp_header[10:14] = struct.pack('<I', 54)  # Смещение данных
    bmp_header[14:18] = struct.pack('<I', 40)  # Размер заголовка
    bmp_header[18:22] = struct.pack('<i', width)
    bmp_header[22:26] = struct.pack('<i', height)
    bmp_header[26:28] = struct.pack('<H', 1)  # Плоскости
    bmp_header[28:30] = struct.pack('<H', 24)  # Бит на пиксель
    bmp_header[34:38] = struct.pack('<I', width * height * 3)  # Размер изображения

    # Конвертация пикселей (BMP хранит строки снизу вверх)
    image_data = bytearray()
    row_size = width * 3
    padding = (4 - (row_size % 4)) % 4

    # Конвертация bytes в bytearray если необходимо
    if isinstance(pixels, bytes):
        pixels = bytearray(pixels)

    for y in range(height - 1, -1, -1):
        row_start = y * width * 3
        row = pixels[row_start:row_start + row_size]
        # Конвертация RGB в BGR для BMP
        bgr_row = bytearray()
        for i in range(0, len(row), 3):
            if i + 2 < len(row):
                bgr_row.append(row[i + 2])  # B
                bgr_row.append(row[i + 1])  # G
                bgr_row.append(row[i])  # R
        image_data.extend(bgr_row)
        image_data.extend(bytearray(padding))  # Выравнивание

    # Запись файла
    with open(filename, 'wb') as f:
        f.write(bmp_header)
        f.write(image_data)


# ==================== OPENGL ВИЗУАЛИЗАЦИЯ ====================

if OPENGL_AVAILABLE:
    # Глобальные переменные для OpenGL
    gl_vertices = []
    gl_faces = []
    gl_normals = []
    gl_vertex_colors = []
    gl_rot_x = 55.0
    gl_rot_y = -45.0
    gl_zoom = 2.2
    gl_center = [0.0, 0.0, 0.0]
    gl_scale = 1.0
    gl_config = None


    def compute_normalization():
        """Вычисление центра и масштаба для нормализации"""
        global gl_center, gl_scale

        if not gl_vertices:
            gl_center = [0.0, 0.0, 0.0]
            gl_scale = 1.0
            return

        min_x = min(v[0] for v in gl_vertices)
        max_x = max(v[0] for v in gl_vertices)
        min_y = min(v[1] for v in gl_vertices)
        max_y = max(v[1] for v in gl_vertices)
        min_z = min(v[2] for v in gl_vertices)
        max_z = max(v[2] for v in gl_vertices)

        gl_center[0] = (min_x + max_x) * 0.5
        gl_center[1] = (min_y + max_y) * 0.5
        gl_center[2] = (min_z + max_z) * 0.5

        span_x = max(1e-6, max_x - min_x)
        span_y = max(1e-6, max_y - min_y)
        span_z = max(1e-6, max_z - min_z)
        max_span = max(span_x, span_y, span_z)
        gl_scale = 2.0 / max_span


    def setup_lights_from_config():
        """Настройка освещения из конфигурации"""
        if not gl_config:
            return

        vis_config = gl_config["visualization"]
        light_pos = vis_config["light_position"]

        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, [light_pos["x"], light_pos["y"], light_pos["z"], 0.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
        glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE)
        glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)


    def draw_axes():
        """Отрисовка осей координат"""
        glDisable(GL_LIGHTING)
        glBegin(GL_LINES)
        # X-axis (red)
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(1.0, 0.0, 0.0)
        # Y-axis (green)
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 1.0, 0.0)
        # Z-axis (blue)
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, 1.0)
        glEnd()
        glEnable(GL_LIGHTING)


    def draw_mesh():
        """Отрисовка 3D сетки с моделью отражения"""
        glDisable(GL_LIGHTING)
        glBegin(GL_TRIANGLES)
        for face in gl_faces:
            if len(face) == 3:
                for vertex_index in face:
                    if 0 <= vertex_index < len(gl_vertices):
                        x, y, z = gl_vertices[vertex_index]
                        if vertex_index < len(gl_vertex_colors):
                            r, g, b = gl_vertex_colors[vertex_index]
                            glColor3f(r, g, b)
                        if vertex_index < len(gl_normals):
                            nx, ny, nz = gl_normals[vertex_index]
                            glNormal3f(nx, ny, nz)
                        glVertex3f(x, y, z)
        glEnd()
        glEnable(GL_LIGHTING)


    def display():
        """Функция отрисовки"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Настройка камеры
        glTranslatef(0.0, 0.0, float(-5.0 * gl_zoom))
        glRotatef(gl_rot_x, 1.0, 0.0, 0.0)
        glRotatef(gl_rot_y, 0.0, 1.0, 0.0)
        glScaled(gl_scale, gl_scale, gl_scale)
        glTranslated(-gl_center[0], -gl_center[1], -gl_center[2])

        # Отрисовка
        draw_axes()
        draw_mesh()

        glutSwapBuffers()


    def reshape(width, height):
        """Обработка изменения размера окна"""
        if height == 0:
            height = 1
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, float(width) / float(height), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)


    def keyboard(key, x, y):
        """Обработка клавиатуры"""
        global gl_rot_x, gl_rot_y, gl_zoom

        key = key.decode('utf-8').lower()

        if key in ['q', '\x1b']:  # Q или ESC
            sys.exit(0)
        elif key == 'w':
            gl_zoom = max(0.2, gl_zoom * 0.95)
        elif key == 's':
            gl_zoom = min(10.0, gl_zoom * 1.05)
        elif key == 'r':  # Reset view
            gl_rot_x = 55.0
            gl_rot_y = -45.0
            gl_zoom = 2.2

        glutPostRedisplay()


    def special_keys(key, x, y):
        """Обработка специальных клавиш"""
        global gl_rot_x, gl_rot_y

        step = 5.0
        if key == GLUT_KEY_LEFT:
            gl_rot_y -= step
        elif key == GLUT_KEY_RIGHT:
            gl_rot_y += step
        elif key == GLUT_KEY_UP:
            gl_rot_x -= step
        elif key == GLUT_KEY_DOWN:
            gl_rot_x += step

        glutPostRedisplay()


    def init_opengl():
        """Инициализация OpenGL"""
        glEnable(GL_DEPTH_TEST)
        glShadeModel(GL_SMOOTH)
        glClearColor(0.1, 0.1, 0.1, 1.0)
        setup_lights_from_config()


    def run_opengl_visualization(vertices, faces, normals, config):
        """Запуск OpenGL визуализации"""
        global gl_vertices, gl_faces, gl_normals, gl_vertex_colors, gl_config
        gl_vertices = vertices
        gl_faces = faces
        gl_normals = normals
        gl_config = config

        # Вычисление цветов вершин
        vis_config = config["visualization"]
        material_config = config["material"]
        light_pos = vis_config["light_position"]
        viewer_pos = vis_config["viewer_position"]
        reflection_model = vis_config["reflection_model"]

        material = {
            "ambient": material_config["ambient"],
            "diffuse": material_config["diffuse"],
            "specular": material_config["specular"],
            "shininess": material_config["shininess"],
            "roughness": material_config.get("roughness", 0.5)
        }

        gl_vertex_colors = []
        for i, vertex in enumerate(vertices):
            color = calculate_vertex_color(
                vertex, normals[i],
                [light_pos["x"], light_pos["y"], light_pos["z"]],
                [viewer_pos["x"], viewer_pos["y"], viewer_pos["z"]],
                material, reflection_model
            )
            gl_vertex_colors.append(color)

        compute_normalization()

        # Инициализация GLUT для визуализации
        global glut_initialized

        # Если GLUT уже инициализирован (для BMP), создаем новое окно
        # Если нет - инициализируем заново
        if not glut_initialized:
            glutInit(sys.argv)
            glut_initialized = True

        # Создаем новое окно для визуализации
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        glutInitWindowSize(900, 600)
        vis_window_id = glutCreateWindow(b"Depth Map Visualizer")

        # КРИТИЧНО: Устанавливаем активное окно ПЕРЕД регистрацией callbacks
        glutSetWindow(vis_window_id)

        init_opengl()

        # Регистрация callback-функций для текущего активного окна
        # Должно быть после glutSetWindow
        glutDisplayFunc(display)
        glutReshapeFunc(reshape)
        glutKeyboardFunc(keyboard)
        glutSpecialFunc(special_keys)

        # Убеждаемся, что мы все еще на правильном окне
        current_window = glutGetWindow()
        if current_window != vis_window_id:
            glutSetWindow(vis_window_id)

        # Принудительно вызываем первый рендер
        glutPostRedisplay()

        print("\nУправление в OpenGL:")
        print("   Стрелки - вращение")
        print("   W/S - приближение/отдаление")
        print("   R - сброс вида")
        print("   Q/ESC - выход")
        print(f"   Модель отражения: {reflection_model}")

        glutMainLoop()


# ==================== ОСНОВНАЯ ПРОГРАММА ====================

def main():
    """Основная программа"""
    config_file = "config.json"

    try:
        # 1. Загрузка конфигурации
        print("Загрузка конфигурации...")
        config = load_config(config_file)
        print(f"Конфигурация загружена из {config_file}")

        # 2. Чтение карты глубины
        input_filename = config["input"]["depth_map_file"]
        if not os.path.exists(input_filename):
            print(f"Файл {input_filename} не найден!")
            print("Убедитесь, что файл находится в той же папке, что и программа")
            return

        print(f"\nЧтение карты глубины из {input_filename}...")
        width, height, depth_data = read_depth_map(input_filename)

        # 3. Создание 3D модели
        print("\nСоздание 3D модели...")
        vertices, faces = create_3d_model_from_depth_map(depth_data, width, height)

        if not vertices:
            print("Нет данных для создания модели")
            return

        # 4. Вычисление нормалей
        print("\nВычисление нормалей...")
        normals = compute_vertex_normals(vertices, faces)

        # 5. Экспорт в выбранный формат(ы)
        output_config = config["output"]
        material_config = config.get("material", {})

        # Проверяем, нужно ли экспортировать во все форматы
        export_all = output_config.get("export_all_formats", False)

        if export_all:
            # Экспорт во все доступные форматы
            formats = output_config.get("formats_available", ["vrml", "obj", "ply"])
            print(f"\nЭкспорт во все форматы: {', '.join(formats)}...")

            exported_files = []
            for fmt in formats:
                output_filename = f"{output_config['filename']}.{fmt}"
                print(f"  Экспорт в {fmt.upper()}...")

                try:
                    if fmt == "vrml":
                        export_to_vrml(vertices, faces, output_filename, material_config)
                    elif fmt == "obj":
                        export_to_obj(vertices, faces, output_filename)
                    elif fmt == "ply":
                        export_to_ply(vertices, faces, output_filename)
                    else:
                        print(f"    Неизвестный формат: {fmt}, пропущен")
                        continue

                    exported_files.append(output_filename)
                    print(f"    ✓ {output_filename} создан")
                except Exception as e:
                    print(f"    ✗ Ошибка при экспорте в {fmt}: {e}")

            print(f"\nУспешно экспортировано файлов: {len(exported_files)}")
            if exported_files:
                print(f"Файлы: {', '.join(exported_files)}")
        else:
            # Экспорт в один выбранный формат (как раньше)
            output_format = output_config["format"].lower()
            output_filename = f"{output_config['filename']}.{output_format}"

            print(f"\nЭкспорт в формат {output_format.upper()}...")

            if output_format == "vrml":
                export_to_vrml(vertices, faces, output_filename, material_config)
            elif output_format == "obj":
                export_to_obj(vertices, faces, output_filename)
            elif output_format == "ply":
                export_to_ply(vertices, faces, output_filename)
            else:
                print(f"Неизвестный формат: {output_format}")
                print(f"Доступные форматы: {', '.join(output_config['formats_available'])}")
                return

            print(f"Файл {output_filename} успешно создан")

        print(f"Результат: {len(vertices)} вершин, {len(faces)} треугольников")

        # 6. Экспорт в BMP (если включен)
        vis_config = config["visualization"]
        if vis_config.get("export_bmp", False) and OPENGL_AVAILABLE:
            print(f"\nЭкспорт в BMP...")
            bmp_filename = vis_config.get("bmp_filename", "output.bmp")
            export_to_bmp(vertices, faces, normals, bmp_filename, config)

        # 7. Визуализация в OpenGL для проверки (если включена)
        show_visualization = vis_config.get("show_opengl_visualization", True)
        if show_visualization and OPENGL_AVAILABLE:
            print("\nЗапуск OpenGL визуализации для проверки...")
            run_opengl_visualization(vertices, faces, normals, config)
        elif not OPENGL_AVAILABLE:
            print("\nOpenGL не доступен для визуализации")
            print("Для установки библиотек OpenGL выполните в терминале:")
            print("  pip install PyOpenGL PyOpenGL-GLUT")
            print("Или если используете виртуальное окружение:")
            print("  .venv\\Scripts\\pip install PyOpenGL PyOpenGL-GLUT")
        else:
            print("\nOpenGL визуализация отключена в конфигурации")

    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
