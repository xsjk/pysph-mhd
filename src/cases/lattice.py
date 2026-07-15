import numpy as np


def close_packed_lattice(bounds, spacing):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    nx = int(np.ceil(np.nextafter((xmax - xmin) / spacing, -np.inf)))
    ny = int(np.ceil(np.nextafter((ymax - ymin) / (spacing * np.sqrt(3.0 / 4.0)), -np.inf)))
    nz = int(np.ceil(np.nextafter((zmax - zmin) / (spacing * np.sqrt(6.0) / 3.0), -np.inf)))
    ny = 2 * (ny // 2)
    nz = 3 * (nz // 3)
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    dz = (zmax - zmin) / nz
    k, layer_y, layer_z = np.meshgrid(
        np.arange(nx),
        np.arange(1, ny + 1),
        np.arange(1, nz + 1),
        indexing="ij",
    )
    layer_y_parity = layer_y % 2
    layer_z_modulo = layer_z % 3
    second_layer = 2
    x_offset = np.where((layer_z_modulo == 0) & (layer_y_parity == 0), 0.5, 0.0)
    x_offset = np.where((layer_z_modulo == second_layer) & (layer_y_parity == 1), 0.5, x_offset)
    x_offset = np.where((layer_z_modulo == 1) & (layer_y_parity == 0), 0.5, x_offset)
    y_offset = np.where(
        layer_z_modulo == 0,
        2.0 / 3.0,
        np.where(layer_z_modulo == second_layer, 1.0 / 3.0, 0.0),
    )
    x = xmin + 0.25 * spacing + (k + x_offset) * dx
    y = ymin + (layer_y - 1.0 + 1.0 / 6.0 + y_offset) * dy
    z = zmin + (layer_z - 0.5) * dz
    return x.ravel(), y.ravel(), z.ravel()
