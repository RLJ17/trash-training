import os
import yaml

def update_data_yalm_routes(yaml_path: str, guardar_como: str = None):
    """
    Actualiza las rutas de train, val y test en un archivo data.yaml
    reemplazando los '..' por rutas absolutas.
    """
    base_dir = os.path.abspath(os.path.dirname(yaml_path))

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    data["train"] = os.path.join(base_dir, "train", "images")
    data["val"] = os.path.join(base_dir, "valid", "images")
    data["test"] = os.path.join(base_dir, "test", "images")

    output_path = guardar_como if guardar_como else yaml_path

    with open(output_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"✅ Rutas actualizadas en: {output_path}")
    print("📂 Nuevas rutas:")
    print(f"  train -> {data['train']}")
    print(f"  val   -> {data['val']}")
    print(f"  test  -> {data['test']}")
