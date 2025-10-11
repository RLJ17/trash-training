from pathlib import Path
from ultralytics import YOLO

def export_all_to_tflite(export_dir: str = "exports/tflite_models"):
    root_dir = Path.cwd()
    projects_dir = root_dir / "projects"
    export_dir = root_dir / export_dir
    export_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔍 Buscando modelos en: {projects_dir}")
    print(f"📁 Guardando modelos .tflite en: {export_dir}")

    for project in projects_dir.iterdir():
        weights_path = project / "weights" / "best.pt"
        if weights_path.exists():
            try:
                print(f"📦 Exportando: {weights_path}")
                model = YOLO(str(weights_path))
                exported = model.export(format="tflite")
                exported_path = Path(exported)
                
                # Mueve el archivo exportado a la carpeta destino
                final_path = export_dir / f"{project.name}.tflite"
                exported_path.rename(final_path)

                print(f"✅ Guardado como: {final_path}")
            except Exception as e:
                print(f"❌ Error al exportar {weights_path}: {e}")
        else:
            print(f"⚠️ No se encontró 'best.pt' en: {project.name}")