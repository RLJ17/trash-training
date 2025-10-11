import subprocess
import shutil
import sys

def get_cuda_version() -> str | None:
    """Devuelve la versión CUDA del driver (p.ej. '12.9') o None si no hay nvidia-smi."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.check_output(["nvidia-smi"], text=True, stderr=subprocess.STDOUT)
        for line in out.splitlines():
            if "CUDA Version" in line:
                # Ej: 'CUDA Version: 12.9'
                v = line.split("CUDA Version:")[-1].split()[0].strip()
                return v
    except Exception:
        return None
    return None

def _parse_ver(v: str) -> tuple[int, int]:
    major, minor = v.split(".")[:2]
    return int(major), int(minor)

def _select_pytorch_wheel_for_driver(cuda_driver: str) -> str:
    """
    Selecciona el wheel de PyTorch soportado más cercano <= versión de driver.
    Mapea 12.9 -> cu128, 12.8 -> cu128, 12.7/12.6 -> cu126, 11.8 -> cu118, etc.
    """
    if not cuda_driver:
        return "cpu"

    drv = _parse_ver(cuda_driver)

    # Ordenado de mayor a menor soporte conocido
    supported = [
        (("12","8"), "cu128"),
        (("12","6"), "cu126"),
        (("11","8"), "cu118"),
    ]

    for (maj, minr), tag in supported:
        if drv >= (int(maj), int(minr)):
            return tag

    return "cpu"

def _pip_run(args: list[str]) -> int:
    print(">", " ".join(args))
    return subprocess.call(args)

def install_pytorch_cuda():
    """
    Detecta versión CUDA (driver) y instala torch/vision/audio con el wheel adecuado.
    - 12.9/12.8 -> cu128
    - 12.6/12.7 -> cu126
    - 11.8      -> cu118
    Si no hay match, instala CPU.
    """
    cuda_driver = get_cuda_version()
    wheel = _select_pytorch_wheel_for_driver(cuda_driver)

    print("🚀 Instalando PyTorch")
    print(f"• CUDA (driver) detectada: {cuda_driver or 'No disponible'}")
    print(f"• Python: {sys.version.split()[0]}")

    # Aviso por Python 3.12 (mejor 3.10/3.11 para CUDA)
    if sys.version_info >= (3,12):
        print("⚠️ Aviso: Python 3.12 puede no tener wheels CUDA completos. "
              "Si falla la detección de GPU tras instalar, usa Python 3.10/3.11.")

    # Desinstalar posibles instalaciones previas
    _pip_run([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"])

    if wheel == "cpu":
        print("➡️ Instalando build CPU (no CUDA).")
        code = _pip_run([sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"])
    else:
        idx = f"https://download.pytorch.org/whl/{wheel}"
        print(f"➡️ Instalando build CUDA: {wheel} (driver {cuda_driver})")
        code = _pip_run([sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio",
                         "--index-url", idx])

    # Verificación
    try:
        import torch
        print("Torch:", torch.__version__)
        print("torch.version.cuda:", torch.version.cuda)
        print("CUDA disponible:", torch.cuda.is_available())
        print("GPUs detectadas:", torch.cuda.device_count())
        if torch.cuda.is_available():
            print("GPU 0:", torch.cuda.get_device_name(0))
        else:
            print("⚠️ GPU no detectada por PyTorch. Si estás en Python 3.12, "
                  "considera usar Python 3.10/3.11 y reinstalar con cu128.")
    except Exception as e:
        print("❌ Error importando torch tras la instalación:", e)

    return {"cuda_driver": cuda_driver, "selected_wheel": wheel}

def install_dependencies():
    """
    Instala PyTorch (CUDA si procede) y luego requirements.txt
    """
    print("=" * 60)
    info = install_pytorch_cuda()
    print("\n📦 Instalando paquetes de requirements.txt…")
    code = _pip_run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    if code == 0:
        print("✅ Dependencias instaladas.")
    else:
        print("❌ Error instalando requirements.txt")
    print(f"Resumen → CUDA driver: {info['cuda_driver']} | wheel: {info['selected_wheel']}")
    print("=" * 60)