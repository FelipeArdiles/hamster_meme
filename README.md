# Hampter Vision

App en Python que detecta expresiones faciales y gestos de mano con MediaPipe para mostrar memes de hampter en tiempo real desde la camara.

## Ejecutar en 30 segundos

1. Descarga o clona este repo.
2. Abre una terminal en la carpeta del proyecto.
3. Ejecuta:

```bash
./scripts/run_hampter.sh
```

Eso es todo.  
En la primera ejecucion el script:

- crea el entorno virtual,
- instala dependencias,
- y luego abre la app.

En siguientes ejecuciones solo abre la app y sera mucho mas rapido.

## Opcion doble clic (macOS)

Puedes abrir `run.command` con doble clic.

## Ejecutar en Windows

### Opcion 1 (PowerShell)

1. Abre **PowerShell** en la carpeta del proyecto.
2. Ejecuta:

```powershell
python -m venv .venv_hampter
.\.venv_hampter\Scripts\Activate.ps1
pip install -r requirements.txt
python .\src\app.py
```

En siguientes ejecuciones:

```powershell
.\.venv_hampter\Scripts\Activate.ps1
python .\src\app.py
```

### Opcion 2 (doble clic)

Tambien puedes crear un archivo `run_windows.bat` en la raiz con este contenido:

```bat
@echo off
if not exist ".venv_hampter\Scripts\python.exe" (
  py -m venv .venv_hampter
  .\.venv_hampter\Scripts\python.exe -m pip install --upgrade pip
  .\.venv_hampter\Scripts\pip.exe install -r requirements.txt
)
.\.venv_hampter\Scripts\python.exe .\src\app.py
```

## Personalizar imagenes

Pon tus memes en `assets/images/` y actualiza los diccionarios y reglas en `src/app.py`:

- `EXPRESSION_IMAGES`
- `HAND_GESTURE_IMAGES`
- `TRIGGER_RULES`

## Estructura del proyecto

- `src/` - codigo principal
- `assets/images/` - memes e imagenes de expresion/gesto
- `assets/models/` - modelos `.task` de MediaPipe
- `scripts/` - script principal de ejecucion

