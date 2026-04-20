# Hampter Vision

App en Python que detecta expresiones faciales y gestos de mano con MediaPipe para mostrar memes de hampter en tiempo real desde la camara.

## Estructura

- `src/` - codigo principal
- `assets/images/` - memes e imagenes de expresion/gesto
- `assets/models/` - modelos `.task` de MediaPipe
- `scripts/` - scripts de ejecucion
- `requirements.txt` - dependencias

## Requisitos

- Python 3.10+
- Webcam habilitada
- macOS/Windows/Linux con entorno grafico

## Instalacion

```bash
python3 -m venv .venv_hampter
source .venv_hampter/bin/activate
pip install -r requirements.txt
```

## Ejecutar

```bash
./scripts/run_hampter.sh
```

## Personalizar imagenes

Pon tus memes en `assets/images/` y actualiza los diccionarios y reglas en `src/app.py`:

- `EXPRESSION_IMAGES`
- `HAND_GESTURE_IMAGES`
- `TRIGGER_RULES`

