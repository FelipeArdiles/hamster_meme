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

