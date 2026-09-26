# Validación de la entrega

Fecha: 26 de septiembre de 2026. Entorno ejecutado: macOS Apple Silicon, Python 3.11, Gradio 5.49.1.

| Comprobación | Resultado |
| --- | --- |
| `python -m pytest -q` | 75 pruebas aprobadas, sin advertencias |
| `python -m compileall -q app tests` | Sin errores |
| `ruff check app tests --select F` | Sin errores |
| `python -m pip check` | Dependencias instaladas compatibles |
| Construcción `create_app()` | Correcta, 166 componentes |
| Arranque `python -m app.main` | Correcto en `127.0.0.1:7860` |
| HTTP `/config` | Respuesta correcta; analytics desactivados |
| Chrome headless, imagen sintética | Importar → exposición → deshacer → retoque → guardar copia: aprobado |
| Errores JavaScript durante el flujo | Ninguno |
| Solicitudes externas del navegador durante el flujo | Ninguna |

La prueba de navegador detectó y permitió corregir la inicialización diferida de ImageEditor. También detectó el script CDN y los preconnects que incluye la plantilla de Gradio: se eliminan de la plantilla en memoria, sin modificar site-packages. Una cabecera CSP restringe recursos y conexiones del navegador al origen local. Se añaden pruebas de regresión para esa plantilla y las cabeceras.

## Alcance implementado

Motor convencional, original/resultado acumulado, historial de 25 estados, undo/reset, guardado de copias, color/HSL/balance de blancos, retoque, pinceles, Telea, clonado, fondos, calibración de cinco regiones, reshape, Liquify, transformaciones, privacidad/marcas, variantes, proyectos atómicos con autoguardado, presets y lotes. Todo está conectado a la interfaz.

U²-Net Human + ONNX Runtime y RealESRGAN_x2plus + Spandrel están integrados con carga lazy y rutas locales. Upscale incluye tiles, alfa, checksum, selección CUDA/MPS/CPU y fallback de MPS. Las pruebas usan segmentación y upscale simulados, e incluyen fallback simulado de MPS a CPU.

## Lo que no se ha validado con hardware/pesos reales

No se descargaron pesos ni se instalaron las dependencias opcionales de segmentación/upscale. La inferencia real necesita completar los pasos de modelos del README. No se afirma una prueba de calidad de segmentación ni de RealESRGAN real, rendimiento GPU o funcionamiento físico en Fedora y macOS Intel.

La fase generativa (Stable Diffusion / inpainting por prompt) permanece pausada por diseño. No contiene dependencias ni descargas.

## Repetir el navegador opcional

La dependencia de prueba `playwright` está instalada en el entorno de esta entrega; no es necesaria para ejecutar el editor:

```bash
python -m pip install playwright
python -m app.main
# En otra terminal, con el servidor activo:
python tests/browser_smoke.py
```

Usa Chrome instalado en macOS. En Linux configura `LPL_CHROME` con la ruta de Chromium/Chrome instalado. La prueba crea una imagen sintética temporal, guarda una copia de prueba en `outputs/` y una captura en `/tmp/local-photo-lab-smoke.png`. No usa fotografías reales ni descarga un navegador.
