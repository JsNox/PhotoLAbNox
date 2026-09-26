# Local Photo Lab

**Local Photo Lab** es un estudio de edición fotográfica privado que funciona en tu equipo, con una interfaz en español y un flujo de trabajo pensado para macOS y Linux. Permite ajustar luz y color, retocar piel, editar fondos y siluetas, aplicar correcciones localizadas y preparar imágenes para publicación, conservando siempre el original.

Organiza tus sesiones en proyectos con historial y autoguardado, reutiliza estilos mediante presets y exporta fotografías por lotes, con marcas de agua y variantes de privacidad. Está construido con Python 3.11 y Gradio; incorpora segmentación de personas con U²-Net y ampliación 2× con RealESRGAN como módulos opcionales de procesamiento local.

Las fotografías permanecen en el equipo: sin procesamiento en la nube, APIs externas ni telemetría activa. Orientado a la producción de contenido para adultos, exclusivamente con fotografías legales de personas adultas y con consentimiento. No reconoce identidades ni regenera rostros.

Repositorio: [JsNox/PhotoLAbNox](https://github.com/JsNox/PhotoLAbNox).

## Inicio rápido

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m app.main
```

Abre **http://127.0.0.1:7860**. Detén con `Ctrl+C`. Si el puerto está ocupado, cierra la otra instancia: la aplicación nunca busca otro puerto ni escucha en la red local. Ejecuta desde la raíz del repositorio.

La edición convencional funciona con `requirements.txt`. Segmentación y upscale requieren sus paquetes y pesos adicionales. La aplicación **no descarga pesos**, ni siquiera cuando faltan. Instalar paquetes y obtener modelos son pasos de preparación con conexión; después, el editor funciona sin Internet.

## Instalación en macOS

1. Instala Homebrew siguiendo su [sitio oficial](https://brew.sh/).
2. Instala Python 3.11:

```bash
brew install python@3.11
"$(brew --prefix python@3.11)/bin/python3.11" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Apple Silicon: usa una terminal nativa ARM64, evitando mezclar Python Intel/Rosetta con paquetes ARM64. PyTorch selecciona MPS cuando está disponible y CPU si no. Un fallo de ejecución en MPS reintenta en CPU FP32. MPS es opcional; los ajustes convencionales y ONNX trabajan en CPU.

Intel: la edición base y U²-Net funcionan en CPU. Las versiones recientes de PyTorch no siempre publican ruedas macOS x86_64. Si no hay una rueda compatible, usa el entorno opcional siguiente (Spandrel acepta esta versión de PyTorch); no intentes instalar el archivo general de upscale encima:

```bash
python -m pip install 'numpy<2' 'opencv-python-headless==4.10.0.84'
python -m pip install 'torch==2.2.2' 'torchvision==0.17.2' 'spandrel==0.4.1'
```

La compatibilidad Intel de este entorno se documenta como alternativa y no se ha ejecutado en este equipo Apple Silicon. Si el sistema operativo no dispone de ruedas compatibles, puedes seguir usando todas las herramientas convencionales.

## Instalación en Fedora / Linux

Con Python 3.11 instalado en tu distribución:

```bash
sudo dnf install python3.11 python3.11-devel
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

El nombre/disponibilidad del paquete Python 3.11 depende de la versión de Fedora. Si no está en sus repositorios, instala Python 3.11 con un gestor de versiones en tu cuenta, sin reemplazar el Python del sistema. OpenCV es headless: no necesita un escritorio X11 para procesar imágenes.

Para upscale **sin NVIDIA**, instala primero las ruedas CPU:

```bash
python -m pip install 'torch>=2.6,<3' torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-upscale.txt
```

Con NVIDIA, instala PyTorch para la versión CUDA compatible siguiendo [PyTorch](https://pytorch.org/get-started/locally/) y después `requirements-upscale.txt`. CUDA usa FP16. Fedora/Linux no se ha validado físicamente en este equipo; las pruebas no dependen de macOS.

## Modelos locales

### U²-Net Human Segmentation

```bash
python -m pip install -r requirements-background.txt
mkdir -p models/rembg
```

Obtén `u2net_human_seg.onnx` de la [distribución oficial de rembg](https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_human_seg.onnx) y colócalo en:

```text
models/rembg/u2net_human_seg.onnx
```

La versión fijada de rembg aporta normalización y postprocesado. Se construye directamente la sesión ONNX CPU con la ruta local para evitar su mecanismo de descarga. Solo se carga cuando pulsas **Segmentar persona** o aplicas un preset con silueta en un lote. El [adaptador oficial de rembg](https://github.com/danielgatis/rembg/blob/v2.0.67/rembg/sessions/u2net_human_seg.py) documenta el peso de segmentación humana.

Sin este modelo, pinta **Máscara manual de persona**. Puedes usar esa máscara para fondo, calibración, reshape y Liquify. Los presets de lotes que incluyen silueta sí necesitan segmentación automática.

### RealESRGAN_x2plus — Fase 6

Apple Silicon / entorno PyTorch compatible:

```bash
python -m pip install -r requirements-upscale.txt
mkdir -p models/upscale
```

Obtén el peso de la [release oficial de Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.1), archivo `RealESRGAN_x2plus.pth`, y guárdalo exactamente en:

```text
models/upscale/RealESRGAN_x2plus.pth
```

SHA-256 exigido antes de cargar el peso:

```text
49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb
```

Verifica también manualmente si lo deseas:

```bash
# macOS
shasum -a 256 models/upscale/RealESRGAN_x2plus.pth
# Linux
sha256sum models/upscale/RealESRGAN_x2plus.pth
```

Carga lazy mediante [Spandrel](https://github.com/chaiNNer-org/spandrel). Tiles 128/256/384/512 con solapamiento y mezcla gradual. Conserva alfa, interpola ese canal y duplica RGB mediante el modelo. CUDA FP16; MPS y CPU FP32. Si MPS produce un error de ejecución, reintenta automáticamente con CPU. Máximo 64 millones de píxeles de salida: una entrada mayor de 16 MP se rechaza para upscale. Comienza con 128 si falta memoria. El ensamblado de la imagen usa RAM incluso cuando los tiles se procesan en GPU.

## Flujo recomendado

1. Importa JPG/JPEG, PNG o WEBP; se normaliza EXIF y, si es posible, se convierte el ICC a sRGB. Se rechazan archivos mayores de 40 MB o 24 MP antes de decodificarlos completamente.
2. El original y el resultado acumulado permanecen visibles. Cada botón aplica una operación al resultado actual, sin escribir encima del archivo importado.
3. Ajusta exposición, color y piel. **Color inteligente** combina CLAHE moderado, sombras y enfoque suaves. Si los cambios manuales son casi imperceptibles, el estado muestra una advertencia.
4. Segmenta/calibra si vas a editar silueta. Aplica retoque local y eliminación de objetos.
5. Realiza las transformaciones y el fondo al final. Los cambios geométricos invalidan la máscara: vuelve a segmentar cuando quieras seguir con herramientas de persona.
6. Aplica upscale si necesitas aumentar resolución. Revisa al 100 %.
7. Guarda copia o variantes. No hay exportación automática sin tu acción.

**Deshacer** recupera hasta 25 estados anteriores comprimidos en memoria. El registro conserva los nombres de todas las operaciones aplicadas que siguen vigentes. **Restablecer** vuelve al original y es una operación deshacible. Importar otra fotografía inicia una sesión nueva y desactiva el autoguardado del proyecto anterior. Los controles conservan sus valores para copiarlos entre fotografías.

## Piel, máscaras y clics

- Retoque: máscara aproximada YCrCb + HSV, filtro bilateral, mediana para imperfecciones pequeñas y reinyección de textura. Es una heurística sensible a iluminación y tono de piel; no reconoce personas.
- **Piel y pinceles**: pinta con blanco sobre una capa transparente. El alfa del trazo define la selección. El borrador elimina trazos; aplicar limpia la capa y sincroniza el lienzo con el resultado.
- **Aclarar / oscurecer**: cantidad positiva aclara; negativa oscurece. Actúa en L de LAB y conserva crominancia, con borde suavizado. Útil para sombras localizadas. Evita cantidades extremas que recorten luces.
- **Healing Telea**: reconstruye solo la máscara y su expansión. Radio controla la vecindad, expansión cubre bordes del objeto. No reconstruye semánticamente cuerpos u objetos grandes.
- **Clone Stamp**: selecciona la acción en Fondo y silueta y pulsa origen/destino en el resultado o el lienzo de clics. Radio y opacidad son independientes de Liquify.
- **Balance de blancos**: elige la acción y pulsa una zona gris o blanca sin recorte. Promedia una pequeña vecindad; muestras demasiado oscuras se rechazan.
- **HSL**: todos los colores o rangos selectivos con transiciones suaves; tono, saturación y luminosidad.
- **Privacidad**: desenfoque o pixelado en la máscara, intensidad y borde configurables. Una selección mal pintada puede dejar partes sin cubrir: revisa antes de exportar.

## Eliminar objetos del fondo

Abre **Eliminar objetos**, pinta sobre el objeto y pulsa **Eliminar objeto seleccionado**. Usa el borrador o **Limpiar selección** para corregir los trazos. La expansión incluye el contorno y el radio regula la reconstrucción con Telea. Funciona sin modelos ni conexión y se aplica al resultado acumulado, con historial, Deshacer y autoguardado del proyecto activo.

Puedes activar **Proteger persona y objetos conservados** tras segmentar o pintar una máscara de persona. La protección se aplica después de expandir la selección para evitar que el relleno invada la silueta. Sin protección, se modifica únicamente la selección expandida: pinta con cuidado cerca de la persona. Los objetos grandes y fondos complejos pueden dejar artefactos; utiliza Clone Stamp para corregir detalles.

## Fondo, silueta y calibración

La máscara de U²-Net se muestra y puede ampliarse con el pincel **Conservar objeto**. La máscara combinada se aplica automáticamente a fondo transparente, sólido, desenfocado o fotográfico. El fondo fotográfico se ajusta al encuadre manteniendo proporción. Usa PNG/WEBP para transparencia.

El botón de estimación propone hombros, busto, cintura, cadera y límite inferior a partir del contorno. No es detección anatómica. Para calibrar, selecciona **Calibrar cinco regiones** y pulsa cinco veces de arriba hacia abajo sobre el torso. Puedes reiniciar los clics. Cada región obtiene altura, centro, ancho y bordes del tramo del contorno seleccionado. Las líneas terminan en la silueta. En poses con brazos separados, pulsa sobre el tramo central del torso para evitar elegir un brazo.

Busto, cintura y cadera aceptan −20 a +20. Curvas 0–20 estrecha ligeramente cintura y amplía ligeramente cadera. Intensidad predeterminada: 60 %. Los porcentajes efectivos se muestran en el estado y se limitan a ±18 %. Por debajo de 25 % se advierte del efecto pequeño. Los valores describen la ganancia solicitada al campo suave; el desplazamiento real en el borde puede ser menor por la atenuación espacial.

Se interpolan centros y anchos independientes para seguir el torso inclinado. El campo comienza en hombros y termina en el límite inferior; **calibra correctamente para dejar la cara fuera**. No hay detector de rostro, por lo que una máscara o calibración incorrecta requiere corrección manual.

Antes de deformar se crea una placa de fondo con Telea sobre la máscara completa de persona, a resolución de trabajo limitada a 768 px en su lado mayor. La persona deformada se compone sobre ella; el fondo fuera de la unión de siluetas permanece intacto. Fondos complejos, personas que llenan todo el encuadre, cabellos y grandes reducciones pueden producir artefactos; utiliza cambios moderados o un fondo sólido nuevo.

## Liquify

Selecciona **Liquify Push / Drag**: primer clic origen, segundo destino. **Liquify Restore** utiliza un clic para reducir el desplazamiento hacia la referencia inicial de la sesión. Pincel recomendado 70–120 px, fuerza 10–25. Campos suaves con desplazamiento limitado y aplicación dentro de la máscara de persona; la composición usa la placa limpia.

**Tamaño y Fuerza solo afectan Liquify**, nunca los controles de reshape automático. La sesión acumula campos sin volver a muestrear repetidamente la referencia. Aplicar otra herramienta o deshacer reinicia la referencia; para recuperar estados anteriores usa Deshacer. El radio se mide en píxeles de imagen, no en píxeles de pantalla.

## Proyectos `.lpl`

En Proyectos y presets, escribe un nombre y pulsa **Guardar / activar autoguardado**. Cada edición posterior, deshacer o restablecer actualiza el archivo. Cargar un proyecto también activa el autoguardado. Puedes desactivarlo explícitamente.

`projects/<nombre-sanitizado>.lpl` es un ZIP versionado con manifiesto JSON, original, resultado y hasta 25 imágenes de historial. Conserva formato de origen, ICC válido, EXIF original y nombres de operaciones. No usa pickle ni ejecuta contenido del proyecto. Se guarda con archivo temporal, `fsync` y reemplazo atómico. Se rechazan versiones incompatibles, archivos dañados, rutas simbólicas y tamaños excesivos. No se extraen rutas del ZIP al disco.

Las máscaras, clics y campos de Liquify son auxiliares temporales; al cargar un proyecto o deshacer se recalculan antes de volver a esas herramientas. Los proyectos contienen información privada **sin cifrado**. El archivo puede ser grande: el historial conserva estados completos para que Deshacer sea exacto. Una copia exportada no incluye el historial.

## Presets y lotes

Incluye **Natural, Cálido, Nocturno y Editorial**. **Copiar preset a controles** no aplica cambios; **Aplicar preset completo** ejecuta color, retoque y silueta en una operación. Guarda presets propios para color, intensidad/textura de piel, busto, cintura, cadera, curvas e intensidad automática. Se almacenan como JSON en `presets/`, con nombres sanitizados y valores validados. No sobrescribe los cuatro incluidos.

Los lotes admiten 1–100 archivos importados mediante la interfaz, procesados secuencialmente. Aplican el preset seleccionado y su retoque, además de marca opcional. Un preset con deformación requiere el modelo de segmentación. Exportan un ZIP con nombres internos, JPEG calidad 92, fondo blanco si había alfa y sin EXIF. Si falla una entrada, se elimina el ZIP incompleto y se informa del error; no se modifican las fuentes.

## Exportación y marcas

**Guardar copia** crea un nombre UUID en `outputs/`. PNG y WEBP conservan alfa; JPEG lo compone sobre blanco. Se puede quitar EXIF/GPS (opción activada por defecto), manteniendo el perfil ICC de color cuando es válido. Desactivar la opción permite conservar EXIF disponible, incluida información privada: úsalo solo cuando lo necesites.

La marca permite texto, tamaño, opacidad y cinco posiciones. Puedes aplicarla al resultado —entra en historial— o guardar tres variantes sin modificarlo: editada, con marca y teaser difuminado en toda la imagen. El teaser es un desenfoque completo, no una garantía de anonimización. Las fuentes básicas pueden no contener todos los glifos Unicode y el texto largo puede salirse de imágenes pequeñas.

## Privacidad y seguridad

- Enlace local fijo `127.0.0.1:7860`; `share=False`, analytics y monitorización desactivados, CORS estricto. No se crean enlaces públicos.
- Fuentes del sistema y recursos servidos localmente. Se eliminan el script CDN y los preconnects externos de la plantilla de Gradio en memoria. Una cabecera CSP restringe recursos y conexiones del navegador al origen local. No hay APIs externas, almacenamiento en nube ni modelos generativos.
- Variables offline / no telemetry antes de importar Gradio; telemetría de ONNX Runtime desactivada explícitamente. El punto de entrada bloquea DNS y conexiones TCP externas de Python como defensa adicional. No es un firewall del sistema; no controla extensiones del navegador ni otras aplicaciones.
- Gradio utiliza `.cache/` para importaciones y vistas previas. **Esta caché temporal se escribe en disco aunque no guardes un proyecto**: es necesaria para la interfaz. Caduca tras una hora según el mecanismo de Gradio; ciérralo y borra `.cache/` si necesitas eliminar temporales inmediatamente.
- Solo `outputs/` está en la lista adicional de rutas servibles. Proyectos, modelos, `.git` y entorno virtual están bloqueados. Las importaciones deben residir en la caché de Gradio; nunca se acepta una ruta arbitraria enviada a un callback como fotografía.
- Directorios de trabajo con permisos privados (0700). Sin secretos ni fotografías reales de prueba. Proyectos, outputs, modelos, presets y caché se excluyen de Git, salvo `.gitkeep`.
- No se cifra el disco ni se controla una copia de seguridad del sistema. Mantén este directorio fuera de servicios de sincronización si necesitas almacenamiento exclusivamente local. Usa un equipo y navegador de confianza; el servidor local no tiene autenticación entre usuarios del mismo equipo.

## Arquitectura

```text
app/main.py                punto de entrada y restricciones de red
app/core/                  operaciones, historial, proyectos, presets y lotes
app/models/model_manager.py rutas de pesos y carga lazy
app/ui/gradio_app.py        interfaz y callbacks en español
app/utils/                 imágenes, dispositivos, rutas y privacidad
tests/                     pruebas con imágenes sintéticas
models/                    pesos opcionales locales
outputs/                   copias y ZIP solicitados
projects/                  proyectos .lpl
presets/                   presets personales JSON
```

Requisitos separados: base, background, upscale y AI. `requirements-ai.txt` contiene solo un comentario: **Stable Diffusion e inpainting por prompt están pausados**, sin dependencias ni pesos. Telea y Clone Stamp cubren edición convencional localizada.

## Pruebas y validación

```bash
python -m pytest -q
python -m compileall -q app tests
python -c 'from app.ui.gradio_app import create_app; create_app(); print("Interfaz construida")'
python -m app.main
# En otra terminal:
curl -f http://127.0.0.1:7860/config
```

La suite cubre neutralidad, orientación, historial, máscaras/segmentación simulada, fondos, piel, dodge & burn, HSL, balance de blancos, censura, marcas, transformaciones, silueta, expansión de busto, calibración, Liquify, proyectos corruptos, autoguardado, presets, lotes, variantes, metadatos, rutas y upscale por tiles simulado en distintas resoluciones. No descarga modelos ni usa fotografías reales.

Las pruebas simuladas no certifican calidad del peso real, rendimiento MPS/CUDA ni ejecución en Fedora/Intel. Los modelos opcionales se deben instalar para usar inferencia real. Consulta `VALIDATION.md` para los resultados de esta entrega.

## Límites del MVP

Herramientas convencionales implementadas y conectadas; segmentación y RealESRGAN integrados como opciones locales; fase generativa pausada. Trabaja en RGB de 8 bits: no es revelador RAW ni editor de 16 bits. No hay detección anatómica/identidad, selección de varias personas, capas persistentes ni calibración de perspectiva por puntos. La reducción de imperfecciones no sustituye retoque manual de detalle. El autoguardado comprime/copía estados completos y puede tardar con fotografías grandes. No ejecutes varios procesos del editor escribiendo el mismo proyecto simultáneamente.

## Licencias para revisar antes de distribuir o usar comercialmente

Revisa las licencias de las versiones exactas y los términos de los **pesos y datos de entrenamiento**, no solo del código. Referencias de los proyectos: [Gradio](https://github.com/gradio-app/gradio/blob/main/LICENSE), [Pillow](https://github.com/python-pillow/Pillow/blob/main/LICENSE), [NumPy](https://github.com/numpy/numpy/blob/main/LICENSE.txt), [OpenCV](https://github.com/opencv/opencv/blob/4.x/LICENSE), [rembg](https://github.com/danielgatis/rembg/blob/main/LICENSE.txt), [U²-Net](https://github.com/xuebinqin/U-2-Net), [ONNX Runtime](https://github.com/microsoft/onnxruntime/blob/main/LICENSE), [PyTorch](https://github.com/pytorch/pytorch/blob/main/LICENSE), [Spandrel](https://github.com/chaiNNer-org/spandrel), [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE).

Mantén los avisos requeridos al redistribuir dependencias. Las condiciones de fotografías, consentimiento, marcas y recursos de fondo se gestionan por separado. Este repositorio no redistribuye pesos ni asigna una licencia comercial a sus modelos externos.
