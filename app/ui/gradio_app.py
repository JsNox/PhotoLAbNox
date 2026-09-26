from app.utils.paths import setup, CACHE, safe_name

setup()
import logging
import numpy as np
import gradio as gr
from PIL import Image
from pathlib import Path
from app.core.image_processor import EditorSession
from app.core import color_correction as color
from app.core.skin_retouch import retouch
from app.core.local_adjustments import dodge_burn, censor, watermark
from app.core.advanced_adjustments import hsl, white_balance, HUES
from app.core.inpainting import heal, clone
from app.core.transforms import transform, RATIOS
from app.core.segmentation import segment, combine
from app.core.background import replace
from app.core.body_reshape import estimate, calibrate, overlay, reshape
from app.core.liquify import LiquifyState
from app.core import project_store, presets
from app.core.export import save_copy, variants
from app.core.batch import process_batch
from app.core.upscale import upscale
from app.utils.image_utils import editor_mask, load_image

CSS = """
.gradio-container {max-width: 1500px !important; margin:auto;}
#masthead {padding:24px 28px; border:1px solid #34483f; border-radius:16px; background:linear-gradient(115deg,#17291f,#162127); margin-bottom:12px; color:#f3f7f4;}
#masthead h1 {font-size:32px; color:#f3f7f4; margin:4px 0; letter-spacing:-1px;}
#masthead p {color:#b5c6be; margin:4px 0;}
#masthead .tag {font:11px monospace; text-transform:uppercase; letter-spacing:2px; color:#94dcbd;}
#statebox textarea {font-family:monospace; font-size:12px;}
footer {display:none !important;}
"""


def uploaded(path):
    if not path:
        raise ValueError("Selecciona un archivo primero.")
    target = Path(path).resolve()
    if not target.is_relative_to(CACHE.resolve()) or not target.is_file():
        raise ValueError("Solo se aceptan archivos importados mediante la interfaz.")
    return target


def render(s, message="Listo."):
    mask = (
        Image.fromarray(combine(s.mask, s.keep, s.current.size))
        if s.mask is not None
        else None
    )
    region_view = (
        overlay(s.current, s.regions)
        if s.current is not None and s.regions
        else s.current
    )
    canvas = (
        {"background": s.current, "layers": [], "composite": s.current}
        if s.current is not None
        else None
    )
    history = (
        "\n".join(f"{i + 1}. {name}" for i, name in enumerate(s.operations))
        or "Sin operaciones."
    )
    project = (
        f" · Proyecto: {s.project} (autoguardado)"
        if s.project
        else " · Sin proyecto activo"
    )
    dimensions = (
        f"{s.current.width} × {s.current.height} px"
        if s.current is not None
        else "Sin fotografía"
    )
    return (
        s,
        s.original,
        s.current,
        mask,
        region_view,
        canvas,
        history,
        f"{message}\n{dimensions}{project}",
        canvas,
    )


def guarded(fn):
    def action(s, *args):
        s = s or EditorSession()
        try:
            message = fn(s, *args)
            return render(s, message or "Cambio aplicado.")
        except ValueError as exc:
            gr.Warning(str(exc))
            return render(s, str(exc))
        except Exception:
            logging.exception("Error en operación local")
            gr.Warning(
                "No se pudo completar la operación. Revisa los modelos, el archivo y la memoria disponible."
            )
            return render(s, "La operación falló; revisa la consola local.")

    return action


def commit(s, out, name, geometric=False):
    s.commit(out, name, preserve_mask=not geometric)


def color_action(s, *values):
    s.require()
    out = color.adjust(s.current, **dict(zip(color.DEFAULTS, values)))
    diff = np.abs(
        np.asarray(out, dtype=float) - np.asarray(s.current, dtype=float)
    ).mean()
    commit(s, out, "Ajustes fotográficos")
    return (
        "Ajustes aplicados."
        if diff > 0.2
        else "Efecto muy pequeño: aumenta los valores para obtener un cambio visible."
    )


def segment_action(s):
    s.require()
    s.mask = segment(s.current)
    s.keep = None
    s.regions = estimate(s.mask)
    s.points = []
    s.liquify = None
    return (
        "Persona segmentada. Revisa la máscara y calibra las regiones si es necesario."
    )


def remove_objects_action(s, value, radius, expansion, protect):
    s.require()
    mask = editor_mask(value, s.current.size)
    if protect and s.mask is None:
        raise ValueError(
            "Segmenta o pinta la máscara de persona en Fondo y silueta antes de activar su protección."
        )
    protected = combine(s.mask, s.keep, s.current.size) if protect else None
    out = heal(s.current, mask, radius, expansion, protected=protected)
    commit(s, out, "Eliminar objetos del fondo")
    return "Objeto eliminado. Revisa el resultado; puedes deshacer o pintar otra selección."


def paint_action(s, value, tool, amount, feather, radius, expansion, strength):
    s.require()
    mask = editor_mask(value, s.current.size)
    if tool == "Conservar objeto":
        if s.mask is None:
            raise ValueError("Segmenta la persona antes de conservar objetos.")
        s.keep = mask if s.keep is None else np.maximum(s.keep, mask)
        return "Objeto añadido a la máscara. Se conservará al cambiar el fondo."
    if tool == "Máscara manual de persona":
        s.mask = np.uint8(mask * 255)
        s.keep = None
        s.regions = estimate(s.mask)
        s.liquify = None
        return "Máscara manual lista. Permite trabajar sin modelo de segmentación."
    if tool == "Aclarar / oscurecer":
        out = dodge_burn(s.current, mask, amount, feather)
    elif tool == "Healing Telea":
        out = heal(s.current, mask, radius, expansion)
    else:
        out = censor(s.current, mask, tool, strength, feather)
    commit(s, out, tool)
    return (
        "Pincel aplicado."
        if tool != "Aclarar / oscurecer" or abs(amount) >= 3
        else "Cantidad muy pequeña: prueba ±10 o más."
    )


def reshape_action(s, bust, waist, hip, curves, intensity):
    s.require()
    if s.mask is None:
        raise ValueError("Segmenta o pinta una máscara de persona primero.")
    out, mask, amounts = reshape(
        s.current, s.mask, s.regions or None, bust, waist, hip, curves, intensity
    )
    commit(s, out, "Silueta automática")
    s.mask = mask
    if s.regions:
        s.regions = calibrate(mask, [(r.center, r.y) for r in s.regions])
    note = (
        " Advertencia: intensidad inferior a 25; el efecto puede ser imperceptible."
        if intensity < 25
        else ""
    )
    return "Efectivo: busto {:+.1f} %, cintura {:+.1f} %, cadera {:+.1f} %.{}".format(
        *(amounts * 100), note
    )


def click_action(
    s, tool, radius, strength, clone_radius, opacity, event: gr.SelectData
):
    s = s or EditorSession()
    try:
        s.require()
        point = tuple(map(int, event.index))
        if tool == "Balance de blancos":
            commit(s, white_balance(s.current, point), "Balance de blancos por muestra")
            return render(s, "Balance de blancos aplicado.")
        if tool == "Calibrar cinco regiones":
            if s.mask is None:
                raise ValueError("Segmenta la persona primero.")
            if len(s.points) == 5:
                s.points = []
            candidate = s.points + [point]
            if len(candidate) > 1 and candidate[-1][1] <= candidate[-2][1]:
                raise ValueError("Haz los clics de arriba hacia abajo.")
            if len(candidate) == 5:
                s.regions = calibrate(s.mask, candidate)
            s.points = candidate
            return render(
                s,
                f"Calibración: {len(candidate)}/5 puntos. Orden: hombros, busto, cintura, cadera, límite inferior.",
            )
        if tool.startswith("Liquify"):
            if s.mask is None:
                raise ValueError("Segmenta la persona primero.")
            if s.liquify is None:
                s.liquify = LiquifyState.create(s.current, s.mask)
            restore = tool == "Liquify Restore"
            if not restore and s.click_source is None:
                s.click_source = (tool, point)
                return render(s, "Origen de Liquify marcado; haz clic en el destino.")
            if not restore and s.click_source[0] != tool:
                s.click_source = (tool, point)
                return render(s, "Origen marcado para la nueva herramienta.")
            origin = point if restore else s.click_source[1]
            top = s.regions[0].y if s.regions else 0
            out, mask = s.liquify.apply(origin, point, radius, strength, restore, top)
            s.commit(out, tool, preserve_mask=True, preserve_liquify=True)
            s.mask = mask
            s.click_source = None
            return render(
                s, "Liquify aplicado. Tamaño y fuerza solo controlan Liquify."
            )
        if s.click_source is None or s.click_source[0] != tool:
            s.click_source = (tool, point)
            return render(s, "Origen del clonado marcado; haz clic en el destino.")
        commit(
            s,
            clone(s.current, s.click_source[1], point, clone_radius, opacity),
            "Clone Stamp",
        )
        s.click_source = None
        return render(s, "Clonado aplicado.")
    except ValueError as exc:
        gr.Warning(str(exc))
        return render(s, str(exc))
    except Exception:
        logging.exception("Error en herramienta de clic")
        return render(s, "No se pudo aplicar la herramienta de clic.")


def create_app():
    from app.utils.privacy import localize_gradio_template

    localize_gradio_template()
    theme = gr.themes.Soft(
        primary_hue="emerald",
        neutral_hue="slate",
        font=["system-ui", "sans-serif"],
        font_mono=["ui-monospace", "monospace"],
    )
    with gr.Blocks(
        title="Local Photo Lab",
        theme=theme,
        css=CSS,
        analytics_enabled=False,
        delete_cache=(3600, 3600),
    ) as demo:
        state = gr.State(EditorSession())
        gr.HTML(
            '<div id="masthead"><div class="tag">Estudio privado · Procesamiento en tu equipo</div><h1>Local Photo Lab</h1><p>Tu luz. Tu estilo. Tus fotografías, siempre locales.</p></div>'
        )
        with gr.Row():
            importer = gr.File(
                label="Importar fotografía · JPG / PNG / WEBP · hasta 40 MB y 24 MP",
                file_types=[".jpg", ".jpeg", ".png", ".webp"],
                type="filepath",
                scale=3,
            )
            with gr.Column(scale=2):
                status = gr.Textbox(
                    label="Estado del estudio",
                    value="Importa una fotografía para comenzar.",
                    interactive=False,
                    elem_id="statebox",
                    lines=3,
                )
                with gr.Row():
                    undo = gr.Button("Deshacer")
                    reset = gr.Button("Restablecer")
        with gr.Row():
            original = gr.Image(
                label="Original · sin modificar",
                type="pil",
                interactive=False,
                format="png",
                height=390,
            )
            current = gr.Image(
                label="Resultado acumulado",
                type="pil",
                interactive=False,
                format="png",
                height=390,
            )
        with gr.Tabs():
            with gr.Tab("Luz y color"):
                gr.Markdown(
                    "Los ajustes se aplican **al resultado acumulado** al pulsar el botón. Cargar un preset copia sus valores; «Aplicar preset completo» incluye retoque y silueta."
                )
                color_inputs = []
                labels = [
                    "Exposición (EV)",
                    "Brillo",
                    "Contraste",
                    "Saturación",
                    "Temperatura",
                    "Sombras",
                    "Altas luces",
                    "Nitidez",
                    "Reducción de ruido",
                ]
                with gr.Row():
                    with gr.Column():
                        for key, label in list(zip(color.DEFAULTS, labels))[:5]:
                            color_inputs.append(
                                gr.Slider(
                                    -3 if key == "exposure" else -100,
                                    3 if key == "exposure" else 100,
                                    value=0,
                                    step=0.05 if key == "exposure" else 1,
                                    label=label,
                                )
                            )
                    with gr.Column():
                        for key, label in list(zip(color.DEFAULTS, labels))[5:]:
                            color_inputs.append(
                                gr.Slider(
                                    0 if key in ("sharpness", "denoise") else -100,
                                    100,
                                    value=0,
                                    step=1,
                                    label=label,
                                )
                            )
                with gr.Row():
                    apply_color = gr.Button("Aplicar ajustes", variant="primary")
                    smart = gr.Button("Color inteligente")
                with gr.Accordion("HSL selectivo", open=False):
                    channel = gr.Dropdown(
                        ["Todos", *HUES], value="Todos", label="Rango de color"
                    )
                    hue = gr.Slider(-60, 60, 0, label="Tono")
                    sat = gr.Slider(-100, 100, 0, label="Saturación selectiva")
                    light = gr.Slider(-100, 100, 0, label="Luminosidad")
                    apply_hsl = gr.Button("Aplicar HSL")
            with gr.Tab("Piel y pinceles", render_children=True):
                with gr.Row():
                    skin_intensity = gr.Slider(
                        0, 100, 35, label="Intensidad de retoque"
                    )
                    texture = gr.Slider(0, 100, 70, label="Preservar textura")
                    skin_apply = gr.Button("Retocar piel")
                gr.Markdown(
                    "Pinta sobre el lienzo con el pincel. Solo los trazos de la capa se usan como máscara. Cada aplicación limpia los trazos y actualiza la fotografía."
                )
                painter = gr.ImageEditor(
                    label="Máscara pintada",
                    type="pil",
                    format="png",
                    image_mode="RGBA",
                    sources=(),
                    transforms=(),
                    brush=gr.Brush(colors=["#ffffff"], color_mode="fixed"),
                    eraser=gr.Eraser(),
                    height=500,
                )
                brush_tool = gr.Dropdown(
                    [
                        "Aclarar / oscurecer",
                        "Healing Telea",
                        "Desenfoque",
                        "Pixelado",
                        "Conservar objeto",
                        "Máscara manual de persona",
                    ],
                    value="Aclarar / oscurecer",
                    label="Herramienta de máscara",
                )
                with gr.Row():
                    amount = gr.Slider(
                        -100, 100, 20, label="Dodge & Burn: + aclara / − oscurece"
                    )
                    feather = gr.Slider(0, 40, 8, label="Suavizado del borde")
                    strength = gr.Slider(
                        1, 100, 20, label="Desenfoque / pixelado: intensidad"
                    )
                with gr.Row():
                    heal_radius = gr.Slider(1, 15, 3, step=1, label="Telea: radio")
                    expansion = gr.Slider(
                        0, 30, 3, step=1, label="Telea: expansión de máscara"
                    )
                apply_paint = gr.Button("Aplicar máscara", variant="primary")
            with gr.Tab("Eliminar objetos", render_children=True):
                gr.Markdown(
                    "Pinta sobre el objeto que quieres quitar del fondo y pulsa **Eliminar objeto seleccionado**. "
                    "Cubre todo su contorno y, si hace falta, su sombra. El borrador permite corregir la selección. "
                    "La herramienta rellena la zona con información de los píxeles vecinos, sin descargar modelos."
                )
                object_painter = gr.ImageEditor(
                    label="Pinta el objeto que quieres eliminar",
                    type="pil",
                    format="png",
                    image_mode="RGBA",
                    sources=(),
                    transforms=(),
                    brush=gr.Brush(colors=["#ffffff"], color_mode="fixed"),
                    eraser=gr.Eraser(),
                    height=500,
                )
                with gr.Row():
                    object_radius = gr.Slider(
                        1, 15, 3, step=1, label="Radio de reconstrucción"
                    )
                    object_expansion = gr.Slider(
                        0, 30, 3, step=1, label="Expandir selección (px)"
                    )
                protect_person = gr.Checkbox(
                    False,
                    label="Proteger persona y objetos conservados (requiere máscara de persona)",
                )
                with gr.Row():
                    remove_object = gr.Button(
                        "Eliminar objeto seleccionado", variant="primary"
                    )
                    clear_object = gr.Button("Limpiar selección")
                gr.Markdown(
                    "Funciona mejor con objetos pequeños y fondos sencillos. En fondos con patrones o al quitar "
                    "objetos grandes puede dejar artefactos; corrige con Clone Stamp en Fondo y silueta. "
                    "Cada aplicación se añade al historial y actualiza el proyecto si tiene autoguardado activo."
                )
            with gr.Tab("Fondo y silueta"):
                with gr.Row():
                    segment_button = gr.Button(
                        "Segmentar persona · U²-Net", variant="primary"
                    )
                    auto_regions = gr.Button("Estimar regiones otra vez")
                gr.Markdown(
                    "Sin modelo, puedes pintar «Máscara manual de persona» en Piel y pinceles. Para conservar accesorios, pinta «Conservar objeto» después de segmentar."
                )
                with gr.Row():
                    mask_view = gr.Image(
                        label="Máscara combinada · persona + objetos",
                        interactive=False,
                        type="pil",
                        height=300,
                        format="png",
                    )
                    regions_view = gr.Image(
                        label="Regiones / lienzo de clics",
                        interactive=False,
                        type="pil",
                        height=450,
                        format="png",
                    )
                with gr.Accordion("Cambiar fondo", open=True):
                    bg_mode = gr.Dropdown(
                        ["Transparente", "Color sólido", "Desenfocado", "Fotografía"],
                        value="Transparente",
                        label="Tipo de fondo",
                    )
                    with gr.Row():
                        bg_color = gr.ColorPicker("#eeeeee", label="Color de fondo")
                        bg_photo = gr.File(
                            label="Fotografía de fondo",
                            file_types=[".jpg", ".jpeg", ".png", ".webp"],
                        )
                        bg_blur = gr.Slider(1, 80, 20, label="Desenfoque de fondo")
                        bg_feather = gr.Slider(0, 15, 2, label="Borde suave")
                    bg_apply = gr.Button("Aplicar fondo")
                with gr.Accordion("Silueta automática", open=True):
                    gr.Markdown(
                        "Estimación inicial: revisa las líneas. Calibra **cinco clics de arriba hacia abajo**: hombros, busto, cintura, cadera, límite inferior. Cada clic debe cruzar el torso. El campo comienza bajo los hombros; el fondo se reconstruye con Telea."
                    )
                    with gr.Row():
                        bust = gr.Slider(-20, 20, 0, label="Busto")
                        waist = gr.Slider(-20, 20, 0, label="Cintura")
                        hip = gr.Slider(-20, 20, 0, label="Cadera")
                    with gr.Row():
                        curves = gr.Slider(0, 20, 0, label="Curvas")
                        intensity = gr.Slider(0, 100, 60, label="Intensidad automática")
                    reshape_button = gr.Button("Aplicar silueta", variant="primary")
                gr.Markdown(
                    "### Herramientas de clic\nHaz clic en **Regiones / lienzo de clics** o en **Resultado acumulado**. Clone y Push requieren origen y destino; Restore y balance de blancos, un clic. Restore recupera el estado al comenzar la sesión de Liquify. Deshacer o aplicar otra herramienta reinicia esa referencia."
                )
                click_tool = gr.Dropdown(
                    [
                        "Calibrar cinco regiones",
                        "Balance de blancos",
                        "Clone Stamp",
                        "Liquify Push / Drag",
                        "Liquify Restore",
                    ],
                    value="Calibrar cinco regiones",
                    label="Acción al hacer clic",
                )
                clear_clicks = gr.Button("Reiniciar clics / calibración")
                with gr.Row():
                    liq_radius = gr.Slider(
                        10, 300, 90, label="Liquify: tamaño de pincel (px)"
                    )
                    liq_strength = gr.Slider(1, 100, 20, label="Liquify: fuerza")
                    clone_radius = gr.Slider(2, 200, 30, label="Clone Stamp: radio")
                    clone_opacity = gr.Slider(1, 100, 80, label="Clone Stamp: opacidad")
                gr.Markdown(
                    "**Tamaño y Fuerza solo afectan Liquify, no la silueta automática.** Recomendado: pincel 70–120 y fuerza 10–25. La placa de fondo convencional puede dejar artefactos en fondos complejos."
                )
            with gr.Tab("Encuadre"):
                with gr.Row():
                    rotation = gr.Slider(
                        -180, 180, 0, step=0.1, label="Rotación / enderezado"
                    )
                    flip = gr.Checkbox(False, label="Volteo horizontal")
                    ratio = gr.Dropdown(
                        list(RATIOS), value="Original", label="Recorte centrado"
                    )
                with gr.Row():
                    perspective_h = gr.Slider(
                        -20, 20, 0, label="Perspectiva horizontal"
                    )
                    perspective_v = gr.Slider(-20, 20, 0, label="Perspectiva vertical")
                apply_transform = gr.Button("Aplicar transformación", variant="primary")
            with gr.Tab("Proyectos y presets"):
                gr.Markdown(
                    "Los proyectos guardan original, resultado e historial. Al crear o cargar uno, **cada edición se autoguarda**. Importar otra foto desactiva el proyecto anterior. Las máscaras de trabajo y las referencias de clic son temporales."
                )
                with gr.Row():
                    project_name = gr.Textbox(label="Nombre del proyecto")
                    project_save = gr.Button("Guardar / activar autoguardado")
                    project_off = gr.Button("Desactivar autoguardado")
                with gr.Row():
                    project_list = gr.Dropdown(
                        project_store.list_projects(), label="Proyectos locales"
                    )
                    project_load = gr.Button("Cargar proyecto")
                    project_refresh = gr.Button("Actualizar lista")
                with gr.Row():
                    preset_list = gr.Dropdown(
                        presets.list_presets(), value="Natural", label="Preset"
                    )
                    preset_load = gr.Button("Copiar preset a controles")
                    preset_apply = gr.Button("Aplicar preset completo")
                with gr.Row():
                    preset_name = gr.Textbox(label="Nombre de preset personalizado")
                    preset_save = gr.Button("Guardar controles como preset")
                history = gr.Textbox(
                    label="Registro acumulado de operaciones · 25 pasos disponibles para deshacer",
                    lines=8,
                    interactive=False,
                )
            with gr.Tab("Exportar y lotes"):
                with gr.Row():
                    export_format = gr.Dropdown(
                        ["PNG", "JPEG", "WEBP"], value="PNG", label="Formato"
                    )
                    strip = gr.Checkbox(True, label="Eliminar EXIF y GPS en producción")
                    export_button = gr.Button("Guardar copia", variant="primary")
                gr.Markdown(
                    "PNG y WEBP conservan transparencia. JPEG compone sobre blanco. La copia se guarda con un nombre nuevo en outputs/."
                )
                with gr.Row():
                    mark_text = gr.Textbox("Local Photo Lab", label="Texto de marca")
                    mark_size = gr.Slider(8, 160, 32, step=1, label="Tamaño")
                    mark_opacity = gr.Slider(0, 100, 40, label="Opacidad")
                    mark_position = gr.Dropdown(
                        [
                            "Inferior derecha",
                            "Inferior izquierda",
                            "Superior derecha",
                            "Superior izquierda",
                            "Centro",
                        ],
                        value="Inferior derecha",
                        label="Posición",
                    )
                with gr.Row():
                    mark_apply = gr.Button("Aplicar marca al resultado")
                    variants_button = gr.Button(
                        "Guardar 3 variantes: editada / marca / teaser"
                    )
                downloads = gr.File(
                    label="Copias guardadas", file_count="multiple", interactive=False
                )
                gr.Markdown(
                    "### Lote local\nUsa el preset seleccionado en Proyectos y presets. Los archivos se procesan uno por uno; siempre se exportan JPEG sin EXIF."
                )
                batch_files = gr.File(
                    label="Fotografías · máximo 100",
                    file_count="multiple",
                    file_types=[".jpg", ".jpeg", ".png", ".webp"],
                )
                batch_mark = gr.Checkbox(False, label="Añadir la marca configurada")
                batch_button = gr.Button("Procesar lote y guardar ZIP")
                batch_download = gr.File(label="ZIP del lote", interactive=False)
            with gr.Tab("Upscale 2×"):
                gr.Markdown(
                    "### RealESRGAN_x2plus · Fase 6\nModelo oficial local mediante Spandrel. CUDA usa FP16; MPS y CPU usan FP32. Si MPS falla, se reintenta en CPU. Salida máxima: **64 MP** (entrada para 2×: hasta 16 MP).\n\nColoca el peso verificado en `models/upscale/RealESRGAN_x2plus.pth`. El modelo se carga únicamente al ejecutar esta operación. Puede requerir varios GB de RAM; comienza con tiles de 128."
                )
                tile = gr.Dropdown(
                    [128, 256, 384, 512], value=256, label="Tile con solapamiento"
                )
                upscale_button = gr.Button("Aplicar upscale 2×", variant="primary")
            with gr.Tab("Generativa · pausada"):
                gr.Markdown(
                    "### Inpainting por prompt: fase pausada\nNo se ha instalado Stable Diffusion ni ningún modelo generativo. `requirements-ai.txt` permanece vacío. Para eliminar objetos utiliza Healing Telea o Clone Stamp; ambas herramientas trabajan localmente sobre una selección. Esta fase se retomará cuando el hardware permita una experiencia adecuada."
                )
        gr.Markdown(
            "**Privado por diseño.** Solo 127.0.0.1 · sin nube ni analytics · usa exclusivamente fotografías legales de personas adultas y con consentimiento. Los archivos temporales de la interfaz permanecen en la caché local."
        )
        outputs = [
            state,
            original,
            current,
            mask_view,
            regions_view,
            painter,
            history,
            status,
            object_painter,
        ]

        def bind(button, fn, inputs=()):
            return button.click(
                guarded(fn),
                [state, *inputs],
                outputs,
                api_name=False,
                concurrency_id="edits",
                concurrency_limit=1,
            )

        importer.upload(
            guarded(lambda s, p: (s.open(uploaded(p)), "Fotografía importada.")[1]),
            [state, importer],
            outputs,
            api_name=False,
            concurrency_id="edits",
        )
        bind(undo, lambda s: s.undo())
        bind(reset, lambda s: s.reset())
        bind(apply_color, color_action, color_inputs)
        bind(
            smart,
            lambda s: (
                s.require(),
                commit(s, color.intelligent(s.current), "Color inteligente"),
            )[-1],
        )
        bind(
            apply_hsl,
            lambda s, *v: (s.require(), commit(s, hsl(s.current, *v), "HSL selectivo"))[
                -1
            ],
            [channel, hue, sat, light],
        )
        bind(
            skin_apply,
            lambda s, i, t: (
                s.require(),
                commit(s, retouch(s.current, i, t), "Retoque de piel"),
            )[-1],
            [skin_intensity, texture],
        )
        bind(
            apply_paint,
            paint_action,
            [painter, brush_tool, amount, feather, heal_radius, expansion, strength],
        )
        bind(
            remove_object,
            remove_objects_action,
            [object_painter, object_radius, object_expansion, protect_person],
        )

        def clear_object_selection(s):
            if s is None or s.current is None:
                return None
            return {"background": s.current, "layers": [], "composite": s.current}

        clear_object.click(
            clear_object_selection,
            [state],
            object_painter,
            api_name=False,
            concurrency_id="edits",
        )
        bind(segment_button, segment_action)

        def regions_again(s):
            s.require()
            if s.mask is None:
                raise ValueError("Primero crea una máscara de persona.")
            s.regions = estimate(s.mask)
            s.points = []
            return "Regiones estimadas; revisa y calibra manualmente."

        bind(auto_regions, regions_again)

        def background_action(s, mode, c, p, b, f):
            s.require()
            out = replace(
                s.current,
                s.mask,
                mode,
                c,
                load_image(uploaded(p)) if p and mode == "Fotografía" else None,
                b,
                f,
                s.keep,
            )
            commit(s, out, "Fondo: " + mode, geometric=True)

        bind(
            bg_apply,
            background_action,
            [bg_mode, bg_color, bg_photo, bg_blur, bg_feather],
        )
        bind(reshape_button, reshape_action, [bust, waist, hip, curves, intensity])

        def clear(s):
            s.points = []
            s.click_source = None
            return "Clics reiniciados; puedes comenzar de nuevo."

        bind(clear_clicks, clear)
        for view in (regions_view, current):
            view.select(
                click_action,
                [
                    state,
                    click_tool,
                    liq_radius,
                    liq_strength,
                    clone_radius,
                    clone_opacity,
                ],
                outputs,
                api_name=False,
                concurrency_id="edits",
                concurrency_limit=1,
            )
        bind(
            apply_transform,
            lambda s, *v: (
                s.require(),
                commit(s, transform(s.current, *v), "Transformación", True),
            )[-1],
            [rotation, flip, ratio, perspective_h, perspective_v],
        )

        def save_project(s, name):
            project_store.save_project(s, name)
            s.project = safe_name(name)
            return "Proyecto guardado. Autoguardado activado."

        saved_project = bind(project_save, save_project, [project_name])

        def off(s):
            s.project = None
            return "Autoguardado desactivado."

        bind(project_off, off)

        def load_project(s, name):
            loaded = project_store.load_project(name)
            s.__dict__.update(loaded.__dict__)
            return "Proyecto cargado. Autoguardado activado."

        bind(project_load, load_project, [project_list])
        project_refresh.click(
            lambda: gr.update(choices=project_store.list_projects()),
            outputs=project_list,
            api_name=False,
        )
        saved_project.then(
            lambda: gr.update(choices=project_store.list_projects()),
            outputs=project_list,
            api_name=False,
        )
        preset_controls = [
            *color_inputs,
            skin_intensity,
            texture,
            bust,
            waist,
            hip,
            curves,
            intensity,
        ]

        def load_preset_values(name):
            try:
                p = presets.load_preset(name)
                return [
                    *[p["color"][k] for k in color.DEFAULTS],
                    *[
                        p[k]
                        for k in (
                            "retouch",
                            "texture",
                            "bust",
                            "waist",
                            "hip",
                            "curves",
                            "intensity",
                        )
                    ],
                ]
            except ValueError as exc:
                raise gr.Error(str(exc)) from exc

        preset_load.click(
            load_preset_values, [preset_list], preset_controls, api_name=False
        )

        def save_preset_values(name, *values):
            p = {"color": dict(zip(color.DEFAULTS, values[:9]))} | dict(
                zip(
                    (
                        "retouch",
                        "texture",
                        "bust",
                        "waist",
                        "hip",
                        "curves",
                        "intensity",
                    ),
                    values[9:],
                )
            )
            try:
                presets.save_preset(name, p)
            except ValueError as exc:
                raise gr.Error(str(exc)) from exc
            return gr.update(choices=presets.list_presets(), value=safe_name(name))

        preset_save.click(
            save_preset_values,
            [preset_name, *preset_controls],
            preset_list,
            api_name=False,
        )
        bind(
            preset_apply,
            lambda s, n: (
                s.require(),
                commit(
                    s,
                    presets.apply_preset(s.current, presets.load_preset(n)),
                    "Preset: " + n,
                    True,
                ),
            )[-1],
            [preset_list],
        )
        bind(
            mark_apply,
            lambda s, *v: (
                s.require(),
                commit(s, watermark(s.current, *v), "Marca de agua"),
            )[-1],
            [mark_text, mark_size, mark_opacity, mark_position],
        )

        def export_action(s, fmt, remove):
            try:
                s.require()
                return [str(save_copy(s.current, fmt, remove))]
            except ValueError as exc:
                raise gr.Error(str(exc)) from exc

        export_button.click(
            export_action,
            [state, export_format, strip],
            downloads,
            api_name=False,
            concurrency_id="edits",
        )

        def export_variants(s, *args):
            try:
                s.require()
                return [str(p) for p in variants(s.current, *args)]
            except ValueError as exc:
                raise gr.Error(str(exc)) from exc

        variants_button.click(
            export_variants,
            [
                state,
                mark_text,
                mark_size,
                mark_opacity,
                mark_position,
                export_format,
                strip,
            ],
            downloads,
            api_name=False,
            concurrency_id="edits",
        )

        def batch_action(files, name, add, text, size, op, pos):
            try:
                if not files or len(files) > 100:
                    raise ValueError("Selecciona entre 1 y 100 fotografías.")
                mark = (
                    dict(text=text, size=size, opacity=op, position=pos)
                    if add
                    else None
                )
                return str(
                    process_batch(
                        [uploaded(p) for p in files], presets.load_preset(name), mark
                    )
                )
            except ValueError as exc:
                raise gr.Error(str(exc)) from exc

        batch_button.click(
            batch_action,
            [
                batch_files,
                preset_list,
                batch_mark,
                mark_text,
                mark_size,
                mark_opacity,
                mark_position,
            ],
            batch_download,
            api_name=False,
            concurrency_id="edits",
        )

        def upscale_action(s, t):
            s.require()
            out, device = upscale(s.current, t)
            commit(s, out, "Upscale RealESRGAN 2×", True)
            return "Upscale 2× completado en " + device

        bind(upscale_button, upscale_action, [tile])
    return demo.queue(default_concurrency_limit=1, max_size=16)
