from dataclasses import dataclass, field
from io import BytesIO
from PIL import Image
from app.utils.image_utils import png_bytes, load_image

HISTORY_LIMIT = 25


@dataclass
class Snapshot:
    name: str
    data: bytes


@dataclass
class EditorSession:
    original: object = None
    current: object = None
    history: list = field(default_factory=list)
    operations: list = field(default_factory=list)
    original_format: str = "PNG"
    project: str | None = None
    mask: object = None
    keep: object = None
    regions: list = field(default_factory=list)
    points: list = field(default_factory=list)
    click_source: object = None
    liquify: object = None
    last_status: str = ""

    def require(self):
        if self.current is None:
            raise ValueError("Importa una fotografía primero.")

    def open(self, source):
        im = load_image(source)
        self.original, self.current = im.copy(), im.copy()
        self.original_format = im.info.get("source_format", "PNG")
        self.history, self.operations = [], []
        self.project = None
        self.invalidate()
        return self

    def invalidate(self):
        self.mask = self.keep = self.click_source = self.liquify = None
        self.regions, self.points = [], []

    def commit(self, image, name, preserve_mask=False, preserve_liquify=False):
        self.require()
        if image.width * image.height > 64_000_000:
            raise ValueError("La salida supera 64 megapíxeles.")
        self.history.append(Snapshot(name, png_bytes(self.current)))
        self.history = self.history[-HISTORY_LIMIT:]
        self.operations.append(name)
        self.current = image
        if not preserve_mask:
            self.invalidate()
        elif not preserve_liquify:
            self.liquify = None
            self.click_source = None
        self.autosave()
        return self

    def undo(self):
        self.require()
        if not self.history:
            raise ValueError("No hay operaciones para deshacer.")
        snap = self.history.pop()
        self.current = Image.open(BytesIO(snap.data)).copy()
        self.current.info.update(self.original.info)
        self.operations.pop()
        self.invalidate()
        self.autosave()

    def reset(self):
        self.require()
        self.commit(self.original.copy(), "Restablecer original")

    def autosave(self):
        if self.project:
            from app.core.project_store import save_project

            try:
                save_project(self, self.project)
            except Exception as exc:
                raise ValueError(
                    "El cambio está en memoria, pero falló el autoguardado. Reintenta guardar el proyecto."
                ) from exc
