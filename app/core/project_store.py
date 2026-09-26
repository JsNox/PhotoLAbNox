import base64
import json
import os
import tempfile
import zipfile
from io import BytesIO
from PIL import Image
from app.utils.paths import PROJECTS, confined, safe_name
from app.utils.image_utils import png_bytes
from app.core.image_processor import EditorSession, Snapshot, HISTORY_LIMIT

VERSION = 1
MAX_ARCHIVE = 3 * 1024**3


def save_project(session, name, root=PROJECTS):
    session.require()
    root.mkdir(exist_ok=True, parents=True)
    path = confined(root, name, ".lpl")
    manifest = dict(
        version=VERSION,
        format=session.original_format,
        operations=session.operations,
        history=[s.name for s in session.history],
        exif=base64.b64encode(session.original.info.get("exif", b"")).decode("ascii"),
    )
    fd, tmp = tempfile.mkstemp(prefix=".lpl-", dir=root)
    try:
        with os.fdopen(fd, "w+b") as stream:
            with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as z:
                z.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False))
                z.writestr("original.png", png_bytes(session.original))
                z.writestr("current.png", png_bytes(session.current))
                for i, snap in enumerate(session.history):
                    z.writestr(f"history/{i}.png", snap.data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def load_project(name, root=PROJECTS):
    path = confined(root, name, ".lpl")
    try:
        if path.stat().st_size > MAX_ARCHIVE:
            raise ValueError("Proyecto demasiado grande.")
        with zipfile.ZipFile(path) as z:
            infos = z.infolist()
            if (
                len(infos) > HISTORY_LIMIT + 3
                or sum(i.file_size for i in infos) > MAX_ARCHIVE
                or any(i.file_size > 300_000_000 for i in infos)
            ):
                raise ValueError("El proyecto supera los límites permitidos.")
            if z.getinfo("manifest.json").file_size > 2_000_000:
                raise ValueError("Manifiesto demasiado grande.")
            m = json.loads(z.read("manifest.json"))
            if m["version"] != VERSION:
                raise ValueError("Versión de proyecto incompatible.")
            names, operations = m["history"], m["operations"]
            if (
                not isinstance(names, list)
                or len(names) > HISTORY_LIMIT
                or not isinstance(operations, list)
                or len(operations) < len(names)
                or any(not isinstance(x, str) for x in names + operations)
            ):
                raise ValueError("Historial inválido.")

            def read_image(member):
                data = z.read(member)
                with Image.open(BytesIO(data)) as im:
                    if im.width * im.height > 64_000_000 or im.format != "PNG":
                        raise ValueError(
                            "Resolución o formato inválido en el proyecto."
                        )
                    im.load()
                    return im.copy(), data

            original, _ = read_image("original.png")
            current, _ = read_image("current.png")
            original.info["exif"] = base64.b64decode(m.get("exif", ""), validate=True)
            history = [
                Snapshot(n, read_image(f"history/{i}.png")[1])
                for i, n in enumerate(names)
            ]
            current.info.update(original.info)
            session = EditorSession(
                original=original,
                current=current,
                history=history,
                operations=operations,
                original_format=m["format"],
                project=safe_name(name),
            )
            return session
    except (
        OSError,
        KeyError,
        TypeError,
        zipfile.BadZipFile,
        json.JSONDecodeError,
        ValueError,
        Image.DecompressionBombError,
    ) as exc:
        raise ValueError(f"Proyecto dañado o incompatible: {exc}") from exc


def list_projects(root=PROJECTS):
    return sorted(p.stem for p in root.glob("*.lpl") if not p.is_symlink())
