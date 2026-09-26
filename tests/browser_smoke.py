"""Optional real-browser smoke. Run manually; requires Playwright and local Chrome.
Uses only a generated color chart, never a real photograph.
"""

import json
import os
import re
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright, expect


def main():
    errors = []
    external = []
    with tempfile.TemporaryDirectory(prefix="lpl-smoke-") as temp:
        source = Path(temp) / "synthetic.png"
        y, x = np.mgrid[:240, :180]
        a = np.stack([70 + x // 2, 50 + y // 2, 60 + (x + y) // 4], axis=-1).astype(
            np.uint8
        )
        a[40:220, 55:125] = (195, 130, 100)
        Image.fromarray(a).save(source)
        with sync_playwright() as p:
            browser = p.chromium.launch(
                executable_path=os.environ.get(
                    "LPL_CHROME",
                    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                ),
                headless=True,
            )
            page = browser.new_page(viewport={"width": 1440, "height": 1100})
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on(
                "request",
                lambda r: (
                    external.append(r.url)
                    if r.url.startswith("http")
                    and not r.url.startswith("http://127.0.0.1:7860/")
                    else None
                ),
            )
            try:
                page.goto(
                    "http://127.0.0.1:7860/?__theme=dark", wait_until="domcontentloaded"
                )
                expect(
                    page.get_by_role("heading", name="Local Photo Lab")
                ).to_be_visible(timeout=30000)
                page.locator("input[type=file]").first.set_input_files(str(source))
                status = page.get_by_label("Estado del estudio")
                expect(status).to_have_value(
                    re.compile("Fotografía importada"), timeout=30000
                )
                page.locator("input[type=number]").first.fill("0.5")
                page.get_by_role("button", name="Aplicar ajustes", exact=True).click()
                expect(status).to_have_value(
                    re.compile("Ajustes aplicados"), timeout=30000
                )
                page.get_by_role("button", name="Deshacer", exact=True).click()
                expect(status).to_have_value(
                    re.compile("Cambio aplicado"), timeout=30000
                )
                page.get_by_role("tab", name="Piel y pinceles", exact=True).click()
                page.get_by_role("button", name="Retocar piel", exact=True).click()
                expect(status).to_have_value(
                    re.compile("Cambio aplicado"), timeout=30000
                )
                page.get_by_role("tab", name="Exportar y lotes", exact=True).click()
                page.get_by_role("button", name="Guardar copia", exact=True).click()
                expect(page.locator('a[href*="/file="]').last).to_be_visible(
                    timeout=30000
                )
                page.get_by_role("tab", name="Luz y color", exact=True).click()
                page.screenshot(path="/tmp/local-photo-lab-smoke.png", full_page=True)
            except Exception:
                page.screenshot(path="/tmp/lpl-smoke-failure.png", full_page=True)
                print("ERRORS", errors, "EXTERNAL", external, flush=True)
                print(page.locator("body").inner_text()[:3000], flush=True)
                raise
            browser.close()
    print(
        json.dumps(
            {
                "browser_errors": errors,
                "external_requests": external,
                "flow": "importar → exposición → deshacer → piel → guardar copia",
            },
            ensure_ascii=False,
        )
    )
    assert not errors, errors
    assert not external, external


if __name__ == "__main__":
    main()
