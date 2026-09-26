"""Run with: python -m app.main"""

from app.utils.paths import setup, OUTPUTS, PROJECTS, MODELS, ROOT

setup()
from app.utils.privacy import disable_outbound_network, LocalSecurityHeaders
from starlette.middleware import Middleware


def main():
    # Set before importing Gradio or optional ML libraries.
    disable_outbound_network()
    from app.ui.gradio_app import create_app

    demo = create_app()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=False,
        allowed_paths=[str(OUTPUTS)],
        blocked_paths=[
            str(PROJECTS),
            str(MODELS),
            str(ROOT / ".git"),
            str(ROOT / ".venv"),
        ],
        max_file_size="40mb",
        show_api=False,
        show_error=False,
        enable_monitoring=False,
        strict_cors=True,
        ssr_mode=False,
        app_kwargs={"middleware": [Middleware(LocalSecurityHeaders)]},
    )


if __name__ == "__main__":
    main()
