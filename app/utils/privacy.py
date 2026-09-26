"""Defense in depth: application process cannot resolve/connect external hosts."""

import ipaddress
import socket


def local_host(host):
    if isinstance(host, bytes):
        host = host.decode("ascii")
    if host in ("localhost", "127.0.0.1", "::1"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def disable_outbound_network():
    if getattr(socket, "_lpl_local_only", False):
        return
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_resolve = socket.getaddrinfo

    def connect(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6) and not local_host(
            address[0]
        ):
            raise OSError("Local Photo Lab bloquea conexiones externas.")
        return original_connect(self, address)

    def connect_ex(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6) and not local_host(
            address[0]
        ):
            raise OSError("Local Photo Lab bloquea conexiones externas.")
        return original_connect_ex(self, address)

    def resolve(host, *args, **kwargs):
        if host is not None and not local_host(host):
            raise OSError("Local Photo Lab bloquea DNS externo.")
        return original_resolve(host, *args, **kwargs)

    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.getaddrinfo = resolve
    socket._lpl_local_only = True


def localize_gradio_template():
    """Remove Gradio's bundled CDN script/preconnects without changing site-packages."""
    import re
    from gradio.routes import templates
    from jinja2 import ChoiceLoader, DictLoader

    env = templates.env
    if getattr(env, "_lpl_localized", False):
        return
    name = "frontend/index.html"
    source, _, _ = env.loader.get_source(env, name)
    source = re.sub(
        r"<script\b[^>]*\bsrc=[\"\']https?://[^>]*>.*?</script>",
        "",
        source,
        flags=re.S | re.I,
    )
    source = re.sub(r"<link\b(?=[^>]*https?://)[^>]*>", "", source, flags=re.S | re.I)
    env.loader = ChoiceLoader([DictLoader({name: source}), env.loader])
    env.cache.clear()
    env._lpl_localized = True


class LocalSecurityHeaders:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        async def local_send(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.extend(
                    [
                        (
                            b"content-security-policy",
                            b"default-src 'self' blob: data:; script-src 'self' 'unsafe-inline' 'unsafe-eval' blob:; style-src 'self' 'unsafe-inline'; connect-src 'self' blob: data:; font-src 'self' data:; img-src 'self' blob: data:; worker-src 'self' blob:; frame-ancestors 'self'; base-uri 'self'",
                        ),
                        (b"referrer-policy", b"no-referrer"),
                        (b"x-content-type-options", b"nosniff"),
                    ]
                )
                message = dict(message, headers=headers)
            await send(message)

        await self.app(scope, receive, local_send)
