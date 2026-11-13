"""ctypes binding to libarctag (Arc Tagger v0.3, native C backend, int8), or the shared model host's copy.

One handle per stream: a handle is not thread-safe. Each call costs about 1-3 ms per word on the native backend.

MODELS=host: `Tagger(...)` returns a RemoteTagger instead, same methods, backed by one dedicated handle in the model
server (WS {MODEL_HOST_URL}/v1/tag/stream). Ops that return nothing (push, endpoint, reset, set) are pipelined; poll /
tag_text / info wait for the reply, and a failed pipelined op raises there. If the host can't be reached the
constructor logs loudly and returns a native Tagger (fail open), and stops trying the host for this process.
"""
from __future__ import annotations

import ctypes
import json
import logging
import os
import socket
import threading
from pathlib import Path
from urllib.parse import urlencode

from . import config

log = logging.getLogger("headstart.tagger")

ROOT = Path(__file__).resolve().parent.parent
LIB_PATH = Path(os.environ.get("ARCTAG_LIB", ROOT / "vendor/arctag/libarctag.dylib"))
MODEL_DIR = Path(os.environ.get("ARCTAG_MODEL_DIR", ROOT / "models/arctag"))


class _Opts(ctypes.Structure):
    _fields_ = [("size", ctypes.c_size_t), ("precision", ctypes.c_int), ("threads", ctypes.c_int),
                ("model_file", ctypes.c_char_p), ("backend", ctypes.c_int)]


_lib = None
_lib_lock = threading.Lock()


def lib():
    global _lib
    with _lib_lock:
        if _lib is None:
            L = ctypes.CDLL(str(LIB_PATH))
            L.arctag_options_init.argtypes = [ctypes.POINTER(_Opts)]
            L.arctag_create_ex.argtypes = [ctypes.c_char_p, ctypes.POINTER(_Opts), ctypes.c_char_p, ctypes.c_size_t]
            L.arctag_create_ex.restype = ctypes.c_void_p
            for fn in ("arctag_tag_text", "arctag_tag_words", "arctag_poll", "arctag_info"):
                getattr(L, fn).restype = ctypes.c_void_p
            L.arctag_tag_text.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            L.arctag_tag_words.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            L.arctag_poll.argtypes = [ctypes.c_void_p]
            L.arctag_info.argtypes = [ctypes.c_void_p]
            L.arctag_push_words.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            L.arctag_push_words.restype = ctypes.c_int
            L.arctag_set_option.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
            L.arctag_set_option.restype = ctypes.c_int
            L.arctag_apply_config.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            L.arctag_apply_config.restype = ctypes.c_int
            L.arctag_reset.argtypes = [ctypes.c_void_p]
            L.arctag_destroy.argtypes = [ctypes.c_void_p]
            L.arctag_free_str.argtypes = [ctypes.c_void_p]
            L.arctag_last_error.argtypes = [ctypes.c_void_p]
            L.arctag_last_error.restype = ctypes.c_char_p
            _lib = L
        return _lib


class Tagger:
    """Streaming Arc Tagger handle.

    push(words, from_=None, endpoint=False) appends words, or replaces words[from_:] (a revision), then poll()
    returns what changed: spans ending after a0, new events, closed utterances and the running read-out (`current`).
    """

    def __new__(cls, *args, **kwargs):
        if cls is Tagger and config.MODELS == "host" and not _HOST["down"]:
            r = _connect_remote(*args, **kwargs)
            if r is not None:
                return r   # not a Tagger subclass, so Tagger.__init__ doesn't run on it
        return super().__new__(cls)

    def __init__(self, domain: str = "assistant", asr: str = "unknown", model_dir: Path = MODEL_DIR,
                 emit: str = "entity,number,pii,events,intent,act,safety,state", options: dict | None = None):
        L = lib()
        o = _Opts()
        L.arctag_options_init(ctypes.byref(o))
        o.precision = 1  # int8
        o.backend = 1    # native
        err = ctypes.create_string_buffer(512)
        h = L.arctag_create_ex(str(model_dir).encode(), ctypes.byref(o), err, 512)
        if not h:
            raise RuntimeError(f"arctag_create_ex failed: {err.value.decode(errors='replace')}")
        self.h = h
        self.L = L
        self.set("domain", domain)
        if asr != "unknown":
            self.set("asr", asr)
        if emit:
            self.set("emit", emit)
        for k, v in (options or {}).items():
            self.set(k, v)

    def set(self, key: str, value: str):
        if self.L.arctag_set_option(self.h, key.encode(), str(value).encode()) != 0:
            raise ValueError(f"arctag_set_option({key}={value}) failed: {self._err()}")

    def _err(self):
        e = self.L.arctag_last_error(self.h)
        return e.decode(errors="replace") if e else ""

    def _take(self, p) -> dict:
        if not p:
            raise RuntimeError(f"arctag call failed: {self._err()}")
        try:
            return json.loads(ctypes.string_at(p).decode("utf-8"))
        finally:
            self.L.arctag_free_str(p)

    def tag_text(self, text: str) -> dict:
        return self._take(self.L.arctag_tag_text(self.h, text.encode()))

    def push(self, words: list, from_: int | None = None, endpoint: bool = False):
        if from_ is None and not endpoint:
            payload = words
        else:
            payload = {"words": words}
            if from_ is not None:
                payload["from"] = int(from_)
            if endpoint:
                payload["endpoint"] = True
        if self.L.arctag_push_words(self.h, json.dumps(payload).encode()) != 0:
            raise ValueError(f"arctag_push_words failed: {self._err()}")

    def endpoint(self):
        if self.L.arctag_push_words(self.h, b'{"endpoint": true}') != 0:
            raise ValueError(f"endpoint failed: {self._err()}")

    def poll(self) -> dict:
        return self._take(self.L.arctag_poll(self.h))

    def info(self) -> dict:
        return self._take(self.L.arctag_info(self.h))

    def reset(self):
        self.L.arctag_reset(self.h)

    def close(self):
        if self.h:
            self.L.arctag_destroy(self.h)
            self.h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# --- MODELS=host: the shared model server's tagger --------------------------------------------------------------------

_HOST = {"down": False}


def _connect_remote(*args, **kwargs):
    try:
        return RemoteTagger(*args, **kwargs)
    except Exception as e:   # fail open: this process loads its own tagger instead
        refused = "1013" not in str(e)   # 1013 = host's handle cap is full: try the host again next time
        if refused:
            _HOST["down"] = True
        log.error("MODELS=host but the model host at %s is unusable (%s: %s); falling back to the LOCAL Arc Tagger%s",
                  config.MODEL_HOST_URL, type(e).__name__, e, " for the rest of this process" if refused else "")
        print(f"[tagger] !!! MODELS=host: model host {config.MODEL_HOST_URL} unusable ({type(e).__name__}: {e}); "
              f"using the LOCAL Arc Tagger", flush=True)
        return None


class RemoteTagger:
    """Tagger's API over WS /v1/tag/stream: one dedicated handle in the model server for the life of this object."""

    def __init__(self, domain: str = "assistant", asr: str = "unknown", model_dir: Path = MODEL_DIR,
                 emit: str = "entity,number,pii,events,intent,act,safety,state", options: dict | None = None,
                 url: str | None = None):
        from websockets.sync.client import connect

        base = (url or config.MODEL_HOST_URL).replace("https://", "wss://").replace("http://", "ws://")
        q = {"domain": domain, "asr": asr}
        if emit:
            q["emit"] = emit
        self.timeout = config.MODEL_HOST_TIMEOUT_S
        self.ws = connect(f"{base}/v1/tag/stream?{urlencode(q)}", open_timeout=self.timeout,
                          compression=None, max_size=None)
        try:
            self.ws.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
        hello = json.loads(self.ws.recv(timeout=self.timeout))   # ConnectionClosed (1013 full, 1008 bad domain)
        if hello.get("op") != "ready":
            self.ws.close()
            raise RuntimeError(f"model host: unexpected hello {hello!r}")
        self.h = self.ws        # non-None while open (callers check `.h is None` for closed), like the native handle
        self.domain = domain
        for k, v in (options or {}).items():
            self.set(k, v)
        if options:
            self.info()         # surface a bad option now, as the native constructor would

    def _send(self, m: dict):
        if self.h is None:
            raise RuntimeError("tagger closed")
        self.ws.send(json.dumps(m))

    def _req(self, m: dict) -> dict:
        self._send(m)
        r = json.loads(self.ws.recv(timeout=self.timeout))
        if r.get("op") == "error":
            raise ValueError(f"model host tagger: {r.get('error')}")
        return r

    def set(self, key: str, value: str):
        self._send({"op": "set", "key": key, "value": str(value)})

    def tag_text(self, text: str) -> dict:
        return self._req({"op": "tag_text", "text": text})

    def push(self, words: list, from_: int | None = None, endpoint: bool = False):
        self._send({"op": "push", "words": list(words), "from": None if from_ is None else int(from_),
                    "endpoint": bool(endpoint)})

    def endpoint(self):
        self._send({"op": "endpoint"})

    def poll(self) -> dict:
        return self._req({"op": "poll"})

    def info(self) -> dict:
        return self._req({"op": "info"})

    def reset(self):
        self._send({"op": "reset"})

    def close(self):
        if self.h is not None:
            self.h = None
            try:
                self.ws.close()
            except Exception:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
