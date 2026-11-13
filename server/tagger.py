"""ctypes binding to libarctag (Arc Tagger v0.3, native C backend, int8).

One handle per stream: a handle is not thread-safe. Each call costs about 1-3 ms per word on the native backend.
"""
from __future__ import annotations

import ctypes
import json
import os
import threading
from pathlib import Path

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
