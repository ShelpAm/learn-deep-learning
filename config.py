import os


def setup():
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["QT_QPA_PLATFORM"] = "wayland"
