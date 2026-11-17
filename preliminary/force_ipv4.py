import socket

_orig_getaddrinfo = socket.getaddrinfo

def _ipv4_only(host, port, family=0, *args, **kwargs):
    return _orig_getaddrinfo(host, port, socket.AF_INET, *args, **kwargs)

socket.getaddrinfo = _ipv4_only
