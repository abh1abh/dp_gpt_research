import socket, time

for fam, name in [(socket.AF_INET, "IPv4"), (socket.AF_INET6, "IPv6")]:
    addr = socket.getaddrinfo("huggingface.co", 443, fam, socket.SOCK_STREAM)[0][4]
    s = socket.socket(fam, socket.SOCK_STREAM)
    s.settimeout(10)
    t = time.time()
    try:
        s.connect(addr)
        print(f"{name} OK      {addr[0]}  {time.time() - t:.2f}s")
    except Exception as e:
        print(f"{name} FAILED  {addr[0]}  {e}")
    finally:
        s.close()