"""Relay UDP packets sent to 127.0.0.1:<port> (inside WSL) out to the real
Windows host IP on the same port.

Why: rlviser_py always targets 127.0.0.1 for the training->rlviser packets,
but WSL2's localhost-forwarding for UDP isn't relaying that traffic to the
Windows host on this machine right now (confirmed: a raw packet to the WSL
gateway IP arrives at a Windows listener instantly; the identical packet to
127.0.0.1 never does). This sidesteps the broken forwarding entirely at the
application layer instead of the kernel/iptables layer (which hit a separate
Linux route_localnet restriction on DNAT-ing loopback-destined traffic).

Usage: python wsl_udp_relay.py <port>
"""
import socket
import subprocess
import sys

port = int(sys.argv[1]) if len(sys.argv) > 1 else 45243

gateway = (
    subprocess.check_output(["ip", "route"])
    .decode()
    .split("default via ")[1]
    .split()[0]
)
print(f"[wsl_udp_relay] relaying 127.0.0.1:{port} -> {gateway}:{port}", flush=True)

listen_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
listen_sock.bind(("127.0.0.1", port))

forward_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

count = 0
while True:
    data, _addr = listen_sock.recvfrom(65536)
    forward_sock.sendto(data, (gateway, port))
    count += 1
    if count <= 5 or count % 500 == 0:
        print(f"[wsl_udp_relay] relayed packet #{count} ({len(data)} bytes)", flush=True)
