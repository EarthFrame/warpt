"""Network test server for warpt stress tests.

This server handles connections from NetworkPointToPointTest and
NetworkBidirectionalTest, automatically detecting which test is running.

Usage:
    python -m warpt.server
"""

import socket
import threading
import time


def handle_client(client_sock: socket.socket, addr: tuple) -> None:
    """Handle a single client connection.

    Automatically detects test type:
    - Latency test: Small payload (<10KB) → echo back
    - Upload test: Large payload → just receive
    - Bidirectional test: "BIDIR:..." header → receive AND send

    Args:
        client_sock: Connected client socket.
        addr: Client address (ip, port).
    """
    try:
        client_sock.settimeout(1.0)
        data = client_sock.recv(100000)  # Initial chunk

        if not data:
            return

        # Mode 1: Check for bidirectional test header FIRST
        if data.startswith(b"BIDIR:"):
            handle_bidirectional(client_sock, addr, data)
            return

        # Mode 2: Latency test (small payload)
        if len(data) < 10000:
            client_sock.sendall(data)
            print(f"Latency test from {addr}: echoed {len(data)} bytes")
            return

        # Mode 3: Upload-only test (default)
        handle_upload_only(client_sock, addr, data)

    except Exception as e:
        print(f"Error handling client {addr}: {e}")
    finally:
        client_sock.close()


def handle_upload_only(
    client_sock: socket.socket, addr: tuple, initial_data: bytes
) -> None:
    """Handle upload-only test (NetworkPointToPointTest).

    Just receives data from client, doesn't send anything back.

    Args:
        client_sock: Connected client socket.
        addr: Client address.
        initial_data: First chunk of data already received.
    """
    total_received = len(initial_data)

    try:
        while True:
            more = client_sock.recv(1024 * 1024)  # 1MB chunks
            if not more:
                break
            total_received += len(more)

        total_mb = total_received / (1024 * 1024)
        print(f"Upload test from {addr}: received {total_mb:.2f} MB")

    except (TimeoutError, OSError):
        pass


def handle_bidirectional(
    client_sock: socket.socket, addr: tuple, header: bytes
) -> None:
    """Handle bidirectional test (NetworkBidirectionalTest).

    Receives data from client AND sends data back simultaneously.

    Header format: b"BIDIR:<duration>:<chunk_size>"
    Example: b"BIDIR:60:1048576" = 60 seconds, 1MB chunks

    Args:
        client_sock: Connected client socket.
        addr: Client address.
        header: Header containing test parameters.
    """
    try:
        # Parse header: "BIDIR:60:1048576"
        parts = header.decode().strip().split(":")
        if len(parts) < 3:
            header_preview = header[:50].decode(errors="replace")
            print(f"Invalid BIDIR header from {addr}: {header_preview}")
            return

        duration = int(parts[1])
        chunk_size = int(parts[2])

        print(
            f"Bidirectional test from {addr}: "
            f"{duration}s, {chunk_size / 1024:.0f}KB chunks"
        )

        # Shared state for threads
        results = {"upload_bytes": 0, "download_bytes": 0}

        # Thread 1: Receive data from client (upload test)
        def receive_worker() -> None:
            """Receive data from client."""
            total = 0
            start = time.time()
            try:
                while (time.time() - start) < duration:
                    data = client_sock.recv(1024 * 1024)
                    if not data:
                        break
                    total += len(data)
            except (TimeoutError, OSError):
                pass
            results["upload_bytes"] = total

        # Thread 2: Send data to client (download test)
        def send_worker() -> None:
            """Send data to client."""
            chunk = bytes(range(256)) * (chunk_size // 256)
            total = 0
            start = time.time()
            try:
                while (time.time() - start) < duration:
                    client_sock.sendall(chunk)
                    total += len(chunk)
            except (TimeoutError, BrokenPipeError, OSError):
                pass
            results["download_bytes"] = total

        # Start both threads
        recv_thread = threading.Thread(target=receive_worker, daemon=True)
        send_thread = threading.Thread(target=send_worker, daemon=True)

        recv_thread.start()
        send_thread.start()

        # Wait for both to complete
        recv_thread.join(timeout=duration + 5)
        send_thread.join(timeout=duration + 5)

        # Report results
        upload_mb = results["upload_bytes"] / (1024 * 1024)
        download_mb = results["download_bytes"] / (1024 * 1024)
        print(
            f"Bidirectional complete from {addr}: "
            f"received {upload_mb:.2f} MB, sent {download_mb:.2f} MB"
        )

    except (ValueError, IndexError) as e:
        print(f"Error parsing BIDIR header from {addr}: {e}")
    except Exception as e:
        print(f"Error in bidirectional test from {addr}: {e}")


def run_server(host: str = "0.0.0.0", port: int = 5201) -> None:
    """Run the network test server.

    Args:
        host: Host address to bind to. Default "0.0.0.0" (all interfaces).
        port: Port to listen on. Default 5201.
    """
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(5)

    print(f"warpt network test server listening on {host}:{port}")
    print("Supports: NetworkPointToPointTest and NetworkBidirectionalTest")
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            client, addr = server.accept()
            # Handle each client in a thread (support concurrent tests)
            thread = threading.Thread(
                target=handle_client, args=(client, addr), daemon=True
            )
            thread.start()

    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        server.close()


def main() -> None:
    """Run the network test server."""
    import argparse

    parser = argparse.ArgumentParser(description="warpt network test server")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host address to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port", type=int, default=5201, help="Port to listen on (default: 5201)"
    )

    args = parser.parse_args()

    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
