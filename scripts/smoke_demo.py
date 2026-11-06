"""Start the npm preview, verify its HTTP response, and stop every child process."""

import os
import signal
import subprocess
import time
import urllib.request


def main() -> None:
    command = ["npm", "run", "dev", "--", "--host", "127.0.0.1", "--port", "7100"]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        for _ in range(60):
            if process.poll() is not None:
                output = process.stdout.read() if process.stdout else ""
                raise RuntimeError(f"Demo exited before becoming ready:\n{output}")
            try:
                with urllib.request.urlopen("http://127.0.0.1:7100/", timeout=1) as response:
                    body = response.read().decode("utf-8")
                    if response.status == 200 and "gradio" in body.lower():
                        print(f"Demo HTTP {response.status}; Gradio page received; server stopped")
                        return
            except OSError:
                time.sleep(0.5)
        raise RuntimeError("Demo did not become ready within 30 seconds")
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)


if __name__ == "__main__":
    main()
