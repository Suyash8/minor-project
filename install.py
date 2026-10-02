#!/usr/bin/env python3
"""
Automated dependency installer for Aerial OBB Detection & Benchmark Suite.
Prefers uv for instant installations, falling back to standard pip.
"""

import os
import sys
import subprocess
import shutil

# Ensure Windows cp1252/terminal compatibility
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def main():
    print("==================================================")
    print(" Aerial OBB Benchmark Suite - Dependency Installer")
    print("==================================================")
    
    use_uv = shutil.which("uv") is not None
    cmd = [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
    if use_uv:
        print("[*] 'uv' detected! Using uv pip for high-speed package installation.")
        cmd = ["uv", "pip", "install", "-r", "requirements.txt"]
    else:
        print("[*] Using standard pip to install requirements.")

    try:
        subprocess.check_call(cmd)
        print("\n[✓] All dependencies successfully installed!")
    except subprocess.CalledProcessError as e:
        print(f"\n[!] Installation failed with error code: {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
