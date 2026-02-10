import os
import sys
import time
import subprocess
import requests

def verify_stability():
    print("=== STABILITY & CRASH VERIFICATION ===")
    
    # 1. Start the app in a subprocess
    print("[1/3] Starting app.py...")
    # Use -u for unbuffered output to catch logs immediately
    process = subprocess.Popen(
        [sys.executable, "-u", "app.py"],
        cwd=r"d:\CPE_CAS\CAS_V 0.7\CAS_V0.7\static",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    start_time = time.time()
    success = False
    load_count = 0
    
    try:
        # 2. Monitor logs for "Connected to local camera" and check for double loading
        print("[2/3] Monitoring logs for startup sequence...")
        while time.time() - start_time < 30: # 30s timeout for heavy AI load
            line = process.stdout.readline()
            if not line:
                break
            print(f"  LOG: {line.strip()}")
            
            if "Total references loaded" in line:
                load_count += 1
                if load_count > 1:
                    print("FAILED: Double loading detected! Reloader might still be active.")
                    process.terminate()
                    return False
            
            if "Running on http" in line:
                print("SUCCESS: Server is up and running.")
                success = True
                break
                
        if not success:
            print("FAILED: Server failed to reach 'Running' state within 30s.")
            process.terminate()
            return False

        # 3. Heartbeat check
        print("[3/3] Performing HTTP Heartbeat...")
        time.sleep(2) # Give it a moment
        try:
            resp = requests.get("http://127.0.0.1:5000/", timeout=5)
            if resp.status_code == 200:
                print("SUCCESS: Heartbeat 200 OK.")
            else:
                print(f"FAILED: Heartbeat returned {resp.status_code}.")
                process.terminate()
                return False
        except Exception as e:
            print(f"FAILED: Heartbeat connection error: {e}")
            process.terminate()
            return False

    finally:
        if process.poll() is None:
            print("Cleaning up process...")
            process.terminate()
            
    print("\n=== STABILITY VERIFIED: APP STARTUP IS ROCK SOLID ===")
    return True

if __name__ == "__main__":
    if verify_stability():
        sys.exit(0)
    else:
        sys.exit(1)
