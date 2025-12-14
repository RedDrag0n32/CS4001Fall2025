import time
import subprocess

def run_command(label, command):
    print(f"\nRunning {label}...")
    start = time.time()
    subprocess.run(command, shell=True, check=True)
    end = time.time()
    latency = end - start
    print(f"{label} completed in {latency:.3f} seconds.")
    return latency

def main():
    # Step A: Stimulus delivery (network creation)
    latency_A = run_command("Step A: Build Network", "python build_network.py")

    # Step B: Decision making (update config + run simulation)
    latency_B1 = run_command("Update Configs", "python update_configs.py")
    latency_B2 = run_command("Run Simulation", "python run_bionet.py config.json")
    latency_B = latency_B1 + latency_B2

    # Step C: Motor response (output processing)
    latency_C = run_command("Step C: Check Output", "python check_output.py")

    # Save results
    with open("latency_results.txt", "w") as f:
        f.write(f"Step A (stimulus delivery): {latency_A:.3f} s\n")
        f.write(f"Step B (decision making): {latency_B:.3f} s\n")
        f.write(f"Step C (motor response): {latency_C:.3f} s\n")

    print("\n✅ Latency measurement complete. Results saved to latency_results.txt")

if __name__ == "__main__":
    main()
