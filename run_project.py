"""run_project.py -- runs every stage of the Copilot in order (same commands as the README's Reproduce section).
Usage (from the repository root, inside the virtual environment):  python run_project.py"""
import subprocess, sys, time

STAGES = [("generate raw data", "generate_data.py"),
          ("Part A: audit + clean + store", "src/part_a_clean.py"),
          ("Part A: validation", "src/validate.py"),
          ("Part A: SQL + pandas analysis", "src/part_a_sql.py"),
          ("Part B: RFM, K-means, PCA", "src/part_b_segments.py"),
          ("Part B: lapse model", "src/part_b_lapse.py"),
          ("Part C: ticket triage", "src/part_c_triage.py"),
          ("Part D: RAG + refund agent", "src/part_d_run.py")]
for name, script in STAGES:
    t = time.time()
    print(f"\n########## {name}  ({script})", flush=True)
    r = subprocess.run([sys.executable, script])
    if r.returncode != 0:
        sys.exit(f"stage failed: {script}")
    print(f"########## done in {time.time()-t:.1f}s", flush=True)
print("\nALL STAGES COMPLETED")
