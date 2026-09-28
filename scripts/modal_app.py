"""Modal fallback so Stage 0 is not blocked on a CARC queue slot."""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("uv")
    # vllm lives in the "gpu" extra, not the base dependencies (it broke `uv sync`
    # on non-Linux dev machines when it was a base dependency), so the image build
    # must request BOTH extras on the editable install. Do not "simplify" this
    # back to a bare `-e /repo`.
    .run_commands("uv pip install --system -e '/repo[dev,gpu]'")
)
app = modal.App("rl-amenability-stage0")


@app.function(gpu="L40S", timeout=24 * 60 * 60, image=image,
              mounts=[modal.Mount.from_local_dir(".", remote_path="/repo")])
def stage0():
    import subprocess
    subprocess.run(
        ["python", "-m", "scripts.run_stage0", "--out", "/repo/results/stage0_report.md"],
        cwd="/repo", check=True,
    )
