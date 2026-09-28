"""Modal fallback so Stage 0 is not blocked on a CARC queue slot."""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("uv")
    # The repository must be present in the image BEFORE the editable install runs,
    # so add_local_dir comes first. modal.Mount.from_local_dir, which this used to
    # call, was removed in Modal 1.0; add_local_dir on the image is the current API.
    .add_local_dir(".", remote_path="/repo", copy=True)
    # vllm lives in the "gpu" extra, not the base dependencies (it broke `uv sync`
    # on non-Linux dev machines when it was a base dependency), so the image build
    # must request BOTH extras on the editable install. Do not "simplify" this
    # back to a bare `-e /repo`.
    .run_commands("uv pip install --system -e '/repo[dev,gpu]'")
)
app = modal.App("rl-amenability-stage0")


# The `modal` client is NOT a dependency of the GPU image: it belongs on the
# submitting machine, which is why it lives in the optional "launch" extra in
# pyproject.toml rather than in "gpu". Install it with `uv sync --extra launch`
# wherever `modal run` is invoked from.
@app.function(gpu="L40S", timeout=24 * 60 * 60, image=image)
def stage0():
    import subprocess
    subprocess.run(
        ["python", "-m", "scripts.run_stage0", "--out", "/repo/results/stage0_report.md"],
        cwd="/repo", check=True,
    )
