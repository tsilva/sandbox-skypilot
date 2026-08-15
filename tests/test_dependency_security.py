from importlib.metadata import version
from pathlib import Path
from runpy import run_path

from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]


def requirements(name: str) -> dict[str, str]:
    packages = {}
    for raw_line in (ROOT / name).read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", "-c ")):
            continue
        package, separator, package_version = line.partition("==")
        assert separator == "==", f"{name} must pin {line!r} exactly"
        packages[package.lower()] = package_version
    return packages


def test_gpu_manifests_keep_official_torchvision_pairs() -> None:
    rtx2060 = requirements("requirements-rtx2060-cifar.txt")
    gpu_image = requirements("requirements-ml-gpu-image.txt")

    assert (rtx2060["torch"], rtx2060["torchvision"]) == ("2.13.0", "0.28.0")
    assert (gpu_image["torch"], gpu_image["torchvision"]) == ("2.9.1", "0.24.1")


def test_training_manifests_share_the_patched_torch_floor() -> None:
    manifests = (
        "requirements-rtx2060-cnn.txt",
        "requirements-rtx2060-cifar.txt",
        "requirements-transformers.txt",
        "requirements-train.txt",
    )

    for manifest in manifests:
        assert requirements(manifest)["torch"] == "2.13.0"


def test_direct_high_risk_packages_are_patched() -> None:
    rtx2060 = requirements("requirements-rtx2060-cifar.txt")
    transformers = requirements("requirements-transformers.txt")

    assert Version(rtx2060["pillow"]) >= Version("12.3.0")
    assert Version(transformers["transformers"]) >= Version("5.5.0")
    assert "urllib3>=2.7.0" in (ROOT / "constraints.txt").read_text().splitlines()


def test_unpatched_torch_internal_apis_are_not_used() -> None:
    sources = "\n".join(path.read_text() for path in (ROOT / "scripts").glob("*.py"))

    assert "_call_end_callbacks_on_jit_fut" not in sources
    assert "torch.ao.nn.quantized" not in sources


def test_control_plane_is_outside_known_vulnerable_ranges() -> None:
    minimum_versions = {
        "aiohttp": "3.14.3",
        "click": "8.3.3",
        "cryptography": "50.0.0",
        "gitpython": "3.1.58",
        "pydantic-settings": "2.14.2",
        "python-multipart": "0.0.31",
        "setuptools": "83.0.0",
        "starlette": "1.3.1",
    }

    for package, minimum in minimum_versions.items():
        assert Version(version(package)) >= Version(minimum)


def test_requirement_manifests_reject_exotic_sources() -> None:
    for manifest in ROOT.glob("requirements-*.txt"):
        contents = manifest.read_text().lower()
        assert " @ " not in contents
        assert "git+" not in contents
        assert "http://" not in contents
        assert "https://" not in contents
        assert "file:" not in contents


def test_remote_installs_enforce_release_age() -> None:
    for task in (ROOT / "sky").glob("*.yaml"):
        contents = task.read_text()
        if "requirements-" not in contents:
            continue
        assert "uv==0.11.13" in contents
        assert "--exclude-newer '7 days'" in contents
        assert "pip install -r requirements-" not in contents


def test_click_independent_launcher_preserves_image_and_overrides_resources() -> None:
    load_task = run_path(str(ROOT / "scripts" / "sky_launch.py"))["load_task"]
    task = load_task("sky/setup-ml-gpu-image.yaml", infra="runpod", gpus="L4:1")
    (resource,) = task.resources

    assert resource.infra.cloud == "RunPod"
    assert resource.accelerators == {"L4": 1}
    assert resource.image_id == {
        None: "docker:us-docker.pkg.dev/sky-dev-465/skypilotk8s/skypilot-gpu:latest"
    }
