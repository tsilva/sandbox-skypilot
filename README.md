<p align="center">
  <!-- repo-tagline:start -->
  <strong>☁️ Try SkyPilot across RunPod, Hyperbolic, and Lambda Cloud 🧪</strong>
  <!-- repo-tagline:end -->
</p>

This is a small local setup for trying SkyPilot on RunPod, Hyperbolic, and Lambda Cloud without rewriting task YAMLs.

## Install

```bash
uv --cache-dir .uv-cache sync --locked --no-config --exclude-newer '7 days'
chmod +x scripts/skyctl
keyenv doctor
```

The three provider keys declared in `.keyenv.toml` live in macOS Keychain.
Launch provider commands through `keyenv run -- ...`; the scripts continue to
read ordinary environment variables. Keep only non-secret overrides such as
`SKY_CLUSTER_PREFIX` in `.env`.

For RunPod, `keyenv run -- uv --cache-dir .uv-cache run runpod config` is also supported. For Hyperbolic, add your local SSH public key in the Hyperbolic dashboard because SkyPilot needs SSH access to launched instances. For Lambda Cloud, `keyenv run -- scripts/configure_lambda` can write SkyPilot's required credential file with mode `0600`.

Verify credentials:

```bash
keyenv run -- ./scripts/skyctl check
```

## Hyperbolic Setup

SkyPilot's Hyperbolic provider expects your API key at `~/.hyperbolic/api_key`.
You can create it from the Keychain value with:

```bash
keyenv run -- ./scripts/configure_hyperbolic
uv --cache-dir .uv-cache run sky check hyperbolic
```

Hyperbolic also requires an SSH public key for on-demand GPU instances. Add your
local public key in the Hyperbolic dashboard:

```bash
cat ~/.ssh/id_ed25519.pub
```

If that file does not exist, generate one first:

```bash
ssh-keygen -t ed25519 -C "your-email@example.com"
```

## Lambda Cloud Setup

SkyPilot's Lambda Cloud provider expects your API key at `~/.lambda_cloud/lambda_keys`.
You can create it from the Keychain value with:

```bash
keyenv run -- ./scripts/configure_lambda
uv --cache-dir .uv-cache run sky check lambda
```

## Switch Providers

```bash
./scripts/skyctl use runpod
keyenv run -- ./scripts/skyctl launch

./scripts/skyctl use hyperbolic
keyenv run -- ./scripts/skyctl launch sky/hyperbolic-smoke.yaml

./scripts/skyctl use lambda
keyenv run -- ./scripts/skyctl launch sky/lambda-smoke.yaml
```

The active provider is stored in `.sky-provider`, which is intentionally ignored by Git.

Launches use `scripts/sky_launch.py` because SkyPilot 0.12's hidden backend flag is incompatible
with the security-fixed Click 8.3+ API. Other SkyPilot commands continue to use the upstream CLI.

All remote requirement manifests use the shared `constraints.txt`, exact runtime pins, and a
seven-day release-age check in CI. The training files use the official Torch 2.13 / torchvision
0.28 pair; the separate Unsloth RTX 4090 image retains its supported Torch 2.9.1 / torchvision
0.24.1 pair.

## Useful Commands

```bash
./scripts/skyctl current
./scripts/skyctl gpus
./scripts/skyctl status
./scripts/skyctl logs
./scripts/skyctl stop
./scripts/skyctl down
```

`launch` uses a separate cluster per provider:

- `sandbox-sky-runpod`
- `sandbox-sky-hyperbolic`
- `sandbox-sky-lambda`

Override the prefix with `SKY_CLUSTER_PREFIX` in `.env`.

## Tasks

- `sky/hello.yaml`: low-resource hello-world task; useful for checking that provisioning works.
- `sky/runpod-smoke.yaml`: RunPod GPU smoke test that runs `scripts/runpod_smoke.py`.
- `sky/hyperbolic-smoke.yaml`: Hyperbolic H100 smoke test that runs `scripts/cloud_smoke.py`.
- `sky/lambda-smoke.yaml`: Lambda Cloud GPU smoke test that runs `scripts/cloud_smoke.py`.
- `sky/train-pytorch.yaml`: provider-selectable PyTorch training smoke test.
- `sky/gpu-smoke.yaml`: GPU task that runs `nvidia-smi`; useful once credentials and quota are known good.

## PyTorch Training Test

Use the same task file and choose the provider with `--infra`:

```bash
uv run python scripts/sky_launch.py --down --infra runpod -c train-runpod sky/train-pytorch.yaml
uv run python scripts/sky_launch.py --infra hyperbolic -c train-hyperbolic sky/train-pytorch.yaml
uv run python scripts/sky_launch.py --infra lambda -c train-lambda sky/train-pytorch.yaml
```

Or use the provider-specific task files:

```bash
uv run python scripts/sky_launch.py --down -c train-runpod sky/train-pytorch-runpod.yaml
uv run python scripts/sky_launch.py -c train-hyperbolic sky/train-pytorch-hyperbolic.yaml
uv run python scripts/sky_launch.py -c train-lambda sky/train-pytorch-lambda.yaml
```

The task installs `requirements-train.txt` on the remote instance, then runs
`scripts/train_torch.py` on CUDA if a GPU is available.

RunPod supports `--down`, which tears down the cluster after the job finishes.
Hyperbolic does not support auto-down in this SkyPilot version, so terminate it
manually as soon as the job finishes:

```bash
sky down train-hyperbolic
```

You can also bypass `skyctl` with the repository's Click-independent launcher:

```bash
uv --cache-dir .uv-cache run python scripts/sky_launch.py --infra runpod -c sandbox-sky-runpod sky/hello.yaml
uv --cache-dir .uv-cache run python scripts/sky_launch.py -c sandbox-sky-hyperbolic sky/hyperbolic-smoke.yaml
uv --cache-dir .uv-cache run python scripts/sky_launch.py -c sandbox-sky-lambda sky/lambda-smoke.yaml
```

For a Hyperbolic test:

```bash
uv --cache-dir .uv-cache run python scripts/sky_launch.py -c hyperbolic-smoke sky/hyperbolic-smoke.yaml
```

Hyperbolic does not support `sky stop` in this SkyPilot version; use `sky down`
to terminate the instance:

```bash
uv --cache-dir .uv-cache run sky down hyperbolic-smoke
```
