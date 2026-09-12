# KT release branch

This branch is based on LLaMA-Factory main `100e9a42c6c09f8f7849b70d60f3da445fb2024b`.
It is an explicit release-specific dependency variant, **not upstream main**.
No Trainer, model loading, data processing, YAML, optimizer, or checkpoint implementation is changed.

The wheel is built from this branch through the normal Hatchling build backend.
There is no build-profile switch and no post-build wheel patching.

| Distribution | Required version |
| --- | --- |
| llamafactory | `0.9.6.dev0+kt.20260912` |
| transformers-kt | `5.6.0.post5` |
| accelerate-kt | `1.14.0.post3` |
| peft | `0.18.1+kt.20260912` |
| trl | `0.24.0+kt.20260912` |

PEFT/TRL are separately approved dependency-only rebuilds of the public releases.
They retain the original runtime files and dependency bounds, with provenance and SHA256 records.
The local versions intentionally prevent pip from silently choosing the unmodified public wheels.
Release artifacts must provide these wheels; this is not a standalone install from today's public index.

Runtime dependency checks use the same exact versions as the wheel metadata, including local versions.
Transformers version detection queries the actual `transformers-kt` distribution rather than treating a
missing upstream `transformers` distribution as version `0.0.0`. Python import names are unchanged.

Use a fresh environment and normal dependency resolution. Do not install upstream `transformers` or
`accelerate` alongside their KT counterparts. Do not use `--no-deps`, `--ignore-installed`,
`DISABLE_VERSION_CHECK`, editable installs, shared site-packages, or runtime patches for acceptance.

Source build:

```bash
python -m build --wheel
```

Building this wheel does not prove model acceptance. Final delivery still requires the selected wheels
to pass clean installation, training, full optimizer resume, SGLang ordinary/expert LoRA loading and
generation, and the non-KT/cross-model regression matrix. The final wheel SHA256 must match the tested
artifact. No upstream PR is submitted as part of preparing this branch.
