# KT build profile

Base: LLaMA-Factory main `100e9a42c6c09f8f7849b70d60f3da445fb2024b`.
This branch is prepared for a KT release; no upstream LF PR has been submitted.

## Build-time selection

Both profiles package the same Python runtime files. Default dependency requirements
and the version in `extras/env.py` are unchanged. The opt-in profile changes wheel
dependency metadata and gives the wheel a distinct local version:

```bash
python -m build --wheel --outdir dist/standard
LLAMAFACTORY_BUILD_PROFILE=kt python -m build --wheel --outdir dist/kt
```

`LLAMAFACTORY_BUILD_PROFILE` accepts `standard` (default) or `kt`; unknown values fail.
It is a release-engineering option, not a runtime flag. The normal Hatchling backend
generates the metadata before building; no finished wheel is patched.

| KT distribution | Candidate version |
| --- | --- |
| llamafactory | `0.9.6.dev0+kt.20260912` |
| transformers-kt | `5.6.0.post5` |
| accelerate-kt | `1.14.0.post3` |
| peft | `0.18.1+kt.20260912` |
| trl | `0.24.0+kt.20260912` |

The source and runtime version remain the upstream LF version. The installed wheel
version and SHA256 identify the KT packaging variant. LF and the approved dependency-only
PEFT/TRL rebuilds must be supplied as explicit, hash-locked wheel artifacts; their local
versions are not uploadable to public PyPI. They are additional training tools, not new
members of the five core release carriers. Do not install both packaging variants.

## Runtime selection

The launcher reads YAML/JSON and CLI overrides before dependency checks and torchrun
dispatch. `use_kt: true` selects KT execution; false or omitted preserves the normal
execution path. CLI `--use_kt`/`--use-kt` uses the existing HF boolean parser. A separate
`USE_KT` environment variable is not required and does not override the YAML.

Version detection checks the installed distribution, not the execution flag. A KT
installation is checked against its candidate dependencies even for non-KT training;
that does not enable KT execution. Overlapping upstream Transformers/Accelerate
distributions in a KT environment are rejected. Normal LF environments retain their
original dependency checks. The shared config reader also corrects the existing JSON
file-read call; no Trainer, model, optimizer or checkpoint implementation is changed.

## Acceptance

Build success alone is not release acceptance. Install the explicit wheels with normal
dependency resolution in fresh environments; run `pip check` and audit import paths and
namespace ownership. Do not use `--no-deps`, `--ignore-installed`, disabled version checks,
editable installs, shared site-packages or runtime patches.

Release readiness additionally requires Kimi training, full optimizer resume, new-process
SGLang ordinary/expert LoRA loading and generation, style evaluation and the agreed
non-KT/cross-model regression matrix. Record the tested wheel SHA256 and source commits.
Publication requires separate approval; this document does not claim the gates passed.
