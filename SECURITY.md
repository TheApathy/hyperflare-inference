# Security Policy

## Reporting a vulnerability

This repository (`TheApathy/hyperflare-inference`) is an independent community
fork of [`NVIDIA/hyperflare-inference`](https://github.com/NVIDIA/hyperflare-inference).

- **Vulnerabilities in fork-specific changes** (scheduler policies, Selene
  multi-host compression, UMA compile/capture, Engram `io_uring` fanout, or
  fork packaging): report privately via **GitHub Private Vulnerability
  Reporting** on `TheApathy/hyperflare-inference` (Security → Advisories →
  Report a vulnerability). Do **not** open public GitHub issues, discussions, or
  pull requests for undisclosed vulnerabilities.
- **Vulnerabilities in upstream `NVIDIA/hyperflare-inference`:** report
  privately to the
  [NVIDIA Product Security Incident Response Team (PSIRT)](https://www.nvidia.com/en-us/security/report-vulnerability/)
  or email [psirt@nvidia.com](mailto:psirt@nvidia.com) encrypted with NVIDIA's
  public PGP key (see [NVIDIA's PSIRT policies](https://www.nvidia.com/en-us/security/psirt-policies/)).

Include the affected commit SHA, relevant configuration, a description of the
vulnerability and its potential impact, and reproduction steps or
proof-of-concept code.

## Model core integrity

Flare core signatures authenticate metadata and stored executable bytes. Executable
resources use one zstd level-3 frame per resource; compression is not encryption and
provides no confidentiality. The accepted publisher namespace must exactly match the
signed namespace. Treat signed cores as executable input, not sandboxed code.
