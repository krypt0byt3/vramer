# Security Policy

## Supported versions

VRAMER is pre-1.0; security fixes land on the latest `0.2.x` and `master`.

| Version | Supported |
|---------|-----------|
| 0.2.x   | ✅        |
| < 0.2   | ❌        |

## Reporting a vulnerability

Please report suspected vulnerabilities **privately** — do not open a public
issue. Use GitHub's "Report a vulnerability" (Security → Advisories) on the
repository, or email the maintainer listed in the repo metadata. Include a
description, affected version/commit, and a minimal reproduction. We aim to
acknowledge within a few days.

## Scope / threat model

VRAMER is a local GPU library, not a network service. The relevant surfaces:

- **`MmapFileSource`** opens a caller-supplied path (`O_RDONLY`) — callers are
  responsible for trusting the path; VRAMER does not sandbox file access.
- **Offset/size validation** — `MemorySource` reads/writes are bounds-checked
  against the source size; report any path that bypasses `checked_range`.
- **The C ABI boundary** (`vramer_c.h`) — no C++ exception may cross it; a crash
  or UB reachable from valid C-ABI usage is in scope.
- Pinned-memory and device allocations are freed on teardown; report leaks or
  double-frees reachable from the public API.

Out of scope: denial of service from deliberately absurd configurations (e.g.
pinning more memory than exists), and issues in third-party code (CUDA, llama.cpp).
