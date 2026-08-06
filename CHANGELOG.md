# Changelog

All notable changes to HyTorch will be recorded in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
HyTorch uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- PyTorch-shaped, directory-native model checkpoints with `state_dir()`,
  `hytorch.save()`, `hytorch.load()`, and `load_state_dir()`.

## [0.1.0] - 2026-08-05

### Added

- Initial Python library for Git-backed agent meta-networks.
- PyTorch-shaped `Space`, `Module`, `Parameter`, `Linear`, `Loss`, and `DFM`
  APIs.
- Dockerized Pi harness with local and remote Docker context support.
- Offline unit tests and an opt-in real Pi integration test.
