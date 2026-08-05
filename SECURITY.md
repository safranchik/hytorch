# Security Policy

## Supported versions

HyTorch is in alpha. Security fixes apply to the latest release on the default
branch.

## Report a vulnerability

Use GitHub's private vulnerability reporting feature for this repository. Do
not open a public issue for a vulnerability that can expose credentials,
modify repositories, escape container boundaries, or execute untrusted code.

Include the affected version, the required environment, a minimal
reproduction, and the expected impact. Remove all real credentials and private
data from the report.

## Trust boundaries

HyTorch executes coding agents against directory trees and Git repositories.
Treat agent output and model-generated code as untrusted. Use isolated Docker
environments. Use credentials with the least required privilege. Review
workspace mutations before you use a trained model in a sensitive system.

HyTorch does not load a project `.env` file. Store agent variables in
`.hytorch.env`, the global HyTorch secrets file, or the file selected by
`HYTORCH_ENV_FILE`. Never commit these files.
