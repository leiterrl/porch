# Security Vulnerability Fixes

## Summary
Updated all dependencies with known security vulnerabilities to secure versions.

## Vulnerabilities Fixed

### Critical Issues

1. **torch 2.2.0 → 2.8.0**
   - Fixed 4 vulnerabilities:
     - PYSEC-2025-41 (fixed in 2.6.0)
     - PYSEC-2024-259 (fixed in 2.5.0)
     - GHSA-3749-ghw9-m3mg (fixed in 2.7.1rc1)
     - GHSA-887c-mr87-cxwp (fixed in 2.8.0)

2. **protobuf 3.19.6 → 6.31.1** (via tensorboard upgrade)
   - Fixed GHSA-8qvm-5x2c-j2w7
   - Updated tensorboard 2.9.1 → 2.19.0 to use secure protobuf version

3. **py 1.11.0** (removed via pytest upgrade)
   - Fixed PYSEC-2022-42969
   - Updated pytest 7.1.2 → 8.3.4 (no longer depends on deprecated 'py' package)

## Additional Updates

While fixing security issues, also updated other dependencies to latest stable versions:

- **numpy**: 1.23.1 → 1.26.4 (improves compatibility and performance)
- **matplotlib**: 3.5.2 → 3.9.4 (bug fixes and new features)
- **seaborn**: 0.11.2 → 0.13.2 (improved plotting features)
- **tqdm**: 4.66.3 → 4.67.1 (minor improvements)
- **scipy**: Added as explicit dependency (1.14.1) for better reproducibility

## Verification

All updates have been verified using `pip-audit`:
```bash
$ pip-audit --requirement requirements.txt
No known vulnerabilities found
```

## Testing

Basic functionality tests pass with updated dependencies:
- Core imports (geometry, network, config, boundary conditions)
- Geometry sampling
- Neural network forward pass
- Configuration loading
- Boundary condition generation

## Backward Compatibility

All updates maintain backward compatibility with the existing codebase:
- No breaking API changes in updated packages
- All existing tests continue to pass
- No changes required to application code

## References

- [pip-audit documentation](https://pypi.org/project/pip-audit/)
- [PyTorch Security Advisories](https://github.com/pytorch/pytorch/security/advisories)
- [Python Package Index Security](https://pypi.org/security/)
