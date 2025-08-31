# .github/PULL_REQUEST_TEMPLATE.md for QuantumFortress 2.0

```markdown
<!--- Provide a general summary of your changes in the Title above -->

## 🌌 Quantum-Topological Context

Describe how this pull request relates to the quantum-topological security model of QuantumFortress 2.0:
- How does it impact the adaptive hypercube structure?
- Does it affect TVI (Topological Vulnerability Index) calculation or interpretation?
- Which topological invariants (Betti numbers, homology, etc.) are involved?

## 🧩 Description of Changes

Provide a detailed description of what this PR accomplishes:
- What problem does it solve?
- How does it align with the core principle: "Topology isn't a hacking tool, but a microscope for diagnosing vulnerabilities"?
- What components are affected? (e.g., `src/core/adaptive_hypercube.py`, `src/topology/metrics.py`)

## 📐 Technical Implementation

Explain your implementation approach:
```python
# Include relevant code snippets if applicable
def example_implementation():
    """Brief description of key implementation details"""
    # Implementation specifics
    return quantum_topology.calculate_enhanced_tvi()
```

- Key algorithms or mathematical principles used:
  - [ ] Persistent homology
  - [ ] Betti number calculations
  - [ ] Sheaf theory applications
  - [ ] Hyperbolic clustering
  - [ ] WDM parallelism
  - [ ] Other: _______________

- Quantum-topological considerations:
  - How does this maintain or improve the quantum state stability?
  - What impact does it have on the hypercube topology?

## 🧪 Testing and Validation

Describe how you tested your changes:
- Unit tests added/modified: `tests/unit/...`
- Integration tests added/modified: `tests/integration/...`
- Performance benchmarks: 
  - Signature verification speed: _______ vs previous _______ (expected improvement: 4.5x)
  - Nonce search speed: _______ vs previous _______ (expected improvement: 4.5x)
  - TVI for secure wallets: _______ (expected: < 0.1)
  - TVI for vulnerable wallets: _______ (expected: > 0.8)

- Test coverage impact:
  - [ ] Increased test coverage
  - [ ] Maintained existing coverage
  - [ ] Coverage decrease (explain why): _______________

## 📈 Performance Impact

Quantify the expected performance impact:
- Expected speedup in signature verification: _______x
- Expected speedup in nonce search: _______x
- Memory usage impact: _______%
- Energy efficiency impact: _______%

*Note: QuantumFortress 2.0 targets minimum 4.5x speedup in both signature verification and nonce search through WDM parallelism*

## 🛡️ Security Implications

Analyze security implications:
- Does this change affect TVI calculation or interpretation? [Yes/No]
  - If yes, explain how: _______________
- Does this impact quantum resistance? [Yes/No]
  - If yes, explain how: _______________
- Does this introduce any new attack vectors? [Yes/No]
  - If yes, explain mitigation: _______________

*Remember: All changes must maintain or improve the TVI-based security model where:*
- *TVI < 0.5 indicates secure implementation*
- *TVI > 0.5 indicates potential vulnerability*

## 📚 Documentation Updates

Check all that apply:
- [ ] Updated API documentation
- [ ] Added new tutorial/example
- [ ] Updated architecture documentation
- [ ] Added/modified security whitepaper section
- [ ] Updated migration guide

## 🔗 Related Issues

- Fixes #_____
- Related to #_____
- Part of roadmap phase: [ ] QuantumSeed 2.0 [ ] QuantumLink 2.0 [ ] QuantumChain 2.0 [ ] QuantumEra 2.0

## ✅ Checklist

Before submitting your PR, please ensure you've completed the following:

- [ ] I've read the [CONTRIBUTING.md](CONTRIBUTING.md) guidelines
- [ ] I've added necessary tests to cover new code
- [ ] I've updated documentation accordingly
- [ ] I've ensured code passes all linting/formatting checks (Black, isort)
- [ ] I've verified type hints are correct (mypy passes)
- [ ] I've confirmed performance meets QuantumFortress 2.0 targets
- [ ] I've validated TVI calculations remain accurate and meaningful
- [ ] I've checked for potential security implications

## 🌟 Additional Context

Add any other context about the pull request here:
- Research papers or references consulted
- Alternative approaches considered
- Future work or follow-up issues
- Special considerations for reviewers
```
