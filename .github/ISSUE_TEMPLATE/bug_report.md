# Bug Report Template

```markdown
---
name: Bug Report
about: Report a bug in QuantumFortress 2.0
title: "[BUG] "
labels: 'bug, needs-triage'
assignees: ''

---

## 🐞 Describe the Bug

A clear and concise description of what the bug is. Include:
- What component is affected (e.g., quantum hypercube, topology engine, consensus mechanism)
- How it impacts security or functionality
- Whether it's related to quantum-topological metrics (TVI, Betti numbers, etc.)

## 🔍 Steps to Reproduce

1. Provide detailed steps to reproduce the issue:
   ```python
   # Example code if applicable
   from quantum_fortress import QuantumFortress
   qf = QuantumFortress(dimension=4)
   # ...
   ```

2. Expected behavior:
   - Describe what should happen under normal circumstances
   - Include expected quantum-topological metrics (e.g., TVI < 0.5, β₁ ≈ n)

3. Actual behavior:
   - Describe what actually happens
   - Include observed metrics (e.g., TVI = 0.7, β₁ = 3.2)
   - Note any error messages or anomalous behavior

## 📋 Environment Information

- **QuantumFortress Version**: [e.g., 2.0.0]
- **Python Version**: [e.g., 3.9.12]
- **Operating System**: [e.g., Ubuntu 22.04, Windows 11, macOS Monterey]
- **Hardware**:
  - CPU: [e.g., Intel i7-1185G7]
  - GPU: [e.g., NVIDIA RTX 3080, or "None"]
  - RAM: [e.g., 32GB]
- **Additional Dependencies**:
  - [e.g., CuPy 10.0.0, GUDHI 3.5.0]

## 📷 Additional Context

- **Screenshots** (if applicable):
  ![Screenshot](url)
  
- **Logs** (please redact sensitive information):
  ```
  [2025-08-15 10:23:45] ERROR: TopologyEngine - Betti number anomaly detected (β₁ = 3.2, expected 4.0)
  [2025-08-15 10:23:46] WARNING: TVI threshold exceeded (0.72 > 0.5)
  ```

- **Relevant Configuration**:
  ```yaml
  # configs/topology.yaml
  dimension: 4
  tvi_threshold: 0.5
  calibration_interval: 3600
  ```

- **Quantum-Topological Metrics** (if available):
  - TVI Score: [value]
  - β₀: [value], β₁: [value], β₂: [value]
  - Topological entropy: [value]
  - Naturalness coefficient: [value]

## 🧪 Impact Assessment

- **Severity**: [Critical/Major/Minor]
  - Critical: System-wide vulnerability, TVI > 0.8, or quantum state corruption
  - Major: Functional issue affecting core components, TVI between 0.5-0.8
  - Minor: Cosmetic or non-critical issue, TVI < 0.5
  
- **Reproducibility**: [Always/Often/Sometimes/Rarely]
- **Security Implications**: [None/Low/Medium/High]
  - Describe any potential security vulnerabilities this bug might introduce

## 🔮 Possible Cause (Optional)

If you have any insights about what might be causing the issue:
- [e.g., "This appears related to the adaptive dimension expansion logic when TVI exceeds threshold"]
- [e.g., "May be connected to GPU acceleration in the topology engine"]
```
