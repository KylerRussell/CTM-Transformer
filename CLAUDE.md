# CTM-Transformer research

**The goal of this research:** a **CTM Transformer**, meaning a Transformer adapted with the theory of the Continuous Thought Machine (CTM) and Continuous Memory Machine (CMM) papers so that it works for LLM language modelling.
- The project does not move away from the Transformer. The CTM Transformer is a merge: the Transformer's own components take on CTM/CMM mechanisms. It is not a separate CTM module, and not a Transformer backbone feeding a CTM head.
- The paper compares **the CTM Transformer's best against the RDT's best**, with the plain Transformer as the baseline.
- Every final comparison and scaling ladder includes the CTM Transformer arm at the same standard as the RDT arm: matched size, tuned learning rate, a 1-tick twin, and replicate seeds.

Design: `research/CTM_TRANSFORMER_DESIGN.md`. RDT recipe history and the diagnosis of the earlier separate-head CTM: `research/RDT_RECIPE.md`.

Environment: venv `~/.venvs/ctm-research`; `LD_LIBRARY_PATH=~/.local/share/ctm-nvidia-driver/usr/lib/x86_64-linux-gnu`; `PYTHONPATH=.`. Launch long runs detached with `setsid nohup`.
