# Can ticks beyond 8 contribute? (100M-token screen)

Protocol and rules: `scripts/tick_contribution.py` (docstring).

| Arm | Seed | 1 | 4 | 8 | 16 | 32 | 64 | 96 | Reference at 16 | Ticks pay | vs reference | Passes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| Dans_film | 1 | 4.777 | 4.745 | 4.744 | 4.744 | 4.744 | 4.744 | 4.744 | 4.747 | no | -0.003 | no |
| CTsq_film | 1 | 4.821 | 4.742 | 4.741 | 4.741 | 4.741 | 4.741 | 4.741 | 4.731 | no | +0.010 | no |
| Dans_mtp | 1 | 4.842 | 4.817 | 4.817 | 4.817 | 4.817 | 4.817 | 4.817 | 4.747 | no | +0.070 | no |
| CTsq_mtp | 1 | 4.871 | 4.811 | 4.811 | 4.811 | 4.811 | 4.811 | 4.811 | 4.731 | no | +0.079 | no |
| Dans_m32 | 1 | 4.815 | 4.787 | 4.786 | 4.786 | 4.786 | 4.786 | 4.786 | 4.747 | no | +0.039 | no |
| CTsq_m32 | 1 | 4.826 | 4.762 | 4.761 | 4.761 | 4.761 | 4.761 | 4.761 | 4.731 | no | +0.029 | no |

## Outcomes

- **film:** Dans: fails the screen; CTsq: fails the screen
- **mtp:** Dans: fails the screen; CTsq: fails the screen
- **m32:** Dans: fails the screen; CTsq: fails the screen
