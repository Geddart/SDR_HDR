# Pull request

## What and why
<!-- What does this change, and what problem does it solve? -->

## How it was tested
<!-- Commands run, new tests added. CI runs `pytest -m "not network"` on Python 3.10-3.12. -->

## Checklist
- [ ] `pytest -m "not network"` passes locally
- [ ] Added/updated tests for any behavioral change
- [ ] No model layer names or tensor shapes in `models/` were renamed/reshaped
- [ ] If color math changed, a numerical test proves correctness
- [ ] Docs (README / CHANGELOG) updated if user-facing behavior changed
