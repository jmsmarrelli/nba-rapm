# Current validation — October 22, 2026

## Connected GitHub review — October 22, 2026

GitHub integration access is verified for `jmsmarrelli`, including repository push permission. The remote `main` commit matched the preserved local base before preparing this review. The completed overhaul is preserved on `feat/portfolio-review-2026-10-22`; publication uses the connected GitHub API because shell Git cannot resolve `github.com`. Original local commits remain intact; the API publication carries their complete final file tree in a review commit based on the verified remote main.

The owner authorizes feature branches and draft pull requests. Merging into `main` requires separate owner approval. Repository visibility, profile settings and Pages deployment configuration are unchanged. Cross-repository links target `main` for eventual publication; review the individual draft PRs until those changes are approved and merged. Merge the navigation hub after its linked study changes.

Fresh Python test rerun: **7 passed in 22.15s**. The rerun uses the existing Python 3.14 environment, not a fresh Python 3.12 install.

## What changed

Original issue: Period/technical-FT handling, score reconciliation and future usage contaminated earlier reconstruction and forecast claims.

Implemented: Integrated the separately prepared repair release and seven regression checks; regenerated the article from corrected saved tables; fixed build/report targets and CI.

## Evidence and limits

7 Python tests passed.

Full raw reconstruction and refits were not rerun. Corrected three-year carryover MAE is worse than prior team strength; marked sequences are a possession proxy.

Test/runtime checks used an existing Python 3.14.7 environment (NumPy 2.5.2, pandas 3.0.5, SciPy 1.18.1, scikit-learn 1.9.0, matplotlib 3.11.1, pytest 9.1.1). The older studies specify Python 3.12 and different NumPy/pandas pins. A fresh installation of that pinned environment was not possible offline and is a release gate. R execution used R 4.6.1 with an existing package library. Earlier validation files and executed notebooks describe earlier runs and were preserved; notebooks were not re-executed in this review.

HTML now uses the pinned CommonMark renderer `markdown-it-py==4.2.0` with tables; local and CI use the same renderer. This replaces an unavailable local Python-Markdown package. Generated result CSVs retain full precision; readable article tables round for display only. The seven-project Quarto collection uses its separate Pandoc fallback.

## Earlier local review and publication record

Changes are local to `audit/2026-10-22-portfolio`. No push, merge, repository visibility change or public release occurred. The local PR draft records the intended summary. Publication requires a reachable authenticated GitHub connection and the owner's approval. Read [PROVENANCE.md](PROVENANCE.md) before describing personal authorship or validation in an application.


## Earlier October 22 visual and publication follow-up

Updated the static-renderer dependency documentation to match the already implemented Markdown renderer. This follow-up does not claim additional statistical testing beyond the earlier execution record.

The latest owner instruction authorizes feature branches and pull requests, but requires separate approval for any default-branch merge. Do not change repository visibility. The owner reports that GitHub is connected through ChatGPT; no GitHub repository or PR tools were exposed to this session when checked, so access and remote state remain unverified. Shell networking is still restricted. No further authentication attempt was made.
