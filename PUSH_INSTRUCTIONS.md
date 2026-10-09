# Updating the public repository

The folder `heat-aware-scheduling/` next to this file is the complete update of
https://github.com/amr21006/heat-aware-scheduling for the revision: code (steps 01-17, including 12b,
15b, 16c and the scheduling options), derived data (risk calendars, archived NBM forecasts, trade and
rate weights, resource exposure, parsed project networks), per-scenario schedules (main run and the
1 April 2024 start) and fitted models. Tables, figures and run logs are not included. Raw third-party data (OSHA
Severe Injury Reports, DSLIB workbooks, O*NET, BLS QCEW, NOAA station files) are not included; the
scripts download them. No file exceeds 95 MB (total about 51 MB).

Push it before submitting the revision, because the data availability statement refers to it.

```bash
git clone https://github.com/amr21006/heat-aware-scheduling.git
cd heat-aware-scheduling
# empty the working copy (the .git folder stays), so files dropped in the revision are also removed
git rm -r -q .
# copy everything from R1/05_repo_stage/heat-aware-scheduling/ into this folder
git add -A
git status                      # review the list of changed files
git commit -m "Revision: planning-time evaluation, sensitivity analyses, derived data and results"
git push origin main
```

To regenerate the staged copy after any change: `bash R1/stage_repo.sh`.
