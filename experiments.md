# Experiments & Tasks

Status: `TODO` | `IN PROGRESS` | `DONE` | `BLOCKED`

## Experiments

| ID | Name | Status | Branch | Depends On | Description |
|----|------|--------|--------|------------|-------------|
| E1 | Binary reward training | TODO | — | — | Run `python train.py --reward binary --max_steps 200 --output_dir checkpoints/binary`. Confirm checkpoints at steps 50, 100, 150, 200. Record wall-clock time per step. |
| E2 | Dense reward training | TODO | — | — | Run `python train.py --reward dense --max_steps 200 --output_dir checkpoints/dense`. Confirm checkpoints at steps 50, 100, 150, 200. Comparable per-step time expected. |
| E3 | Eval all checkpoints | TODO | — | E1, E2 | Run `bash eval/eval_all_checkpoints.sh binary` and `bash eval/eval_all_checkpoints.sh dense`. Record eval wall-clock time. |
| E4 | Plot trajectories & go/no-go | TODO | — | E3 | Run `python analysis/plot_trajectories.py --results_dir results`. Record gap at step 200. Apply go/no-go criteria from logbook.md. |

## Tasks

| ID | Name | Status | Branch | Description |
|----|------|--------|--------|-------------|
