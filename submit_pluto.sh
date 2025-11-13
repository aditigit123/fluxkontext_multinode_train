#!/bin/bash

################################################################################
# Flux Kontext Multinode Training Pluto Submission Script
################################################################################
# This script mirrors the structure of the Mori "submit_pluto.sh" helper that
# Pluto users rely on.  It prepares the environment (Job Butler registration,
# optional MinIO staging, Git checkout) and finally executes the repo's
# production multinode entrypoint: pluto_main_multinode_accelerate.sh.
#
# Update the USERDEF_ variables below before launching your Pluto job.  They are
# separated so you can quickly adapt the launch to a different experiment without
# touching the rest of the script.
################################################################################

set -euo pipefail

################################################################################################
# User configuration (edit these for each run)
################################################################################################
export USERDEF_EMAIL="your_email@adobe.com"
export USERDEF_SLACK_CHANNEL_ID="CXXXX"
export USERDEF_SLACK_MEMBER_ID="UXXXX"
export USERDEF_WANDB_ENTITY="your-wandb-entity"
export USERDEF_WANDB_PROJECT="flux-kontext"
export USERDEF_WANDB_API_KEY="replace-with-real-key"
export USERDEF_GIT_BRANCH="main"
export USERDEF_GIT_COMMIT="none"        # use "none" to keep HEAD of branch
export USERDEF_EXP_ROOT="/sensei-fs-3/users/your_user/flux_kontext_logs"
export USERDEF_EXP_NAME="flux-kontext-multinode"
export USERDEF_S3_EXPERIMENT_ROOT="s3://your-bucket/experiments/flux-kontext"
export USERDEF_LOCAL_OUTPUT_ROOT="/mnt/localssd/flux-kontext"
export USERDEF_METADATA_PATH="/sensei-fs/users/you/path/to/metadata.json"
export USERDEF_S3_BUCKET="s3://path/to/kontext/data"
export USERDEF_BATCH_SIZE="8"
export USERDEF_GRAD_ACCUM="1"
export USERDEF_NUM_EPOCHS="10"
export USERDEF_LEARNING_RATE="1e-5"
export USERDEF_SAVE_STEPS="1000"
export USERDEF_DATASET_REPEAT="1"
export USERDEF_DATA_BATCH_VIZ_STEPS="1000,2000,3000"
export USERDEF_TRAINABLE_MODELS="dit"
export USERDEF_KEEP_CKPT_STEP_DIV_BY=""
export USERDEF_WANDB_RUN_NAME=""

# Uncomment the following if you want MinIO to stage arrow data locally.
# export USERDEF_MINIO_FROM_S3_PATH="s3://bucket/path/to/*.arrow"

################################################################################################
# MinIO staging (optional)
################################################################################################
if [ -n "${USERDEF_MINIO_FROM_S3_PATH:-}" ]; then
  echo "MinIO download enabled: ${USERDEF_MINIO_FROM_S3_PATH}"
  export MINIO_READY_FILE="/mnt/localssd/minio.ready"
  SUBDIR="${USERDEF_EXP_NAME}"
  export MINIO_FULL_TO_LOCAL_DIR="/mnt/localssd/mori-shared/node0_source/${SUBDIR}"
  {
    bash /sensei-fs/users/fengw/start_minio.sh
    m-download "${USERDEF_MINIO_FROM_S3_PATH}" --subdir "${SUBDIR}" > /mnt/localssd/minio_train.log
  } &
  echo "MinIO data download running in background"
fi

################################################################################################
# Foundation Job Butler registration and log syncing
################################################################################################
{
if [ -n "${RUNAI_NUM_OF_GPUS:-}" ]; then
  export NUM_GPUS="${RUNAI_NUM_OF_GPUS}"
else
  export NUM_GPUS="${NUM_OF_GPUS}"
fi
export NUM_NODES="${WORLD_SIZE}"
export NODE_RANK="${RANK}"
export NODE_LAUNCH_TIMESTAMP="$(date -u +"%Y-%m-%dT%H:%M:%S.%3N")+00:00"
export FOUNDATION_JOB_BUTLER_SERVER="http://$((s5cmd --no-sign-request cp s3://dit-scale-up/kaiz/softwares/job-butler-server-url.txt /tmp && cat /tmp/job-butler-server-url.txt) | awk 'END {print}')"
echo "FOUNDATION_JOB_BUTLER_SERVER=${FOUNDATION_JOB_BUTLER_SERVER}"
export FOUNDATION_JOB_BUTLER_WATCH_DIR="/mnt/localssd/foundation_job_butler_watch_dir"
mkdir -p "${FOUNDATION_JOB_BUTLER_WATCH_DIR}"
export MAIN_LOG_FILE="${FOUNDATION_JOB_BUTLER_WATCH_DIR}/train_${NODE_RANK}.log"

export S3_MAIN_LOG_DIR="s3://dit-scale-up/foundation-job-logs/${JOB_UUID}/"
export SSD_TMP_DIR="/mnt/localssd/tmp"
mkdir -p "${SSD_TMP_DIR}"
nohup bash -c 'while true; do s5cmd --no-sign-request sync "'"${FOUNDATION_JOB_BUTLER_WATCH_DIR}/*.log"'" "'"${S3_MAIN_LOG_DIR}"'" >"${SSD_TMP_DIR}/s5cmd_${NODE_RANK}.log" 2>&1; sleep 10; done' &
export LOG_VIEWER_URL="https://dit-scale-up.s3.us-west-2.amazonaws.com/kaiz/softwares/log_viewer/dist/index.html?logdir=/foundation-job-logs/${JOB_UUID}/"
echo "LOG_VIEWER_URL: ${LOG_VIEWER_URL}"

export EMAIL="${USERDEF_EMAIL}"
export SLACK_CHANNEL_ID="${USERDEF_SLACK_CHANNEL_ID}"
export SLACK_MEMBER_ID="${USERDEF_SLACK_MEMBER_ID}"
source /opt/venv/bin/activate
pip install /sensei-fs-3/users/kaiz/github_repos/foundation-job-butler
}
################################################################################################

################################################################################################
# Main execution block
################################################################################################
{
REPO_FOLDER="/mnt/localssd/github_repos/fluxkontext_multinode_train"
SCRIPT_PATH="${REPO_FOLDER}/pluto_main_multinode_accelerate.sh"
GIT_BRANCH="${USERDEF_GIT_BRANCH}"
GIT_COMMIT="${USERDEF_GIT_COMMIT}"
EXP_ROOT="${USERDEF_EXP_ROOT}"
EXP_NAME="${USERDEF_EXP_NAME}"
S3_EXPERIMENT_ROOT="${USERDEF_S3_EXPERIMENT_ROOT}"
LOCAL_OUTPUT_ROOT="${USERDEF_LOCAL_OUTPUT_ROOT}"
WANDB_ENTITY="${USERDEF_WANDB_ENTITY}"
WANDB_PROJECT="${USERDEF_WANDB_PROJECT}"
WANDB_API_KEY="${USERDEF_WANDB_API_KEY}"
WANDB_RUN_NAME="${USERDEF_WANDB_RUN_NAME}"
METADATA_PATH="${USERDEF_METADATA_PATH}"
S3_BUCKET="${USERDEF_S3_BUCKET}"
BATCH_SIZE="${USERDEF_BATCH_SIZE}"
GRAD_ACCUM="${USERDEF_GRAD_ACCUM}"
NUM_EPOCHS="${USERDEF_NUM_EPOCHS}"
LEARNING_RATE="${USERDEF_LEARNING_RATE}"
SAVE_STEPS="${USERDEF_SAVE_STEPS}"
DATASET_REPEAT="${USERDEF_DATASET_REPEAT}"
DATA_BATCH_VIZ_STEPS="${USERDEF_DATA_BATCH_VIZ_STEPS}"
TRAINABLE_MODELS="${USERDEF_TRAINABLE_MODELS}"
KEEP_CKPT_STEP_DIV_BY="${USERDEF_KEEP_CKPT_STEP_DIV_BY}"

mkdir -p "${EXP_ROOT}/${EXP_NAME}"
mkdir -p "${LOCAL_OUTPUT_ROOT}"

# Clone repo onto local SSD for performance
if [[ "${REPO_FOLDER}" == /mnt/localssd/github_repos/* ]]; then
  echo "Preparing repo checkout in ${REPO_FOLDER}"
  mkdir -p /mnt/localssd/github_repos && cd /mnt/localssd/github_repos
  REPO_NAME="fluxkontext_multinode_train"
  if [ ! -d "${REPO_NAME}" ]; then
    GITHUB_TOKEN=$(cat /sensei-fs/projects/firefly/clio/training/autogen/gh_token)
    REPO_URL="https://cliobot_adobe:${GITHUB_TOKEN}@github.com/Adobe-Firefly/${REPO_NAME}.git"
    git clone --single-branch --branch "${GIT_BRANCH}" "${REPO_URL}" "${REPO_NAME}"
  fi
  cd "${REPO_NAME}"
  git fetch origin
  git checkout "${GIT_BRANCH}"
  if [ "${GIT_COMMIT}" != "none" ]; then
    git checkout "${GIT_COMMIT}"
  fi
else
  cd "${REPO_FOLDER}"
fi

# Export overrides consumed by pluto_main_multinode_accelerate.sh
export EXPERIMENT_NAME="${EXP_NAME}"
export LOG_DIR="${EXP_ROOT}/${EXP_NAME}/logs"
export S3_EXPERIMENT_FOLDER="${S3_EXPERIMENT_ROOT}/${EXP_NAME}"
export OUTPUT_PATH="${LOCAL_OUTPUT_ROOT}/${EXP_NAME}"
export S3_CHECKPOINT_PATH="${S3_EXPERIMENT_FOLDER}/checkpoints"
export WANDB_API_KEY
export WANDB_ENTITY
export WANDB_PROJECT
if [ -n "${WANDB_RUN_NAME}" ]; then
  export WANDB_RUN_NAME
fi
export METADATA_PATH
export S3_BUCKET
export BATCH_SIZE="${BATCH_SIZE}"
export GRADIENT_ACCUM="${GRAD_ACCUM}"
export NUM_EPOCHS="${NUM_EPOCHS}"
export LEARNING_RATE
export SAVE_STEPS
export DATASET_REPEAT
export DATA_BATCH_VIZ_STEPS
export TRAINABLE_MODELS
if [ -n "${KEEP_CKPT_STEP_DIV_BY}" ]; then
  export KEEP_CKPT_STEP_DIV_BY
fi

chmod +x "${SCRIPT_PATH}"

# Launch training and tee logs so Job Butler picks them up
cd "${REPO_FOLDER}"
bash "${SCRIPT_PATH}"

} 2>&1 | tee "${MAIN_LOG_FILE}"

sleep infinity
