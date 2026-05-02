#!/usr/bin/env bash
# backup_mongodb.sh — Automated mongodump backup with rotation for thumalien_db
set -euo pipefail

# ── Configuration ────────────────────────────────────────────────
MONGO_HOST="${MONGO_HOST:-mongodb}"
MONGO_PORT="${MONGO_PORT:-27017}"
MONGO_USER="${MONGO_USER:?MONGO_USER is required}"
MONGO_PASSWORD="${MONGO_PASSWORD:?MONGO_PASSWORD is required}"
MONGO_DB="${MONGO_DB:-thumalien_db}"
BACKUP_DIR="${BACKUP_DIR:-/backups}"
RETENTION="${BACKUP_RETENTION:-7}"

# ── Derived variables ────────────────────────────────────────────
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
DUMP_DIR="${BACKUP_DIR}/${MONGO_DB}_${TIMESTAMP}"
ARCHIVE="${DUMP_DIR}.gz"

# ── Ensure backup directory exists ───────────────────────────────
mkdir -p "${BACKUP_DIR}"

# ── Run mongodump ────────────────────────────────────────────────
echo "[$(date --iso-8601=seconds 2>/dev/null || date '+%Y-%m-%dT%H:%M:%S')] Starting backup of ${MONGO_DB}..."

mongodump \
  --host="${MONGO_HOST}" \
  --port="${MONGO_PORT}" \
  --username="${MONGO_USER}" \
  --password="${MONGO_PASSWORD}" \
  --authenticationDatabase=admin \
  --db="${MONGO_DB}" \
  --gzip \
  --archive="${ARCHIVE}"

echo "[$(date --iso-8601=seconds 2>/dev/null || date '+%Y-%m-%dT%H:%M:%S')] Backup written to ${ARCHIVE}"

# ── Rotation: keep only the last N backups ───────────────────────
BACKUP_COUNT=$(ls -1t "${BACKUP_DIR}"/${MONGO_DB}_*.gz 2>/dev/null | wc -l)

if [ "${BACKUP_COUNT}" -gt "${RETENTION}" ]; then
  ls -1t "${BACKUP_DIR}"/${MONGO_DB}_*.gz | tail -n +"$((RETENTION + 1))" | while read -r old; do
    echo "[$(date --iso-8601=seconds 2>/dev/null || date '+%Y-%m-%dT%H:%M:%S')] Removing old backup: ${old}"
    rm -f "${old}"
  done
fi

echo "[$(date --iso-8601=seconds 2>/dev/null || date '+%Y-%m-%dT%H:%M:%S')] Backup complete. ${BACKUP_COUNT} archive(s) present, retention=${RETENTION}."
