"""Sync data from a mounted Google Drive folder into this repo's data/ layout.

Colab-only. The shared Drive folder that backs this repo's data/raw/ and
data/processed/ (confirmed path: "My Drive/2. Datos" -- it's a shortcut
sitting directly in the user's My Drive root; the "Indice Climatico
Actuarial" name seen in Drive's breadcrumb/location panel is the owner's
canonical path, not where the shortcut lives) is shared with specific
named collaborators, not "anyone with the link" -- confirmed from the
Drive UI's "Who has access" panel, which also explains why anonymous
access (tested with gdown) returned HTTP 401. The reliable path is to
mount Drive as a logged-in user who actually has access
(google.colab.drive) and copy/symlink from there.

    from google.colab import drive
    drive.mount("/content/drive")

    import drive_sync
    drive_sync.sync(
        drive_root="/content/drive/MyDrive/2. Datos",
        repo_root="/content/aca_indice_climatico_opt",
    )

Confirmed full structure of "2. Datos" (2026-08-10, via
notebooks/explore_drive_structure.ipynb in the aca_aci_collab repo):

    shapefiles/            48 files, flat -- matches data/shapefiles/
    era5/
      completos/           195 files, flat -- era5_{rain,tmp,wind}_<year>.grib,
                            i.e. the actual raw archive data/raw/era5/ expects
      union/                3 files -- era5_{rain,tmp,wind}_union.nc (final
                            merged daily files, but named differently from
                            what calcular_percentil_*.py hardcodes -- would
                            need renaming, not just copying, to be used directly)
      Combined/             dated snapshots (202412/, 202506/, ...) of the
                            same 3 merged files, named era5_daily_combined_*.nc
                            (matches local naming, but which date is "current"
                            is a judgment call, not automatable)
      percentiles_nc/        dated snapshots (202412/, 202506/, ...) of
                            era5_*_percentil.nc (Stage 2 baseline output --
                            names match data/processed/ exactly)
      otros/, ejemplo_4_73/, "Subsets para excel /"  -- one-off/example
                            files, not part of the steady pipeline layout
    salidas_colombia/       1 file: salidas_colombia.zip (not a flat folder!)
    salidas_cundinamarca_bogota/  1 file: salidas_cundinamarca_bogota.zip
    datos_indice/           dated personal archive (2024-12/, 2025-06/,
                            2025-12/) of anomalias_<region> combined CSVs --
                            does not map 1:1 onto any single local directory,
                            deliberately left out of DEFAULT_MAPPING

Only shapefiles/, era5/completos/, and the two salidas_*.zip are mapped by
default -- the rest (union/, Combined/, percentiles_nc/, datos_indice/) are
dated snapshots or use different filenames than the pipeline scripts
expect, so pulling them in requires a judgment call (which date, rename to
what) that sync() deliberately does not make for you. Pass your own
`mapping=` for those.
"""
import os
import shutil
import zipfile

DEFAULT_MOUNT = "/content/drive"

# Drive subfolder (relative to drive_root) -> local path (relative to repo_root).
# See the module docstring for the full confirmed structure and why the
# other subfolders (union/, Combined/, percentiles_nc/, datos_indice/)
# aren't included here.
DEFAULT_MAPPING = {
    "shapefiles": "data/shapefiles",
    "era5/completos": "data/raw/era5",
    "salidas_colombia": "data/processed/anomalias_colombia",
    "salidas_cundinamarca_bogota": "data/processed/anomalias_cundinamarca_bogota",
}


def ensure_mounted(mount_point=DEFAULT_MOUNT):
    """Mount Google Drive if it isn't already. No-op outside Colab-with-drive-unmounted."""
    if not os.path.isdir(mount_point):
        from google.colab import drive
        drive.mount(mount_point)
    return mount_point


def sync(drive_root, repo_root, mapping=None, mount_point=DEFAULT_MOUNT, copy=True, only=None, extract_zips=True):
    """Copy (or symlink) each configured Drive subfolder into its local data/ path.

    drive_root: path to the shared folder inside the mounted Drive, e.g.
        "/content/drive/MyDrive/2. Datos" -- wherever that folder (or a
        shortcut to it) actually sits in your Drive.
    repo_root: local checkout of the aca_indice_climatico_opt pipeline repo.
    mapping: overrides DEFAULT_MAPPING.
    copy: True copies files (safe -- local runs can't mutate the shared Drive
        folder); False symlinks instead (fast, no duplication, but writes from
        the pipeline would land back in Drive -- only use for read-only work,
        and only for non-zip sources, since symlinking a .zip doesn't extract it).
    only: iterable of mapping keys to sync, e.g. ["era5/completos"], instead
        of everything.
    extract_zips: if the Drive source folder contains one or more .zip files
        (e.g. salidas_colombia/salidas_colombia.zip), extract them into the
        local destination instead of copying the .zip itself. True by default.

    Returns {drive_subfolder: local_path} for whatever was actually synced.
    """
    ensure_mounted(mount_point)
    mapping = mapping or DEFAULT_MAPPING
    if only is not None:
        mapping = {k: v for k, v in mapping.items() if k in only}

    synced = {}
    for drive_sub, local_rel in mapping.items():
        src = os.path.join(drive_root, drive_sub)
        dst = os.path.join(repo_root, local_rel)
        if not os.path.isdir(src):
            print(f"skip {drive_sub!r}: not found under {drive_root}")
            continue

        zip_files = [f for f in os.listdir(src) if f.lower().endswith(".zip")] if extract_zips else []
        if zip_files:
            os.makedirs(dst, exist_ok=True)
            for zf in zip_files:
                with zipfile.ZipFile(os.path.join(src, zf)) as archive:
                    archive.extractall(dst)
                print(f"extracted {drive_sub}/{zf} -> {dst}")
        else:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if copy:
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                if os.path.islink(dst) or os.path.exists(dst):
                    (os.remove if os.path.islink(dst) else shutil.rmtree)(dst)
                os.symlink(src, dst, target_is_directory=True)
            print(f"synced {drive_sub!r} -> {dst}")

        synced[drive_sub] = dst

    return synced


# ---------------------------------------------------------------------------
# Result caching: upload computed outputs to Drive, restore them on a later
# run instead of recomputing. Deliberately file-list-based, not whole-folder
# -- callers name exactly what should be cached (e.g. the Stage 1/2 baseline
# outputs, which are small, stable, and shared across every run regardless
# of which regions/years Stage 3 is scoped to) rather than caching everything
# and risking silently reusing something still-being-debugged (see
# ARCHITECTURE.pdf's open multiprocessing-correctness questions -- Stage 3's
# per-region outputs are deliberately NOT covered by this, only the baseline).
# ---------------------------------------------------------------------------

def cache_complete(filenames, drive_cache_dir):
    """True if every name in filenames already exists in drive_cache_dir."""
    if not os.path.isdir(drive_cache_dir):
        return False
    return all(os.path.exists(os.path.join(drive_cache_dir, name)) for name in filenames)


def cache_restore(local_dir, filenames, drive_cache_dir, mount_point=DEFAULT_MOUNT):
    """Copy filenames from drive_cache_dir down into local_dir, where present.

    Returns the list of filenames actually restored (a partial or empty list
    if the cache is incomplete -- callers should treat that as a cache miss
    and fall back to recomputing, not assume partial results are usable).
    """
    ensure_mounted(mount_point)
    if not os.path.isdir(drive_cache_dir):
        return []
    os.makedirs(local_dir, exist_ok=True)
    restored = []
    for name in filenames:
        src = os.path.join(drive_cache_dir, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(local_dir, name))
            restored.append(name)
    return restored


def cache_upload(local_dir, filenames, drive_cache_dir, mount_point=DEFAULT_MOUNT):
    """Copy filenames from local_dir up into drive_cache_dir (created if needed).

    Returns the list of filenames actually uploaded (files missing from
    local_dir are silently skipped, not an error -- e.g. if only some
    variables finished).
    """
    ensure_mounted(mount_point)
    os.makedirs(drive_cache_dir, exist_ok=True)
    uploaded = []
    for name in filenames:
        src = os.path.join(local_dir, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(drive_cache_dir, name))
            uploaded.append(name)
    return uploaded
