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


def _progress(label, done, total, unit="MB"):
    """Print a single updating progress line (\\r, no newline) -- call once
    per chunk/item, then print() a bare newline when done. Every long
    transfer in this module uses this so a large file in progress looks
    different from a hung cell, instead of going silent until it finishes.
    """
    pct = 100 * done / total if total else 100
    print(f"\r  {label}: {pct:5.1f}% ({done:.0f}/{total:.0f} {unit})", end="", flush=True)


def _copy_with_progress(src, dst, label, chunk_size=64 * 1024 * 1024):
    """shutil.copy, but prints a progress line -- shutil.copy itself gives
    no feedback until an entire multi-GB file is done."""
    total = os.path.getsize(src) / 1e6
    copied = 0.0
    with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
        while True:
            chunk = fsrc.read(chunk_size)
            if not chunk:
                break
            fdst.write(chunk)
            copied += len(chunk) / 1e6
            _progress(label, copied, total)
    print()

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
    for i, name in enumerate(filenames, 1):
        src = os.path.join(drive_cache_dir, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(local_dir, name))
            restored.append(name)
        if len(filenames) > 1:
            _progress(f"restoring from {os.path.basename(drive_cache_dir)}", i, len(filenames), unit="files")
    if len(filenames) > 1:
        print()
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
    for i, name in enumerate(filenames, 1):
        src = os.path.join(local_dir, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(drive_cache_dir, name))
            uploaded.append(name)
        if len(filenames) > 1:
            _progress(f"caching to {os.path.basename(drive_cache_dir)}", i, len(filenames), unit="files")
    if len(filenames) > 1:
        print()
    return uploaded


# ---------------------------------------------------------------------------
# Raw-archive caching: cache_upload/cache_restore above copy one file at a
# time, which is fine for a handful of small outputs but is exactly the slow
# path for era5/completos's 195 loose .grib files -- many small files over a
# mounted Drive is the specific bottleneck reported in
# https://www.reddit.com/r/MachineLearning/comments/sxui5r/ (an hour just to
# glob a Drive folder; OP's own fix was zip-once, download-the-single-file).
# These two functions apply that fix to the raw grib fetch specifically:
# zip a local directory's files into one archive (fast -- local disk, no
# network-mount overhead) and cache that single file, instead of caching each
# .grib individually.
# ---------------------------------------------------------------------------

def archive_and_cache_raw(local_dir, filenames, drive_cache_dir, archive_name="era5_completos.zip", mount_point=DEFAULT_MOUNT):
    """Zip filenames from local_dir into one archive and upload it to drive_cache_dir.

    ZIP_STORED (no compression), not ZIP_DEFLATED: GRIB2 already applies its
    own internal packing, so further DEFLATE compression buys little size
    reduction for real CPU cost -- the win here is one file instead of many,
    not smaller bytes. Returns the Drive path of the uploaded archive, or
    None if no listed files were found in local_dir.
    """
    ensure_mounted(mount_point)
    present = [name for name in filenames if os.path.exists(os.path.join(local_dir, name))]
    if not present:
        return None
    os.makedirs(drive_cache_dir, exist_ok=True)
    local_archive = os.path.join(local_dir, archive_name)
    with zipfile.ZipFile(local_archive, "w", zipfile.ZIP_STORED) as zf:
        for i, name in enumerate(present, 1):
            zf.write(os.path.join(local_dir, name), arcname=name)
            _progress(f"zipping {archive_name}", i, len(present), unit="files")
    print()
    dst = os.path.join(drive_cache_dir, archive_name)
    _copy_with_progress(local_archive, dst, f"uploading {archive_name} (mounted)")
    os.remove(local_archive)  # don't leave a duplicate multi-GB file locally
    return dst


def restore_raw_archive(local_dir, expected_filenames, drive_cache_dir, archive_name="era5_completos.zip", mount_point=DEFAULT_MOUNT):
    """Restore local_dir from a cached single-zip archive, if it covers expected_filenames.

    Checks the archive's directory listing (fast -- metadata only, no
    decompression) against expected_filenames before extracting anything.
    Returns the list of filenames the archive actually covers (a subset of
    expected_filenames) if it covers ALL of them and extraction happened, or
    an empty list if the archive is missing/incomplete -- callers should
    treat an empty list as a cache miss and fall back to the normal
    per-file sync, same convention as cache_restore().
    """
    ensure_mounted(mount_point)
    src = os.path.join(drive_cache_dir, archive_name)
    if not os.path.exists(src):
        return []
    with zipfile.ZipFile(src) as zf:
        # metadata only (reads the central directory, not file bodies) -- cheap
        # even over the mount, so check completeness before touching any bytes
        available = set(zf.namelist())
        if not set(expected_filenames).issubset(available):
            return []
        os.makedirs(local_dir, exist_ok=True)
        for i, name in enumerate(expected_filenames, 1):
            zf.extract(name, local_dir)
            _progress(f"extracting {archive_name}", i, len(expected_filenames), unit="files")
    print()
    return list(expected_filenames)


# ---------------------------------------------------------------------------
# Drive-API fast path: everything above reads/writes through drive.mount()'s
# FUSE-mounted filesystem, which has real per-read/per-write overhead for
# large sequential I/O separate from the "many small files" problem the
# archive functions above already solve -- widely reported for Colab
# specifically. These functions use the Drive API directly instead
# (google.colab.auth + googleapiclient, both preinstalled in Colab), which
# streams the file over HTTP rather than through the mount. Stays private:
# authenticates as whichever Google account is already logged into the
# Colab session, no link-sharing needed (a plain public-link+gdown
# download was tried early in this project for the shared "2. Datos"
# folder and returned HTTP 401, since it isn't publicly shared -- see the
# module docstring; the API path works the same way regardless of sharing
# settings, using the account's own permissions instead).
#
# UNVERIFIED beyond code review: there is no Google Drive access from
# outside Colab to test any of this against -- confirm the actual speedup,
# and that it works at all in a given Colab environment, there. Every
# function below falls back to the already-verified mount-based read/write
# above on any failure (missing library, auth issue, folder not found,
# quota, etc.), so this can only add speed, never remove functionality.
# ---------------------------------------------------------------------------

def _mounted_to_api_path(drive_dir, mount_point=DEFAULT_MOUNT):
    """Convert a '/content/drive/MyDrive/a/b' mounted path to 'a/b' for the API.

    Accepts an already-relative path unchanged, so callers can pass either
    style. Path components are matched against Drive folder names via the
    API, starting from the account's own Drive root -- unrelated to how the
    path looks on the mounted filesystem, just a convenient shared way to
    name the same location that the rest of this module already uses.

    Deliberately POSIX-only (hardcoded "/", not os.path.join/os.sep): the
    mounted path is always a Colab (Linux) path regardless of what platform
    this function itself runs on, e.g. under a local test on Windows.
    """
    prefix = mount_point.rstrip("/") + "/MyDrive/"
    if drive_dir.startswith(prefix):
        drive_dir = drive_dir[len(prefix):]
    return drive_dir.strip("/")


def _drive_api_service():
    from google.colab import auth
    from googleapiclient.discovery import build

    auth.authenticate_user()
    return build("drive", "v3")


def _resolve_folder_id(service, api_path, create=False):
    """Walk 'a/b/c' one component at a time from the account's Drive root.

    With create=False (read path): returns None as soon as a component
    isn't found -- the caller should treat that as "nothing cached yet",
    not an error. With create=True (write path): creates any missing
    component so the destination always exists.
    """
    parent = "root"
    for part in [p for p in api_path.split("/") if p]:
        query = f"'{parent}' in parents and name = '{part}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        found = service.files().list(q=query, fields="files(id)").execute().get("files", [])
        if found:
            parent = found[0]["id"]
        elif create:
            meta = {"name": part, "mimeType": "application/vnd.google-apps.folder", "parents": [parent]}
            parent = service.files().create(body=meta, fields="id").execute()["id"]
        else:
            return None
    return parent


def download_via_api(drive_dir, filename, dest_path):
    """Download filename from a Drive folder via the API. Returns True if
    downloaded, False if the folder or file wasn't found (a normal cache
    miss, not an error -- raises only on an actual API/auth failure, which
    callers should treat as "try the mount-based path instead").
    """
    from googleapiclient.http import MediaIoBaseDownload

    service = _drive_api_service()
    folder_id = _resolve_folder_id(service, _mounted_to_api_path(drive_dir))
    if folder_id is None:
        return False

    query = f"'{folder_id}' in parents and name = '{filename}' and trashed = false"
    found = service.files().list(q=query, fields="files(id, size)").execute().get("files", [])
    if not found:
        return False
    size_mb = int(found[0].get("size") or 0) / 1e6

    request = service.files().get_media(fileId=found[0]["id"])
    with open(dest_path, "wb") as f:
        downloader = MediaIoBaseDownload(f, request, chunksize=100 * 1024 * 1024)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                _progress(f"downloading {filename} (API)", status.progress() * size_mb, size_mb)
    print()
    return True


def upload_via_api(local_path, drive_dir, filename=None):
    """Upload local_path to a Drive folder via the API, creating the folder
    path if needed. Replaces (deletes then re-creates) any existing file of
    the same name rather than leaving duplicate copies. Returns the
    uploaded file's Drive ID.
    """
    from googleapiclient.http import MediaFileUpload

    service = _drive_api_service()
    folder_id = _resolve_folder_id(service, _mounted_to_api_path(drive_dir), create=True)
    filename = filename or os.path.basename(local_path)

    query = f"'{folder_id}' in parents and name = '{filename}' and trashed = false"
    for existing in service.files().list(q=query, fields="files(id)").execute().get("files", []):
        service.files().delete(fileId=existing["id"]).execute()

    size_mb = os.path.getsize(local_path) / 1e6
    media = MediaFileUpload(local_path, resumable=True, chunksize=100 * 1024 * 1024)
    request = service.files().create(body={"name": filename, "parents": [folder_id]}, media_body=media, fields="id")
    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            _progress(f"uploading {filename} (API)", status.progress() * size_mb, size_mb)
    print()
    return response.get("id")


def restore_raw_archive_fast(local_dir, expected_filenames, drive_cache_dir, archive_name="era5_completos.zip",
                              staging_dir="/content/_drive_api_staging", mount_point=DEFAULT_MOUNT):
    """Same contract as restore_raw_archive() (empty list = miss), tries the
    Drive-API download first and falls back to the mount-based read for
    anything the API path doesn't resolve cleanly -- an API failure, the
    folder/file not found, or an incomplete archive.
    """
    try:
        os.makedirs(staging_dir, exist_ok=True)
        staged_path = os.path.join(staging_dir, archive_name)
        try:
            if download_via_api(drive_cache_dir, archive_name, staged_path):
                with zipfile.ZipFile(staged_path) as zf:
                    available = set(zf.namelist())
                    if set(expected_filenames).issubset(available):
                        os.makedirs(local_dir, exist_ok=True)
                        for i, name in enumerate(expected_filenames, 1):
                            zf.extract(name, local_dir)
                            _progress(f"extracting {archive_name}", i, len(expected_filenames), unit="files")
                        print()
                        return list(expected_filenames)
        finally:
            if os.path.exists(staged_path):
                os.remove(staged_path)
    except Exception as e:
        print(f"  (Drive-API download failed ({e}) -- falling back to mounted read)")
    return restore_raw_archive(local_dir, expected_filenames, drive_cache_dir, archive_name, mount_point)


def archive_and_cache_raw_fast(local_dir, filenames, drive_cache_dir, archive_name="era5_completos.zip", mount_point=DEFAULT_MOUNT):
    """Same contract as archive_and_cache_raw() (returns the archive's
    destination identifier, or None if nothing to archive), but uploads via
    the Drive API instead of shutil.copy through the mount. Still builds the
    zip locally first (same reasoning as archive_and_cache_raw -- local disk
    has no per-file mount overhead), only the upload step differs. Falls
    back to the mount-based upload on any API failure.
    """
    ensure_mounted(mount_point)
    present = [name for name in filenames if os.path.exists(os.path.join(local_dir, name))]
    if not present:
        return None
    local_archive = os.path.join(local_dir, archive_name)
    with zipfile.ZipFile(local_archive, "w", zipfile.ZIP_STORED) as zf:
        for i, name in enumerate(present, 1):
            zf.write(os.path.join(local_dir, name), arcname=name)
            _progress(f"zipping {archive_name}", i, len(present), unit="files")
    print()
    try:
        file_id = upload_via_api(local_archive, drive_cache_dir, archive_name)
        os.remove(local_archive)
        return file_id
    except Exception as e:
        print(f"  (Drive-API upload failed ({e}) -- falling back to mounted write)")
        os.makedirs(drive_cache_dir, exist_ok=True)
        dst = os.path.join(drive_cache_dir, archive_name)
        _copy_with_progress(local_archive, dst, f"uploading {archive_name} (mounted)")
        os.remove(local_archive)
        return dst
