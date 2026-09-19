from __future__ import annotations

import hashlib
import logging
import uuid

import httpx

from .attachments import UploadedFile
from .config import ClientCfg

logger = logging.getLogger("yun139.upload")

CLOUD_BASE_URL = "https://personal-kd-njs.yun.139.com"

# Fixed folder path every attachment gets filed under, mirroring the real
# app's own behavior exactly: AI空间 / 灵犀互动记录 / 本地上传.
FOLDER_PATH = ["AI空间", "灵犀互动记录", "本地上传"]

# In-memory cache: client_cfg.id -> resolved "本地上传" folder fileId.
# Folder creation is idempotent server-side (repeat calls just return
# exist:true) so this cache is purely to save 3 round trips per upload,
# not needed for correctness.
_folder_cache: dict[str, str] = {}

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/152.0.0.0 Safari/537.36"
)
_CLIENT_INFO = "||||chrome|152.0.0.0|||windows 10||zh-CN|||"


class UploadError(RuntimeError):
    pass


def _cloud_headers(client_cfg: ClientCfg) -> dict:
    return {
        "Host": "personal-kd-njs.yun.139.com",
        "x-NetType": "4",
        "sec-ch-ua-platform": '"Windows"',
        "Authorization": client_cfg.authorization,
        "x-yun-client-info": _CLIENT_INFO,
        "mcloud-client": "10904",
        "mcloud-version": "2.0.0",
        "x-inner-ntwk": "2",
        "sec-ch-ua": '"Chromium";v="152", "Not?A_Brand";v="24", "Google Chrome";v="152"',
        "mcloud-channel": "1000101",
        "sec-ch-ua-mobile": "?0",
        "x-yun-api-version": "v1",
        "mcloud-skey": "",
        "Accept": "application/json, text/plain, */*",
        "Content-Type": "application/json;charset=UTF-8",
        "x-yun-tid": str(uuid.uuid4()),
        "x-huawei-channelSrc": "10254500",
        "mcloud-network": "102",
        "x-DeviceInfo": _CLIENT_INFO,
        "INNER-HCY-ROUTER-HTTPS": "1",
        "x-yun-app-channel": client_cfg.source_channel,
        "User-Agent": _USER_AGENT,
        "DNT": "1",
        "Origin": "https://appmail.mail.10086.cn",
        "Sec-Fetch-Site": "cross-site",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Dest": "empty",
        "Referer": "https://appmail.mail.10086.cn/",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,zh-TW;q=0.7",
    }


async def _create_folder(
    client: httpx.AsyncClient, client_cfg: ClientCfg, parent_file_id: str, name: str
) -> str:
    resp = await client.post(
        f"{CLOUD_BASE_URL}/hcy/file/create",
        headers=_cloud_headers(client_cfg),
        json={
            "parentFileId": parent_file_id,
            "name": name,
            "type": "folder",
            "fileRenameMode": "refuse",
        },
    )
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success"):
        raise UploadError(f"folder create failed for {name!r}: {data}")
    return data["data"]["fileId"]


async def _ensure_upload_folder(client: httpx.AsyncClient, client_cfg: ClientCfg) -> str:
    cached = _folder_cache.get(client_cfg.id)
    if cached:
        return cached
    parent = "/"
    for name in FOLDER_PATH:
        parent = await _create_folder(client, client_cfg, parent, name)
    _folder_cache[client_cfg.id] = parent
    logger.info("resolved upload folder for %s -> %s", client_cfg.id, parent)
    return parent


async def upload_attachment(
    file_bytes: bytes,
    filename: str,
    content_type: str,
    client_cfg: ClientCfg,
) -> UploadedFile:
    """Full upload flow, matching the real capture step for step:
    ensure folder path -> create file entry -> PUT bytes to the presigned
    URL -> complete. Raises UploadError on any failure."""
    content_hash = hashlib.sha256(file_bytes).hexdigest()
    size = len(file_bytes)

    async with httpx.AsyncClient(timeout=60) as client:
        parent_file_id = await _ensure_upload_folder(client, client_cfg)

        create_resp = await client.post(
            f"{CLOUD_BASE_URL}/hcy/file/create",
            headers=_cloud_headers(client_cfg),
            json={
                "fileRenameMode": "auto_rename",
                "contentType": content_type or "application/oct-stream",
                "type": "file",
                "name": filename,
                "size": size,
                "contentHashAlgorithm": "sha256",
                "contentHash": content_hash,
                "partInfos": [
                    {"parallelHashCtx": {"partOffset": 0}, "partNumber": 1, "partSize": size}
                ],
                "parentFileId": parent_file_id,
            },
        )
        create_resp.raise_for_status()
        create_data = create_resp.json()
        if not create_data.get("success"):
            raise UploadError(f"file create failed: {create_data}")
        d = create_data["data"]
        file_id = d["fileId"]
    rapid_upload = bool(d.get("rapidUpload"))
    part_infos = d.get("partInfos") or []

    if rapid_upload or not part_infos:
        logger.info(
            "file create reported rapidUpload=%s or no partInfos; treating %s as already-uploaded fileId=%s",
            rapid_upload, filename, file_id,
        )
        return UploadedFile(file_id=file_id, parent_file_id=parent_file_id, name=filename, size=size)

    upload_id = d.get("uploadId")
    try:
        upload_url = part_infos[0]["uploadUrl"]
    except (KeyError, IndexError, TypeError) as e:
        raise UploadError(f"file create response missing uploadUrl: {create_data}") from e

    # PUT the raw bytes to the presigned URL. The signature covers
    # content-length + host; httpx sets both automatically from the
    # request, no extra headers needed.
    put_resp = await client.put(upload_url, content=file_bytes)
    put_resp.raise_for_status()

    complete_resp = await client.post(
        f"{CLOUD_BASE_URL}/hcy/file/complete",
        headers=_cloud_headers(client_cfg),
        json={
            "fileId": file_id,
            "uploadId": upload_id,
            "contentHash": content_hash,
            "contentHashAlgorithm": "sha256",
        },
    )
    complete_resp.raise_for_status()
    complete_data = complete_resp.json()
    if not complete_data.get("success"):
        raise UploadError(f"file complete failed: {complete_data}")
    return UploadedFile(file_id=file_id, parent_file_id=parent_file_id, name=filename, size=size)
