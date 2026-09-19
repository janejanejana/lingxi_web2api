from __future__ import annotations

from dataclasses import dataclass

# Limits and allowed types as specified by 灵犀's own stated support:
# docs up to 150MB, images up to 10MB.
ALLOWED_DOC_EXTENSIONS = {
    "txt", "doc", "docx", "pdf", "epub", "md",
    "ppt", "pptx", "xls", "xlsx", "csv",
}
ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "webp", "bmp"}

MAX_DOC_BYTES = 150 * 1024 * 1024
MAX_IMAGE_BYTES = 10 * 1024 * 1024

# attachmentTypeList value — confirmed from real captures across
# multiple file types: 3 = image (jpg/png), 5 = every document type
# tried (txt/doc/pdf/epub/pptx) — looks like docs all share one code
# rather than getting distinct values per format.
ATTACHMENT_TYPE_IMAGE = 3
ATTACHMENT_TYPE_DOC = 5

# dialogueInput.command — confirmed shape from a real image-attachment
# capture. Whether doc attachments use the same command/subCommand is
# UNCONFIRMED.
ATTACHMENT_COMMAND = {"command": "036", "subCommand": "036006"}


class AttachmentError(ValueError):
    pass


def classify_extension(filename: str) -> str:
    """Returns 'doc', 'image', or raises AttachmentError for an
    unsupported extension."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext in ALLOWED_IMAGE_EXTENSIONS:
        return "image"
    if ext in ALLOWED_DOC_EXTENSIONS:
        return "doc"
    raise AttachmentError(
        f"unsupported attachment type: .{ext or '(none)'} — allowed: "
        f"{sorted(ALLOWED_DOC_EXTENSIONS | ALLOWED_IMAGE_EXTENSIONS)}"
    )


def validate_attachment(filename: str, size_bytes: int) -> None:
    """Raises AttachmentError if filename/size_bytes don't meet 灵犀's
    stated limits. Call this BEFORE attempting any upload."""
    kind = classify_extension(filename)
    limit = MAX_IMAGE_BYTES if kind == "image" else MAX_DOC_BYTES
    if size_bytes > limit:
        raise AttachmentError(
            f"{filename} is {size_bytes} bytes, which exceeds the "
            f"{limit} byte limit for {kind} attachments"
        )


@dataclass
class UploadedFile:
    """What a successful upload gives back — matches the fileList entry
    shape seen in a real capture (dialogueInput.attachment.fileList[0])."""
    file_id: str
    parent_file_id: str
    name: str
    size: int
    type: str = "file"


def build_attachment_field(files: list[UploadedFile]) -> dict:
    """Builds the dialogueInput.attachment object from already-uploaded
    files. Matches this real capture exactly:

        "attachment": {
          "attachmentTypeList": [3],
          "fileList": [{"fileId": "...", "parentFileId": "...",
                         "name": "...", "type": "file", "size": ...}]
        }

    attachmentTypeList uses ATTACHMENT_TYPE_IMAGE for every file here if
    any image is present — the real multi-file / mixed-type shape is
    unconfirmed (only a single-image capture exists so far).
    """
    if not files:
        return {}
    kinds = {classify_extension(f.name) for f in files}
    type_list = [ATTACHMENT_TYPE_IMAGE if "image" in kinds else ATTACHMENT_TYPE_DOC]
    return {
        "attachmentTypeList": type_list,
        "fileList": [
            {
                "fileId": f.file_id,
                "parentFileId": f.parent_file_id,
                "name": f.name,
                "type": f.type,
                "size": f.size,
            }
            for f in files
        ],
    }


# --- NOT YET IMPLEMENTED ---
# The upload step itself — turning raw file bytes into a fileId/
# parentFileId pair — is still unknown. Every attachment-bearing request
# captured so far already has the file uploaded beforehand; the upload
# call itself (likely a separate endpoint on the same 139 cloud-storage
# backend, given the fileId/parentFileId shape matches generated-image
# results too) hasn't been captured yet.
#
# To finish this: in the web/app UI, click the attach button and watch
# Network BEFORE hitting send — the upload almost certainly happens the
# moment the file is selected/previewed, not at send time. Capture that
# request (URL, headers, body — likely multipart/form-data) and share it;
# then implement:
#
#   async def upload_attachment(file_bytes: bytes, filename: str, client_cfg) -> UploadedFile:
#       raise NotImplementedError(...)
#
# and call it from server.py before build_attachment_field().

