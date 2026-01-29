from pydantic import BaseModel, HttpUrl
from schemas.common import Repository, User


class Release(BaseModel):
    url: HttpUrl
    html_url: HttpUrl
    id: int
    author: User
    tag_name: str
    name: str | None
    draft: bool
    prerelease: bool
    created_at: str
    updated_at: str
    published_at: str | None
    body: str | None


class ReleaseSchema(BaseModel):
    action: str
    release: Release
    repository: Repository
    sender: User
