from pydantic import BaseModel, HttpUrl
from schemas.common import Repository, User
from typing import Optional


class Branch(BaseModel):
    label: str
    ref: str
    sha: str
    user: User
    repo: Repository


class PullRequest(BaseModel):
    url: HttpUrl
    issue_url: HttpUrl
    number: int
    state: str
    title: str
    user: User
    body: Optional[str]
    created_at: str
    updated_at: str
    closed_at: Optional[str]
    merged_at: Optional[str]
    merge_commit_sha: Optional[str]
    commits: int
    additions: int
    deletions: int
    changed_files: int
    head: Branch
    base: Branch


class PullRequestSchema(BaseModel):
    action: str
    number: int
    pull_request: PullRequest
    repository: Repository
    sender: User
