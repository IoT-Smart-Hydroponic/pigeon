from pydantic import BaseModel, HttpUrl
from schemas.common import User, Repository, Account


class HeadCommit(BaseModel):
    id: str
    message: str
    timestamp: str
    author: Account
    committer: Account


class WorkflowRun(BaseModel):
    id: int
    name: str
    head_branch: str
    head_sha: str
    path: str
    display_title: str
    run_number: int
    event: str
    status: str
    conclusion: str | None
    html_url: HttpUrl
    created_at: str
    updated_at: str
    triggering_actor: User
    head_commit: HeadCommit
    actor: User


class Workflow(BaseModel):
    id: int
    name: str
    state: str


class WorkflowSchema(BaseModel):
    action: str
    workflow_run: WorkflowRun
    workflow: Workflow
    repository: Repository
    sender: User
