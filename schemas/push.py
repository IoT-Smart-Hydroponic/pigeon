from pydantic import BaseModel, HttpUrl
from schemas.common import Commit, Repository, Account, User


class PushSchema(BaseModel):
    ref: str
    before: str
    after: str
    repository: Repository
    pusher: Account
    sender: User
    created: bool
    deleted: bool
    forced: bool
    compare: HttpUrl
    commits: list[Commit]
    head_commit: Commit
