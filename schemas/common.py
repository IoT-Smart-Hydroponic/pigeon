from pydantic import BaseModel, HttpUrl
from typing import Optional


class Account(BaseModel):
    name: str
    email: Optional[str] = None


class Commit(BaseModel):
    id: str
    message: str
    url: HttpUrl
    author: Account
    committer: Account
    timestamp: str


class User(BaseModel):
    login: str
    id: int
    avatar_url: HttpUrl
    html_url: HttpUrl


class Repository(BaseModel):
    name: str
    full_name: str
    private: bool
    html_url: HttpUrl
    owner: User
    homepage: Optional[HttpUrl] = None
