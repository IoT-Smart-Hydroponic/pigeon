import time
from typing import Any, List
from pathlib import Path
import subprocess
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from utils.config import config
from utils.discord import DiscordNotifier
from utils.verify import verify_signature
import logging
import json
import requests
from schemas import (
    PushSchema,
    PullRequestSchema,
    ReleaseSchema,
    WorkflowSchema,
    DockerServiceSchema,
)
from jinja2 import Environment, FileSystemLoader
import re

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
discord_notifier = DiscordNotifier()
app = FastAPI(root="/pigeon")
env_template = Environment(loader=FileSystemLoader("templates"))


def resolve_path(value: str) -> Path | None:
    if not value:
        return None
    return Path(value).expanduser()


PROJECT_PATHS = {
    "smart-hydroponic-backend": resolve_path(config.PATH_BACKEND_HYDROPONIC),
}


def log_error(
    message: str, event: str, delivery: str, exc: Exception | None = None
) -> None:
    if exc is not None:
        logger.error(
            "%s | event=%s delivery=%s", message, event, delivery, exc_info=exc
        )
    else:
        logger.error("%s | event=%s delivery=%s", message, event, delivery)


def get_scope_key(model: Any, fallback: str) -> str:
    repo = getattr(model, "repository", None)
    full_name = getattr(repo, "full_name", None) if repo else None
    return full_name or fallback


def _truncate_output(text: str, limit: int = 1800) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 50] + "\n... (truncated) ..."


def _run_command(cmd: list[str], work_dir: Path) -> str:
    result = subprocess.run(
        cmd,
        cwd=str(work_dir),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    output = result.stdout.strip()
    error = result.stderr.strip()
    combined = "\n".join(line for line in [output, error] if line)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd, output=combined)
    return combined


def _parse_compose_services(raw: str) -> List[DockerServiceSchema]:
    services: List[DockerServiceSchema] = []
    for line in raw.splitlines():
        json_start = line.find("{")
        if json_start == -1:
            continue

        json_part = line[json_start:]
        service_info = json.loads(json_part)
        services.append(DockerServiceSchema(**service_info))

    return services


def _list_compose_services(work_dir: Path) -> List[DockerServiceSchema]:
    raw = _run_command(["docker", "compose", "ps", "--format", "json"], work_dir)
    try:
        return _parse_compose_services(raw)
    except json.JSONDecodeError:
        logger.warning("Failed to parse docker compose ps JSON output")
        raise


def _compose_status_summary(work_dir: Path) -> List[dict[str, Any]] | str:
    max_retries = 12
    interval = 5
    final_status: List[DockerServiceSchema] = []

    for _ in range(max_retries):
        try:
            current_services = _list_compose_services(work_dir)
        except json.JSONDecodeError:
            raw = _run_command(
                ["docker", "compose", "ps", "--format", "json"], work_dir
            )
            return _truncate_output(raw)

        pending_services = [
            s
            for s in current_services
            if "starting" in s.status.lower() or "unhealthy" in s.status.lower()
        ]

        if not pending_services:
            final_status = current_services
            break

        time.sleep(interval)

    if not final_status:
        try:
            final_status = _list_compose_services(work_dir)
        except json.JSONDecodeError:
            raw = _run_command(
                ["docker", "compose", "ps", "--format", "json"], work_dir
            )
            return _truncate_output(raw)

    embeds: list[dict[str, Any]] = []
    for service in final_status:
        embed = {
            "title": f"🐳 Service: {service.name} | ID: {service.id}",
            "color": service.get_color(),
            "fields": [
                {"name": "Status", "value": service.status, "inline": True},
                {"name": "State", "value": service.state, "inline": True},
                {"name": "Image", "value": service.image, "inline": False},
                {"name": "Created", "value": service.createdAt, "inline": False},
            ],
        }
        embeds.append(embed)

    return embeds if embeds else "No containers reported."


def trigger_docker_compose(
    repo_key: str, payload: Any, event: str, delivery: str
) -> None:
    work_dir = PROJECT_PATHS.get(repo_key)
    repo_name = (
        getattr(payload.repository, "name", "")
        if hasattr(payload, "repository")
        else ""
    )
    if not work_dir:
        logger.warning("No Docker compose path configured for repo=%s", repo_key)
        return

    try:
        _run_command(["docker", "compose", "pull"], work_dir)
        up_output = _run_command(["docker", "compose", "up", "-d"], work_dir)
        _run_command(["docker", "image", "prune", "-f"], work_dir)

        status_output = _compose_status_summary(work_dir)
        up_output = up_output.strip()

        payload = {
            "username": "Pigeon Bot",
            "avatar_url": "https://raw.githubusercontent.com/IoT-Smart-Hydroponic/pigeon/refs/heads/main/assets/avatar_besar.jpg",
            "content": (
                f"🧰 Docker Compose Deployment Status for **{repo_name or repo_key}**:\n"
                f"```bash\n{up_output}\n```"
            ),
        }

        if isinstance(status_output, list):
            payload["embeds"] = status_output
        else:
            payload["content"] = payload["content"] + f"\n```\n{status_output}\n```"

        discord_notifier.send_message(
            payload,
            "docker_compose",
            force=True,
            track=False,
            scope_key="docker_compose",
        )
        logger.info("Docker compose triggered for repo=%s", repo_key)
    except subprocess.CalledProcessError as e:
        error_output = getattr(e, "output", "") or str(e)
        error_output = _truncate_output(error_output)
        log_error(f"Docker compose failed: {error_output}", event, delivery, e)
        discord_notifier.send_message(
            {
                "username": "Pigeon Bot",
                "avatar_url": "https://raw.githubusercontent.com/IoT-Smart-Hydroponic/pigeon/refs/heads/main/assets/avatar_besar.jpg",
                "content": (
                    f"❌ **Deploy Failed**\n"
                    f"Repo: **{repo_name or repo_key}**\n"
                    f"```\n{error_output}\n```"
                ),
            },
            "docker_compose",
            force=True,
            track=False,
            scope_key="docker_compose",
        )
    except subprocess.TimeoutExpired as e:
        log_error("Docker compose timed out", event, delivery, e)
        discord_notifier.send_message(
            {
                "username": "Pigeon Bot",
                "avatar_url": "https://raw.githubusercontent.com/IoT-Smart-Hydroponic/pigeon/refs/heads/main/assets/avatar_besar.jpg",
                "content": (
                    f"⏱️ **Deploy Timed Out**\nRepo: **{repo_name or repo_key}**"
                ),
            },
            "docker_compose",
            force=True,
            track=False,
            scope_key="docker_compose",
        )


def should_send_event(event_name: str, model: Any) -> bool:
    if event_name == "pull_request":
        allowed_actions = {
            "opened",
            "reopened",
            "closed",
            "edited",
            "synchronize",
            "ready_for_review",
            "converted_to_draft",
        }
        return model.action in allowed_actions
    if event_name == "release":
        return model.action in {"published", "released", "prereleased"}
    if event_name == "workflow_run":
        return model.action == "completed" and model.workflow_run.conclusion is not None
    if event_name == "push":
        return bool(model.commits or model.head_commit)
    return True


@app.post("/webhook")
async def webhook_handler(request: Request, background_tasks: BackgroundTasks):
    event = request.headers.get("X-GitHub-Event", "unknown")
    delivery = request.headers.get("X-GitHub-Delivery", "unknown")
    secret = request.headers.get("X-Hub-Signature-256", "")

    if not secret:
        logger.warning(
            "Missing X-Hub-Signature-256 header | event=%s delivery=%s", event, delivery
        )
        raise HTTPException(status_code=400, detail="Missing signature header")

    raw_body = await request.body()
    verify_signature(raw_body, secret)

    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError as e:
        log_error("Invalid JSON payload received", event, delivery, e)
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    schema_map = {
        "push": PushSchema,
        "pull_request": PullRequestSchema,
        "release": ReleaseSchema,
        "workflow_run": WorkflowSchema,
    }

    logger.info(f"Received event: {event}, delivery ID: {delivery}")

    schema_cls = schema_map.get(event)
    if schema_cls is None:
        logger.warning(
            "Unsupported event type received: %s | delivery=%s", event, delivery
        )

        filename = Path("data") / f"{event}.json"

        if not filename.parent.exists():
            filename.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(filename, "r", encoding="utf-8") as f:
                try:
                    existing = json.load(f)
                except json.JSONDecodeError:
                    existing = []
        except FileNotFoundError:
            existing = []

        if not isinstance(existing, list):
            existing = [existing]

        existing.append(payload)

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=4)

        if config.WEBHOOK_DEVLOGS_CHANNEL:
            logger.info(f"Stored unsupported event data to {filename}")

            try:
                data = {
                    "username": "Pigeon Bot",
                    "content": f"⚠️ Received unsupported GitHub event: **{event}**\nDelivery ID: `{delivery}`\nStored payload to `{filename}`.",
                }
                discord_notifier.send_message(
                    data, f"devlog:{event}", force=True, track=False, scope_key="devlog"
                )
                logger.info(
                    f"Notification sent to Discord for unsupported event: {event}"
                )
            except requests.exceptions.RequestException as e:
                log_error("Failed to send notification to Discord", event, delivery, e)
        else:
            logger.warning(
                "WEBHOOK_DEVLOGS_CHANNEL is not set. Skipping Discord notification."
            )

        return {"status": "unsupported_event_stored", "event": event}

    try:
        template = env_template.get_template(f"{event}_message.j2")
    except Exception as e:
        log_error(
            f"Template not found/failed to load for event '{event}'", event, delivery, e
        )
        raise HTTPException(
            status_code=500, detail="Template not found or failed to load."
        )

    if not isinstance(payload, list):
        payload = [payload]

    for _, item in enumerate(payload):
        try:
            model = schema_cls(**item)
        except Exception as e:
            log_error(
                f"Failed to validate payload for event '{event}'", event, delivery, e
            )
            raise HTTPException(status_code=500, detail="Payload validation failed.")

        try:
            if not should_send_event(event, model):
                logger.info(
                    "Filtered event '%s' with action '%s'",
                    event,
                    getattr(model, "action", None),
                )
                continue

            scope_key = get_scope_key(model, event)
            message_content = template.render(data=model)

            try:
                message_payload = json.loads(message_content)
            except json.JSONDecodeError as e:
                log_error("Template did not render valid JSON", event, delivery, e)
                raise HTTPException(
                    status_code=500, detail="Template rendered invalid JSON."
                )

            if event == "push":
                discord_notifier.send_message(
                    message_payload, event, scope_key=scope_key
                )
                discord_notifier.remove_id_file(scope_key=scope_key)
            elif event == "pull_request":
                if not discord_notifier.has_active_message(event, scope_key=scope_key):
                    discord_notifier.send_message(
                        message_payload, event, scope_key=scope_key
                    )
                else:
                    discord_notifier.edit_last_message(
                        message_payload, event, scope_key=scope_key
                    )

                if model.action == "closed":
                    discord_notifier.remove_id_file(scope_key=scope_key)
            elif event == "release":
                if not discord_notifier.has_active_message(event, scope_key=scope_key):
                    discord_notifier.send_message(
                        message_payload, event, scope_key=scope_key
                    )
                else:
                    discord_notifier.edit_last_message(
                        message_payload, event, scope_key=scope_key
                    )

                if model.action in {"published", "released", "prereleased"}:
                    discord_notifier.remove_id_file(scope_key=scope_key)
                    repo_key = getattr(model.repository, "name", "")
                    if repo_key and repo_key in PROJECT_PATHS:
                        background_tasks.add_task(
                            trigger_docker_compose,
                            repo_key,
                            model,
                            event,
                            delivery,
                        )
            elif event == "workflow_run":
                if not discord_notifier.has_active_message(event, scope_key=scope_key):
                    discord_notifier.send_message(
                        message_payload, event, scope_key=scope_key
                    )
                else:
                    discord_notifier.edit_last_message(
                        message_payload, event, scope_key=scope_key
                    )

                if model.workflow_run.conclusion in {
                    "success",
                    "failure",
                    "cancelled",
                    "timed_out",
                    "action_required",
                }:
                    discord_notifier.remove_id_file(scope_key=scope_key)
                if model.workflow_run.conclusion == "success":
                    workflow_path = getattr(model.workflow_run, "path", "")
                    repo_name = getattr(model.repository, "name", "")

                    # get (backend or frontend) from path using regex
                    match = re.search(
                        r"(backend|frontend)", workflow_path, re.IGNORECASE
                    )
                    project_type = match.group(1).lower() if match else ""

                    # combine repo name and project type to form key
                    key = f"{repo_name}-{project_type}" if project_type else repo_name
                    if key:
                        logger.info(
                            f"Scheduling Docker compose trigger for repo_key='{key}'"
                        )
                        background_tasks.add_task(
                            trigger_docker_compose,
                            key,
                            model,
                            event,
                            delivery,
                        )
                if (
                    model.workflow_run.name == "Deploy Docs"
                    and model.workflow_run.conclusion == "success"
                ):
                    announce_template = env_template.get_template("announce_docs.j2")
                    version = model.workflow_run.head_commit.message.split(" ")[-1]
                    announce_content = announce_template.render(
                        data=model, version=version
                    )

                    announce_payload = json.loads(announce_content)
                    discord_notifier.send_message(
                        announce_payload, event, scope_key=scope_key
                    )

            logger.info(f"Message sent to Discord for event '{event}' successfully.")
        except requests.exceptions.RequestException as e:
            log_error("Failed to send message to Discord", event, delivery, e)
            raise HTTPException(
                status_code=500, detail="Failed to send message to Discord."
            )


@app.get("/")
async def root():
    return {"message": "Pigeon Bot is running."}


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/serialize/{event}")
async def serialize_workflow(event: str):
    event = event.lower().replace("-", "_")
    with open(f"{event}.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    try:
        match event:
            case "push":
                if isinstance(data, list):
                    return [PushSchema(**item).model_dump() for item in data]
                return PushSchema(**data).model_dump()

            case "pull_request":
                if isinstance(data, list):
                    return [PullRequestSchema(**item).model_dump() for item in data]
                return PullRequestSchema(**data).model_dump()

            case "release":
                if isinstance(data, list):
                    return [ReleaseSchema(**item).model_dump() for item in data]
                return ReleaseSchema(**data).model_dump()

            case "workflow_run":
                if not isinstance(data, list):
                    data = [data]
                for index, item in enumerate(data):
                    data_workflow = WorkflowSchema(**item)
                    data[index] = data_workflow.model_dump()
                return data
            case _:
                raise HTTPException(status_code=400, detail="Unsupported event type")
    except Exception as e:
        logger.error(f"Serialization error: {e}")
        raise HTTPException(status_code=500, detail="Serialization error")


@app.get("/send-message/{event}")
async def send_test_message(event: str, background_tasks: BackgroundTasks):
    event = event.lower().replace("-", "_")

    schema_map = {
        "push": PushSchema,
        "pull_request": PullRequestSchema,
        "release": ReleaseSchema,
        "workflow_run": WorkflowSchema,
    }

    schema_cls = schema_map.get(event)
    if schema_cls is None:
        raise HTTPException(status_code=400, detail="Unsupported event type")

    try:
        template = env_template.get_template(f"{event}_message.j2")
    except Exception as e:
        logger.error(
            "Template not found/failed to load for event '%s'", event, exc_info=e
        )
        raise HTTPException(
            status_code=500, detail="Template not found or failed to load."
        )

    try:
        with open(f"./data/{event}.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"No data file found for event '{event}'."
        )
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in %s.json", event, exc_info=e)
        raise HTTPException(status_code=500, detail="Invalid JSON data file.")

    if not config.WEBHOOK_DEVLOGS_CHANNEL:
        logger.warning("WEBHOOK_DEVLOGS_CHANNEL is not set. Cannot send test message.")
        raise HTTPException(
            status_code=500, detail="WEBHOOK_DEVLOGS_CHANNEL is not configured."
        )

    # Ensure we can iterate even if {event}.json contains a single object
    if not isinstance(data, list):
        data = [data]

    message_ids: list[str] = []

    for idx, payload in enumerate(data):
        try:
            model = schema_cls(**payload)
        except Exception as e:
            logger.error("Failed to validate payload for event '%s'", event, exc_info=e)
            raise HTTPException(status_code=500, detail="Payload validation failed.")

        try:
            if not should_send_event(event, model):
                logger.info(
                    "Filtered test event '%s' with action '%s'",
                    event,
                    getattr(model, "action", None),
                )
                continue

            scope_key = get_scope_key(model, event)
            message_content = template.render(data=model)

            try:
                message_payload = json.loads(message_content)
            except json.JSONDecodeError as e:
                logger.error(f"Template did not render valid JSON: {e}")
                raise HTTPException(
                    status_code=500, detail="Template rendered invalid JSON."
                )

            if event == "push":
                discord_notifier.send_message(
                    message_payload, event, scope_key=scope_key
                )
                discord_notifier.remove_id_file(scope_key=scope_key)
            elif event == "pull_request":
                if not discord_notifier.has_active_message(event, scope_key=scope_key):
                    discord_notifier.send_message(
                        message_payload, event, scope_key=scope_key
                    )
                else:
                    discord_notifier.edit_last_message(
                        message_payload, event, scope_key=scope_key
                    )

                if model.action == "closed":
                    discord_notifier.remove_id_file(scope_key=scope_key)
            elif event == "release":
                if not discord_notifier.has_active_message(event, scope_key=scope_key):
                    discord_notifier.send_message(
                        message_payload, event, scope_key=scope_key
                    )
                else:
                    discord_notifier.edit_last_message(
                        message_payload, event, scope_key=scope_key
                    )

                if model.action in {"published", "released", "prereleased"}:
                    discord_notifier.remove_id_file(scope_key=scope_key)
            elif event == "workflow_run":
                if not discord_notifier.has_active_message(event, scope_key=scope_key):
                    discord_notifier.send_message(
                        message_payload, event, scope_key=scope_key
                    )
                else:
                    discord_notifier.edit_last_message(
                        message_payload, event, scope_key=scope_key
                    )

                if model.workflow_run.conclusion in {
                    "success",
                    "failure",
                    "cancelled",
                    "timed_out",
                    "action_required",
                }:
                    discord_notifier.remove_id_file(scope_key=scope_key)
                if model.workflow_run.conclusion == "success":
                    workflow_path = getattr(model.workflow_run, "path", "")
                    repo_name = getattr(model.repository, "name", "")

                    # get (backend or frontend) from path using regex
                    match = re.search(
                        r"(backend|frontend)", workflow_path, re.IGNORECASE
                    )
                    project_type = match.group(1).lower() if match else ""

                    # combine repo name and project type to form key
                    key = f"{repo_name}-{project_type}" if project_type else repo_name
                    if key:
                        logger.info(
                            f"Scheduling Docker compose trigger for repo_key='{key}'"
                        )
                        background_tasks.add_task(
                            trigger_docker_compose,
                            key,
                            model,
                            event,
                            "Test Message",
                        )

                if (
                    model.workflow_run.name == "Deploy Docs"
                    and model.workflow_run.conclusion == "success"
                ):
                    announce_template = env_template.get_template("announce_docs.j2")
                    version = model.workflow_run.head_commit.message.split(" ")[-1]
                    announce_content = announce_template.render(
                        data=model, version=version
                    )

                    announce_payload = json.loads(announce_content)
                    discord_notifier.send_message(
                        announce_payload, event, scope_key=scope_key
                    )

            logger.info(
                f"Test message sent to Discord for event '{event}' successfully."
            )

        except requests.exceptions.RequestException as e:
            logger.error("Failed to send test message to Discord", exc_info=e)
            raise HTTPException(status_code=500, detail="Failed to send test message.")

    return {"status": "success", "event": event, "message_ids": message_ids}
