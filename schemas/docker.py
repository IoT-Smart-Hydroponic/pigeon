from pydantic import BaseModel, Field, field_validator


class DockerServiceSchema(BaseModel):
    id: str = Field(
        ..., description="Unique identifier for the Docker service", alias="ID"
    )
    name: str = Field(..., description="Name of the Docker service", alias="Name")
    state: str = Field(
        ..., description="Current state of the Docker service", alias="State"
    )
    createdAt: str = Field(
        ..., description="Creation timestamp of the Docker service", alias="CreatedAt"
    )
    status: str = Field(
        ..., description="Status message of the Docker service", alias="Status"
    )
    image: str = Field(
        ..., description="Docker image used by the service", alias="Image"
    )

    @field_validator("*", mode="before")
    def strip_whitespace(cls, value):
        return value.strip() if isinstance(value, str) else value

    def get_color(self):
        state_color_map = {
            "running": 3066993,
            "healthy": 3066993,
            "starting": 15105570,
            "unhealthy": 15158332,
            "stopped": 10070709,
            "exited": 10070709,
        }
        return state_color_map.get(self.state.lower(), 8421504)  # Default to gray
