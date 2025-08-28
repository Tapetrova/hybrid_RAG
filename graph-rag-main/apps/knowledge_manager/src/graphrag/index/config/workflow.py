"""A module containing 'PipelineWorkflowReference' model."""

from __future__ import annotations

from typing import Any

from libs.python.schemas.basic_models import BaseModelUpd, BaseEnum
from pydantic import Field as pydantic_Field

PipelineWorkflowStep = dict[str, Any]
"""Represent a step in a workflow."""

PipelineWorkflowConfig = dict[str, Any]
"""Represent a configuration for a workflow."""


class PipelineWorkflowReference(BaseModelUpd):
    """Represent a reference to a workflow, and can optionally be the workflow itself."""

    name: str | None = pydantic_Field(description="Name of the workflow.", default=None)
    """Name of the workflow."""

    steps: list[PipelineWorkflowStep] | None = pydantic_Field(
        description="The optional steps for the workflow.", default=None
    )
    """The optional steps for the workflow."""

    config: PipelineWorkflowConfig | None = pydantic_Field(
        description="The optional configuration for the workflow.", default=None
    )
    """The optional configuration for the workflow."""
