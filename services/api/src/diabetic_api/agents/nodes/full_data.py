"""Full data fetch node - loads comprehensive dataset for research analysis."""

import logging

from diabetic_api.agents.state import ChatState, FullDataset
from diabetic_api.services.full_data import FullDataService

logger = logging.getLogger(__name__)


class FullDataNode:
    """
    Node that fetches the full 90-day dataset for research analysis.

    Used when the router decides on 'research_full' or 'research_full_query'
    workflow paths, which require access to the complete dataset.

    Maps to N8N's 'Get Diabetic Data from Mongo (30 days)' sub-workflow.
    """

    def __init__(self, days: int = 90):
        """
        Initialize full data node.

        Args:
            days: Number of days of data to fetch (default: 90)
        """
        self.days = days

    async def __call__(self, state: ChatState) -> dict:
        """
        Fetch full dataset and add to state.

        Args:
            state: Current graph state

        Returns:
            Dict with full_data to merge into state
        """
        logger.info(f"FullDataNode fetching {self.days} days of data...")

        uow = state.get("uow")
        if uow is None:
            logger.error("No UoW in state - cannot fetch full data")
            return {
                "full_data": None,
                "last_error": "Database connection not available",
            }

        try:
            # Create service and fetch data
            service = FullDataService(uow, days=self.days)
            data = await service.get_full_dataset()

            # Log data sizes
            sensor_lines = len(data["sensorData"].split("\n")) if data["sensorData"] else 0
            basal_lines = len(data["basalData"].split("\n")) if data["basalData"] else 0
            bolus_lines = len(data["bolusData"].split("\n")) if data["bolusData"] else 0

            logger.info(
                f"FullDataNode fetched: sensor={sensor_lines} rows, "
                f"basal={basal_lines} rows, bolus={bolus_lines} rows"
            )

            full_data = FullDataset(
                sensorData=data["sensorData"],
                basalData=data["basalData"],
                bolusData=data["bolusData"],
            )

            return {"full_data": full_data}

        except Exception as e:
            logger.error(f"FullDataNode error: {e}")
            return {
                "full_data": None,
                "last_error": f"Failed to fetch full dataset: {str(e)}",
            }


def needs_full_data(state: ChatState) -> bool:
    """
    Check if the current workflow needs full data.

    Args:
        state: Current graph state

    Returns:
        True if workflow is research_full or research_full_query
    """
    decision = state.get("route_decision")
    if decision is None:
        return False

    workflow = decision.get("workflow", "")
    return workflow in ("research_full", "research_full_query")

