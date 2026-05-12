import asyncio
from typing import Callable, Awaitable

from vela.core.output_generator import OutputResult
from vela.core.pipeline import VelaPipeline
from vela.domain.base import DomainPlugin
from vela.llm.base import BaseLLM


class VelaScheduler:
    """
    능동성의 핵심 — 사용자가 앱을 열지 않아도 백그라운드에서 파이프라인 실행.

    사용 예:
        scheduler = VelaScheduler(plugin=FandomDomainPlugin(), llm=ClaudeLLM(...))
        await scheduler.run_loop(user_data_fn=fetch_all_users)
    """

    def __init__(
        self,
        plugin: DomainPlugin,
        llm: BaseLLM,
        interval_minutes: int = 30,
    ) -> None:
        self.pipeline = VelaPipeline(plugin, llm)
        self._interval = interval_minutes * 60
        self._histories: dict[str, list[dict]] = {}

    async def run_once(self, user_id: str, data: dict) -> OutputResult | None:
        """
        1. Signal Detector 실행
        2. threshold 미달 → None 반환 (종료, 비용 $0)
        3. threshold 초과 → Intent Classifier → Output Generator
        4. 결과 반환 (푸시 알림 or 인앱 카드)
        """
        history = self._histories.get(user_id, [])
        result = self.pipeline.run(data, history)
        history.append(data)
        self._histories[user_id] = history
        return result

    async def run_loop(
        self,
        user_data_fn: Callable[[], Awaitable[dict[str, dict]]],
    ) -> None:
        """주기적으로 모든 활성 사용자에 대해 run_once 실행."""
        while True:
            user_data: dict[str, dict] = await user_data_fn()
            for user_id, data in user_data.items():
                await self.run_once(user_id, data)
            await asyncio.sleep(self._interval)
