from vela.core.output_generator import OutputGenerator, OutputResult
from vela.domain.base import DomainPlugin
from vela.llm.base import BaseLLM


class VelaPipeline:
    """
    3-layer pipeline runner.

    Layer 01: Signal Detector  — 로컬 연산, LLM 없음, ~5%만 통과
    Layer 02: Intent Classifier — 단순 케이스는 조건 분기, 복합만 LLM
    Layer 03: Output Generator  — LLM으로 최종 인사이트 생성
    """

    def __init__(self, plugin: DomainPlugin, llm: BaseLLM) -> None:
        self._plugin = plugin
        self._generator = OutputGenerator(llm=llm)
        self._baseline: dict = {}

    def run(self, current: dict, history: list[dict]) -> OutputResult | None:
        """
        current: 현재 데이터 스냅샷
        history: 과거 데이터 목록 (베이스라인 계산용)
        반환값이 None이면 개입 불필요 (비용 $0)
        """
        detector = self._plugin.get_signal_detector()

        if not self._baseline:
            self._baseline = detector.compute_baseline(history)

        signal = detector.detect(current, self._baseline)
        if not signal.should_act:
            return None

        intent = self._plugin.get_intent_classifier().classify(signal)
        return self._generator.generate(signal, intent, self._plugin.get_output_prompt())
