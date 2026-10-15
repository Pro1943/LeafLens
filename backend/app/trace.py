from app.schemas import TraceStep, TraceStatus


class TraceRecorder:
    def __init__(self) -> None:
        self.steps: list[TraceStep] = []
        self._step_counter: int = 1

    def record(
        self,
        tool: str,
        reason: str,
        result: str,
        status: TraceStatus = "ok",
        duration_ms: int = 0,
    ) -> TraceStep:
        step = TraceStep(
            step=self._step_counter,
            tool=tool,
            reason=reason,
            result=result,
            status=status,
            duration_ms=duration_ms,
        )
        self.steps.append(step)
        self._step_counter += 1
        return step

    def get_steps(self) -> list[TraceStep]:
        return list(self.steps)
