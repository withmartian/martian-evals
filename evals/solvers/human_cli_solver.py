from evals.solvers.solver import Solver, SolverResult
from evals.task_state import Message, TaskState


class HumanCliSolver(Solver):
    """Solver that prints prompts to the command line and reads input from it.

    NOTE: With more than a single thread messages from different threads will mix,
          so this makes sense only with EVALS_SEQUENTIAL=1.
    """

    def __call__(
        self,
        task_state: TaskState,
        **kwargs,
    ) -> SolverResult:

        msgs = [Message("system", task_state.task_description)]
        msgs += task_state.messages

        prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in msgs])
        answer = input(prompt)

        return SolverResult(answer)

    @property
    def name(self) -> str:
        return "human"
