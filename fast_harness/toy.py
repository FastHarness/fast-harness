import uuid

from .types import Observation, Proposal, Review, ReviewRequest, Usage


def decision(proposal: Proposal, stage: str, *, mode: str = 'student',
             failed: bool = False, automatic: bool = False, steps: int = 15) -> dict:
    first = proposal.observation.step == 0
    status = 'not_started' if first else ('failed' if failed else ('uncertain' if automatic else 'progressing'))
    return dict(request_id=proposal.request_id, mode=mode, steps=steps,
        reason='Automatic student prefix; no teacher visual review.' if automatic else 'Toy observed transition and current proposal reviewed.',
        assessment=dict(task_progress=dict(verified_completed=[], currently_attempting=stage, remaining=[]),
            current_subgoal=stage, execution_status=status,
            execution_evidence='Toy feedback indicates a failed transition.' if failed else 'Toy observation is available.',
            expected_next_intent=stage, predicted_next_intent=stage,
            intent_status='uncertain' if automatic else 'aligned',
            intent_evidence='Automatic execution does not certify visual intent.' if automatic else 'Toy proposal follows the current goal.'),
        edit={arm: dict(delta_position=[0., 0., 0.], delta_rotation_vector=[0., 0., 0.], gripper='keep')
              for arm in ('left', 'right')},
        target={arm: dict(position=[0., 0., 0.], quaternion_wxyz=[1., 0., 0., 0.], gripper_closed=False)
                for arm in ('left', 'right')})


class ToyBackend:
    def __init__(self, fault: bool = False, chunks: int = 36):
        self.episode = uuid.uuid4().hex
        self.fault, self.chunks = fault, chunks
        self.step = self.cursor = self.executions = self.corrections = 0
        self.fault_fired = self.damaged = False
        self.last_ok = True
        self.stopped = False
        self.current = None

    def observation(self) -> Observation:
        stage_index = min(2, self.cursor // max(1, self.chunks // 3))
        stage = ('approach', 'transfer', 'place')[stage_index]
        payload = dict(instruction='Complete three symbolic phases in order.',
            current_state=[self.cursor * 0.01, float(self.damaged)], images=[],
            current_eef={arm: dict(position=[0., 0., 0.], quaternion_wxyz=[1., 0., 0., 0.],
                                   gripper_opening_command=1.) for arm in ('left', 'right')},
            toy=dict(stage=stage, cursor=self.cursor, damaged=self.damaged,
                     last_ok=self.last_ok, corrections=self.corrections))
        return Observation(self.episode, self.step,
            (stage_index * 0.4, (self.cursor % max(1, self.chunks // 3)) * 0.001, float(self.damaged)),
            payload, self.cursor >= self.chunks, self.cursor >= self.chunks)

    def start(self) -> Observation:
        return self.observation()

    def infer(self, observation: Observation) -> Proposal:
        if observation.terminal or observation.step != self.step:
            raise ValueError('Cannot infer from a stale or terminal observation')
        self.current = Proposal(uuid.uuid4().hex, observation,
            dict(student_eef_trajectory=[], action_diagnostics={}),
            warnings=('toy_visible_failure',) if self.damaged else ())
        return self.current

    def auto_response(self, proposal: Proposal, steps: int, stage: str) -> dict:
        return decision(proposal, stage, automatic=True, steps=steps)

    def execute(self, proposal: Proposal, response: dict, source: str) -> Observation:
        if self.current is not proposal or response['request_id'] != proposal.request_id:
            raise ValueError('Stale toy proposal')
        self.current = None
        self.executions += 1
        self.step += response['steps']
        if response['mode'] == 'student':
            if self.fault and not self.fault_fired and self.cursor >= self.chunks // 2:
                self.fault_fired = self.damaged = True
            self.last_ok = not self.damaged
            if self.last_ok:
                self.cursor += 1
        else:
            self.corrections += 1
            self.last_ok = self.corrections not in (1, 2, 4, 5)
            if self.last_ok:
                target_cursor = round(response['target']['left']['position'][0] * 1000)
                self.cursor = target_cursor
                self.damaged = False
        return self.observation()

    def stop(self, reason: str) -> None:
        self.stopped = True


class ToyReviewer:
    def review(self, request: ReviewRequest) -> Review:
        toy = request.observation.payload['toy']
        stage = toy['stage']
        outcome = ('unknown' if request.previous is None else ('ok' if toy['last_ok'] else 'error'))
        phase = request.recovery['phase']
        active = phase != 'normal'
        status = 'continue'
        if active and request.recovery['attempt_chunks']:
            status = 'succeeded' if toy['last_ok'] else 'failed'
            if phase == 'after_return' and toy['corrections'] == 3:
                status = 'failed'
        selected = None
        if phase == 'select':
            excluded_target = request.recovery['target_checkpoint_id']
            eligible = [point for point in request.checkpoints if point.id != excluded_target]
            selected = eligible[0].id if eligible else None
        response = None
        if request.proposal is not None:
            correcting = active or toy['damaged']
            response = decision(request.proposal, stage, mode='eef' if correcting else 'student',
                                failed=toy['damaged'] or outcome == 'error', steps=5 if correcting else 15)
            if correcting:
                response['assessment']['intent_status'] = 'misaligned'
                response['assessment']['intent_evidence'] = 'The normal proposal does not pursue the active toy recovery goal.'
                cursor = toy['cursor']
                if phase.startswith('return_'):
                    point = next(point for point in request.checkpoints
                                 if point.id == request.recovery['target_checkpoint_id'])
                    cursor = point.observation.payload['toy']['cursor']
                for arm in ('left', 'right'):
                    response['target'][arm]['position'][0] = cursor / 1000
        checkpoint = None
        if outcome == 'ok' and not active:
            checkpoint = dict(goal=f'Resume {stage} from verified toy cursor {toy["cursor"]}',
                              prerequisites=['Symbolic task state remains reachable'])
        return Review(stage, outcome, 'toy_progress_failure' if outcome == 'error' else 'none',
            'Deterministic toy feedback; not a visual robotics judgement.', response,
            checkpoint, status, selected, Usage())
