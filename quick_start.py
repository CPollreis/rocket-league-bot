# pyright: reportMissingTypeStubs=false, reportMissingParameterType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false

import os

# needed to prevent numpy from using a ton of memory in env processes and causing them to throttle each other
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from typing import Any, Literal, TypeAlias

import numpy as np
from rlgym.rocket_league.api import GameState

from reward_shaping import (
    AnnealSpec,
    BoostPickupReward,
    PlayerToBallDistanceReward,
    RewardTerm,
    VelocityBallToGoalReward,
    VelocityPlayerToBallReward,
    build_combined_reward,
)

AgentID: TypeAlias = str
# float32 (not float64): matches the network dtype and halves obs memory / bandwidth.
ObsType: TypeAlias = np.ndarray[tuple[Literal[92]], np.dtype[np.float32]]
ActionType: TypeAlias = np.ndarray[tuple[Literal[90]], np.dtype[np.int64]]
EngineActionType: TypeAlias = np.ndarray[tuple[Literal[8]], np.dtype[np.generic]]
RewardType: TypeAlias = float
StateType: TypeAlias = GameState[AgentID]
ObsSpaceType: TypeAlias = tuple[str, int]
ActionSpaceType: TypeAlias = tuple[str, int]

# ---------------------------------------------------------------------------
# REWARD DESIGN -- this is the knob to turn.
#
# Each RewardTerm is either a constant weight (weight=W) or an annealed one
# (anneal=AnnealSpec(start, end, steps)) whose weight ramps linearly over
# training. See reward_shaping.py for what each term actually computes and
# for how AnnealSpec.steps relates to global training timesteps.
#
# The numbers below are placeholders wired up to run end-to-end, not a
# tuned design -- replace them with your own weights/schedule.
# ---------------------------------------------------------------------------
from rlgym.rocket_league.reward_functions import GoalReward, TouchReward  # noqa: E402

REWARD_TERMS: list[RewardTerm] = [
    RewardTerm("goal", GoalReward, weight=10.0),
    RewardTerm("touch", TouchReward, weight=0.1),
    RewardTerm("boost_pickup", BoostPickupReward, weight=0.02),
    RewardTerm(
        "dist_to_ball",
        lambda: PlayerToBallDistanceReward(dispersion=1.0),
        anneal=AnnealSpec(start_weight=0.05, end_weight=0.0, steps=200_000),
    ),
    RewardTerm(
        "vel_to_ball",
        VelocityPlayerToBallReward,
        anneal=AnnealSpec(start_weight=0.1, end_weight=0.0, steps=200_000),
    ),
    RewardTerm(
        "ball_vel_to_goal",
        VelocityBallToGoalReward,
        anneal=AnnealSpec(start_weight=0.1, end_weight=0.02, steps=500_000),
    ),
]


def build_rlgym_v2_env():
    import numpy as np
    from rlgym.api import RLGym
    from rlgym.rocket_league import common_values
    from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
    from rlgym.rocket_league.done_conditions import (
        AnyCondition,
        GoalCondition,
        NoTouchTimeoutCondition,
        TimeoutCondition,
    )
    from rlgym.rocket_league.obs_builders import DefaultObs
    from rlgym.rocket_league.rlviser import RLViserRenderer
    from rlgym.rocket_league.sim import RocketSimEngine
    import time
    from rlgym.rocket_league.state_mutators import (
        FixedTeamSizeMutator,
        KickoffMutator,
        MutatorSequence,
    )
    from typing_extensions import override

    class Float32DefaultObs(DefaultObs[AgentID]):
        # DefaultObs emits float64; cast to float32 to match the network dtype.
        @override
        def _build_obs(
            self, agent: AgentID, state: GameState[AgentID], shared_info: dict[str, Any]
        ) -> np.ndarray:
            return super()._build_obs(agent, state, shared_info).astype(np.float32)

    class RealTimeRLViserRenderer(RLViserRenderer):
        """RLViserRenderer that reports the actually-measured wall-clock gap
        between render() calls as tick_rate, instead of a fixed guess. rlviser
        only uses tick_rate to time its interpolation between the last two
        states it received, so telling it the truth here makes on-screen
        playback track the sim's real speed -- faster or slower than real
        time -- instead of a constant tied to a target render fps. Only
        meaningful with no artificial per-step delay (see render_delay below);
        a fixed delay would make every measured gap the same anyway.
        """

        # Clamp + smooth so one anomalously fast/slow call (e.g. right after
        # an episode reset) doesn't cause a visible single-frame speed spike.
        _MIN_TICK_RATE = 1.0
        _MAX_TICK_RATE = 1000.0
        _SMOOTHING = 0.25  # weight given to the newest sample

        def __init__(self):
            super().__init__()
            self._last_call_time: float | None = None
            self._primed = False

        def _prime_field_load(self) -> None:
            # rlviser's GameMode resource defaults to TheVoid (confirmed via
            # a patched debug build with added logging -- not Soccar, despite
            # Soccar being flatbuffer variant 0). load_field() explicitly
            # skips loading any stadium mesh at all when game_mode is
            # TheVoid, and only reloads on a later genuine mode *change*
            # (caught by update_field's states.current.game_mode != *game_mode
            # check). So without this, rlviser silently stays on its bare
            # fallback scene (ball only) forever, even though the full
            # stadium + car meshes are right there in its bundled cache --
            # confirmed via the same debug build: sending a real different
            # mode then Soccar produces dozens of "Spawning Field_STD_*" /
            # "Goal_STD_*" log lines, i.e. the stadium genuinely spawns.
            #
            # rlviser only keeps the *last* GameState packet received
            # between two of its own render ticks (up to ~120/s), and with
            # its default packet smoothing, an applied packet doesn't even
            # reach `current` until the *next* one arrives (one-packet
            # lag). A single different-mode packet gets lost in either gap.
            # So instead, broadcast a throwaway mode continuously for long
            # enough that rlviser is guaranteed to tick on it and settle
            # `current` there, then broadcast real Soccar the same way --
            # two genuine, sustained mode changes, each easily surviving
            # the coalescing/lag above.
            import time as _time

            import RocketSim as rsim
            import rlviser_py as rlviser

            def _broadcast(mode, seconds: float) -> None:
                deadline = _time.perf_counter() + seconds
                while _time.perf_counter() < deadline:
                    rlviser.render(
                        tick_count=0,
                        tick_rate=self.tick_rate,
                        game_mode=mode,
                        boost_pad_states=[],
                        ball=rsim.BallState(),
                        cars=[],
                    )
                    _time.sleep(1 / 60)

            _broadcast(rsim.GameMode.HOOPS, 0.5)
            _broadcast(rsim.GameMode.SOCCAR, 0.5)

        @override
        def render(self, state: GameState, shared_info: dict[str, Any]) -> Any:
            if not self._primed:
                self._primed = True
                self._prime_field_load()
            now = time.perf_counter()
            if self._last_call_time is not None:
                dt = now - self._last_call_time
                if dt > 0:
                    measured = min(
                        self._MAX_TICK_RATE, max(self._MIN_TICK_RATE, 1.0 / dt)
                    )
                    self.tick_rate = (
                        self._SMOOTHING * measured
                        + (1 - self._SMOOTHING) * self.tick_rate
                    )
            self._last_call_time = now
            return super().render(state, shared_info)

    spawn_opponents = True
    team_size = 2
    blue_team_size = team_size
    orange_team_size = team_size if spawn_opponents else 0
    action_repeat = 8
    no_touch_timeout_seconds = 30
    game_timeout_seconds = 300

    action_parser = RepeatAction(LookupTableAction(), repeats=action_repeat)
    termination_condition = GoalCondition()
    truncation_condition = AnyCondition(
        NoTouchTimeoutCondition(timeout_seconds=no_touch_timeout_seconds),
        TimeoutCondition(timeout_seconds=game_timeout_seconds),
    )

    reward_fn = build_combined_reward(REWARD_TERMS)

    obs_builder = Float32DefaultObs(
        zero_padding=team_size,
        pos_coef=np.asarray(  # pyright: ignore [reportArgumentType]
            [
                1 / common_values.SIDE_WALL_X,
                1 / common_values.BACK_NET_Y,
                1 / common_values.CEILING_Z,
            ]
        ),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
        ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL,
        boost_coef=1 / 100.0,
    )

    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=blue_team_size, orange_size=orange_team_size),
        KickoffMutator(),
    )
    return RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=termination_condition,
        truncation_cond=truncation_condition,
        transition_engine=RocketSimEngine(),
        # Only used when process_config.render is True.
        renderer=RealTimeRLViserRenderer(),
    )


if __name__ == "__main__":
    from typing import cast

    import numpy as np
    from pydantic import JsonValue
    from rlgym_learn import (
        BaseConfigModel,
        LearningCoordinator,
        LearningCoordinatorConfigModel,
        ProcessConfigModel,
        SerdeTypesModel,
        generate_config,
    )
    from rlgym_learn.pyany_serde import PyAnySerdeType
    from rlgym_learn_algos.logging.wandb import (
        WandbMetricsLogger,
        WandbMetricsLoggerConfigModel,
        ppo_additional_derived_config_factory,
    )
    from rlgym_learn_algos.ppo import (
        ActorCritic,
        BasicCritic,
        DiscreteFF,
        ExperienceBufferConfigModel,
        GAETrajectoryProcessor,
        GAETrajectoryProcessorConfigModel,
        NumpyExperienceBuffer,
        PPOAgentController,
        PPOAgentControllerConfigModel,
        PPOLearnerConfigModel,
        PPOMetricsLogger,
        SeparateActorCritic,
        log_actor_critic_parameter_counts,
    )
    from torch import device as _device
    from torch import dtype as _dtype
    from torch.optim import Adam, Optimizer

    # The obs_space_type and action_space_type are determined by your choice of ObsBuilder and ActionParser respectively.
    # The logic used here assumes you are using the types defined by the DefaultObs and LookupTableAction above.
    DefaultObsSpaceType = tuple[str, int]
    DefaultActionSpaceType = tuple[str, int]

    def actor_critic_factory(
        obs_space: tuple[str, int],
        action_space: tuple[str, int],
        dtype: _dtype,
        device: _device,
        agent_controller: str | None,
    ) -> ActorCritic[AgentID, ObsType, ActionType]:
        actor = DiscreteFF(
            obs_space[1], action_space[1], (256, 256, 256), dtype, device
        )
        critic = BasicCritic(obs_space[1], (256, 256, 256), dtype, device)
        log_actor_critic_parameter_counts(actor, critic, agent_controller)
        return SeparateActorCritic(
            actor,
            critic,
        )

    def optimizers_factory(
        actor_critic: ActorCritic[AgentID, ObsType, ActionType],
        optimizer_named_parameter_group_kwargs: dict[str, dict[str, JsonValue]],
        agent_controller: str | None,
    ) -> list[Optimizer]:
        actor_critic = cast(
            SeparateActorCritic[AgentID, ObsType, ActionType], actor_critic
        )
        print(
            f"{agent_controller}: Current Actor Optimizer Kwargs: {optimizer_named_parameter_group_kwargs['actor']}"
        )
        print(
            f"{agent_controller}: Current Critic Optimizer Kwargs {optimizer_named_parameter_group_kwargs['critic']}"
        )
        actor_optimizer = Adam(
            actor_critic.actor.parameters(),
            **optimizer_named_parameter_group_kwargs["actor"],  # pyright: ignore [reportArgumentType]
        )
        critic_optimizer = Adam(
            actor_critic.critic.parameters(),
            **optimizer_named_parameter_group_kwargs["critic"],  # pyright: ignore [reportArgumentType]
        )
        return [actor_optimizer, critic_optimizer]

    import torch

    def resolve_training_device() -> str:
        # Force a specific device with e.g. RL_DEVICE=cpu / cuda:0 / cuda:1
        forced = os.environ.get("RL_DEVICE")
        if forced:
            return forced
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    training_device = resolve_training_device()
    print(f"[quick_start] training on device: {training_device}")

    # RocketSim envs are CPU-only. Rough starting point: (physical cores - 2).
    # More procs = more RAM and, up to a point, more steps/sec. Override with RL_N_PROC.
    n_proc = int(os.environ.get("RL_N_PROC", "8"))

    # ------------------------------------------------------------------
    # HYPERPARAMETERS -- everything PPO/GAE actually reads is spelled out
    # here instead of relying on rlgym_learn_algos' pydantic defaults, so
    # this is the one place to look/edit. Placeholders, not a tuned config.
    #
    # Gt      = sum_k gamma^k * r_{t+k}                    (GAMMA)
    # delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    # A_t     = sum_k (gamma*lambda)^k * delta_{t+k}        (GAMMA, GAE_LAMBDA)
    # L_CLIP  = E[min(rho*A, clip(rho, 1-eps, 1+eps)*A)]    (CLIP_RANGE = eps)
    # ------------------------------------------------------------------
    GAMMA = 0.99  # discount factor, in (0, 1)
    GAE_LAMBDA = 0.95  # GAE bias/variance trade-off, in [0, 1]
    CLIP_RANGE = 0.2  # PPO clip width (epsilon), typically 0.1-0.3
    ENT_COEF = 0.01  # entropy bonus weight; higher = more exploration
    LR_ACTOR = 5e-5
    LR_CRITIC = 5e-5
    N_EPOCHS = 1  # passes over each collected batch
    BATCH_SIZE = 50_000  # timesteps per learning batch
    N_MINIBATCHES = 1  # minibatches per epoch (BATCH_SIZE must divide evenly)
    MAX_GRAD_NORM = 0.5  # gradient clipping
    REWARD_CLIP = 10.0  # clip per-step reward magnitude before GAE
    EXPERIENCE_BUFFER_SIZE = 150_000  # timesteps retained (rolling window)
    TIMESTEPS_PER_ITERATION = 50_000  # learner update cadence

    # Set RL_RENDER=1 to watch env process 0 in the RLViser window (needs a
    # local display; not over SSH or inside Docker).
    render = os.environ.get("RL_RENDER", "0") == "1"
    # No artificial per-step delay: env process 0 steps as fast as it
    # actually can, and RealTimeRLViserRenderer (in build_rlgym_v2_env)
    # measures the real wall-clock gap between steps and reports that to
    # rlviser as the tick rate, so on-screen playback tracks the sim's true
    # speed -- faster or slower than real time -- instead of a fixed
    # multiplier. Watch with RL_N_PROC=1 (or 2): the render loop competes
    # with every other env process and the learner for CPU, and less
    # contention means a more even, less jittery feed. rlviser also reads
    # fps_limit / vsync from its settings.txt and has a live game-speed
    # control if you want to slow down or speed up what you're watching.
    render_delay = None
    if render:
        print("[quick_start] rendering env 0 in real time (uncapped, no artificial pacing)")
        if n_proc > 2:
            print(
                f"[quick_start] RL_N_PROC={n_proc}: playback will be smoother "
                f"with RL_N_PROC=1 while watching"
            )

    # wandb logging is ON by default. Run `wandb login` once (or export
    # WANDB_API_KEY), or set WANDB_MODE=offline to log locally with no account.
    # Set RL_WANDB=0 to turn it off entirely.
    use_wandb = os.environ.get("RL_WANDB", "1") != "0"
    if use_wandb:
        # rlgym_learn's WandbMetricsLoggerConfigModel has no entity field, so
        # wandb.init() relies on the account's default entity. Pin it here so
        # runs land in a known team even if that account default changes.
        # Override by exporting WANDB_ENTITY before launching.
        os.environ.setdefault("WANDB_ENTITY", "cpollreis")
        metrics_logger = WandbMetricsLogger(
            PPOMetricsLogger(), ppo_additional_derived_config_factory
        )
        metrics_logger_config = WandbMetricsLoggerConfigModel(
            inner_metrics_logger_config=None, group="rlgym-learn-testing"
        )
    else:
        metrics_logger = PPOMetricsLogger()
        metrics_logger_config = None

    # Create the config that will be used for the run
    config = LearningCoordinatorConfigModel(
        base_config=BaseConfigModel(
            serde_types=SerdeTypesModel(
                agent_id_serde_type=PyAnySerdeType.STRING(),
                action_serde_type=PyAnySerdeType.NUMPY(np.int64),
                obs_serde_type=PyAnySerdeType.NUMPY(np.float32),
                reward_serde_type=PyAnySerdeType.FLOAT(),
                obs_space_serde_type=PyAnySerdeType.TUPLE(
                    (PyAnySerdeType.STRING(), PyAnySerdeType.INT())
                ),
                action_space_serde_type=PyAnySerdeType.TUPLE(
                    (PyAnySerdeType.STRING(), PyAnySerdeType.INT())
                ),
            ),
            timestep_limit=1_000_000_000,  # Train for 1B steps
        ),
        process_config=ProcessConfigModel(
            n_proc=n_proc,  # Number of processes to spawn to run environments. Increasing will use more RAM but should increase steps per second, up to a point
            render=render,
            render_delay=render_delay,
        ),
        agent_controller_config=PPOAgentControllerConfigModel(
            timesteps_per_iteration=TIMESTEPS_PER_ITERATION,
            learner_config=PPOLearnerConfigModel(
                n_epochs=N_EPOCHS,
                batch_size=BATCH_SIZE,
                n_minibatches=N_MINIBATCHES,
                ent_coef=ENT_COEF,
                clip_range=CLIP_RANGE,
                max_grad_norm=MAX_GRAD_NORM,
                optimizer_named_parameter_group_kwargs={
                    "actor": {
                        "lr": LR_ACTOR  # see optimizers_factory above
                    },
                    "critic": {
                        "lr": LR_CRITIC  # see optimizers_factory above
                    },
                },
                device=training_device,  # pyright: ignore [reportArgumentType]
            ),
            experience_buffer_config=ExperienceBufferConfigModel(
                max_size=EXPERIENCE_BUFFER_SIZE,  # timesteps to retain; older ones are pruned
                trajectory_processor_config=GAETrajectoryProcessorConfigModel(
                    gamma=GAMMA,
                    lmbda=GAE_LAMBDA,
                    reward_clip=REWARD_CLIP,
                ),
                device="cpu",  # pyright: ignore [reportArgumentType]
            ),
            metrics_logger_config=metrics_logger_config,
        ),
        agent_controller_save_folder="agent_controller_checkpoints",  # (default value) WARNING: THIS PROCESS MAY DELETE ANYTHING INSIDE THIS FOLDER. This determines the parent folder for the runs for each agent controller. The runs folder for the agent controller will be this folder and then the agent controller config key as a subfolder.
    )

    # Generate the config file for reference (this file location can be
    # passed to the learning coordinator via config_location instead of defining
    # the config object in code and passing that)
    generate_config(
        learning_coordinator_config=config,
        config_location="config.json",
        force_overwrite=True,
    )

    learning_coordinator = LearningCoordinator(
        build_rlgym_v2_env,
        agent_controller=PPOAgentController(
            actor_critic_factory=actor_critic_factory,
            optimizers_factory=optimizers_factory,
            experience_buffer=NumpyExperienceBuffer(GAETrajectoryProcessor()),
            metrics_logger=metrics_logger,
            obs_standardizer=None,
        ),
        config=config,
    )
    learning_coordinator.start()