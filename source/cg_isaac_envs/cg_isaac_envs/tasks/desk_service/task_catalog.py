"""Deterministic task catalog and prompt vocabulary for desk service."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Literal

RelationKind = Literal["inside", "on", "at"]


@dataclass(frozen=True)
class RelationGoal:
    """A required or forbidden object relation."""

    subject: str
    relation: RelationKind
    target: str
    required: bool = True


@dataclass(frozen=True)
class TaskSpec:
    """One semantic task independent of wording."""

    task_id: int
    family: str
    factors: tuple[tuple[str, str], ...]
    prompts: tuple[str, str]
    alternatives: tuple[tuple[RelationGoal, ...], ...]
    stages: tuple[str, ...] = ("prepare", "accessorize", "pack", "deliver")

    @property
    def factor_dict(self) -> dict[str, str]:
        return dict(self.factors)

    @property
    def canonical_prompt(self) -> str:
        return self.prompts[0]


OBJECT_NAMES = (
    "mug",
    "bowl",
    "takeaway_box",
    "tray",
    "tea",
    "sugar",
    "red_candy",
    "blue_candy",
    "cookie",
    "spoon",
    "coaster",
    "napkin",
)
DESTINATIONS = ("left", "right")
STAGE_NAMES = ("prepare", "accessorize", "pack", "deliver")


def _goal(subject: str, relation: RelationKind, target: str, required: bool = True) -> RelationGoal:
    return RelationGoal(subject, relation, target, required)


def _forbid_unrequested(selected: set[str], container: str) -> list[RelationGoal]:
    payloads = {"tea", "sugar", "red_candy", "blue_candy", "cookie"}
    return [_goal(name, "inside", container, False) for name in sorted(payloads - selected)]


def build_catalog() -> tuple[TaskSpec, ...]:
    """Build the stable 36-task catalog."""

    tasks: list[TaskSpec] = []

    def add(family: str, factors: dict[str, str], prompts: tuple[str, str], alternatives: list[list[RelationGoal]]):
        tasks.append(
            TaskSpec(
                task_id=len(tasks),
                family=family,
                factors=tuple(factors.items()),
                prompts=prompts,
                alternatives=tuple(tuple(goals) for goals in alternatives),
            )
        )

    for sweetener, spoon, destination in product(("none", "sugar"), ("no", "yes"), DESTINATIONS):
        selected = {"tea"} | ({"sugar"} if sweetener == "sugar" else set())
        goals = [_goal(name, "inside", "mug") for name in sorted(selected)]
        goals += _forbid_unrequested(selected, "mug")
        goals += [_goal("coaster", "on", "tray"), _goal("mug", "on", "coaster")]
        if spoon == "yes":
            goals.append(_goal("spoon", "on", "tray"))
        goals.append(_goal("tray", "at", destination))
        extras = " with sugar" if sweetener == "sugar" else ""
        utensil = " and include the spoon" if spoon == "yes" else ""
        add(
            "tea",
            {"sweetener": sweetener, "spoon": spoon, "destination": destination},
            (
                f"Prepare tea{extras}{utensil} and serve it at the {destination} station.",
                f"Serve {destination}: tea{extras}{' with the spoon' if spoon == 'yes' else ''}.",
            ),
            [goals],
        )

    for candies, napkin, destination in product(("red", "blue", "both"), ("no", "yes"), DESTINATIONS):
        selected = {"red_candy", "blue_candy"} if candies == "both" else {f"{candies}_candy"}
        alternatives = []
        for container in ("mug", "bowl"):
            goals = [_goal(name, "inside", container) for name in sorted(selected)]
            goals += _forbid_unrequested(selected, container)
            if napkin == "yes":
                goals.append(_goal("napkin", "on", "tray"))
            goals += [_goal(container, "on", "tray"), _goal("tray", "at", destination)]
            alternatives.append(goals)
        candy_words = "both candies" if candies == "both" else f"the {candies} candy"
        napkin_words = " with a napkin" if napkin == "yes" else ""
        add(
            "candy",
            {"candies": candies, "napkin": napkin, "destination": destination},
            (
                f"Serve {candy_words}{napkin_words} at the {destination} station.",
                f"Take {candy_words}{napkin_words} to the {destination} serving area.",
            ),
            alternatives,
        )

    for sweetener, candy, destination in product(("none", "sugar"), ("red", "blue"), DESTINATIONS):
        tea_selected = {"tea"} | ({"sugar"} if sweetener == "sugar" else set())
        candy_name = f"{candy}_candy"
        goals = [_goal(name, "inside", "mug") for name in sorted(tea_selected)]
        goals += _forbid_unrequested(tea_selected, "mug")
        goals += [_goal(candy_name, "inside", "bowl")]
        goals += _forbid_unrequested({candy_name}, "bowl")
        goals += [
            _goal("coaster", "on", "tray"),
            _goal("napkin", "on", "tray"),
            _goal("mug", "on", "coaster"),
            _goal("bowl", "on", "tray"),
            _goal("tray", "at", destination),
        ]
        add(
            "combo",
            {"sweetener": sweetener, "candy": candy, "destination": destination},
            (
                f"Prepare {'sweet ' if sweetener == 'sugar' else ''}tea and the {candy} candy, then serve both at the {destination} station.",
                f"Bring the {destination} station tea{' with sugar' if sweetener == 'sugar' else ''} plus the {candy} candy.",
            ),
            [goals],
        )

    for candy, cookie, destination in product(("red", "blue"), ("no", "yes"), DESTINATIONS):
        selected = {f"{candy}_candy"} | ({"cookie"} if cookie == "yes" else set())
        goals = [_goal(name, "inside", "takeaway_box") for name in sorted(selected)]
        goals += _forbid_unrequested(selected, "takeaway_box")
        goals.append(_goal("takeaway_box", "at", destination))
        add(
            "takeaway",
            {"candy": candy, "cookie": cookie, "destination": destination},
            (
                f"Pack the {candy} candy{' and cookie' if cookie == 'yes' else ''} and take the box to the {destination} pickup area.",
                f"For {destination} pickup, box the {candy} candy{' together with the cookie' if cookie == 'yes' else ''}.",
            ),
            [goals],
        )

    return tuple(tasks)


CATALOG = build_catalog()
TASK_BY_ID = {task.task_id: task for task in CATALOG}


def split_task_ids(name: str) -> tuple[int, ...]:
    """Return deterministic research splits."""

    if name == "all":
        return tuple(range(len(CATALOG)))
    train = tuple(task.task_id for task in CATALOG if sum(ord(c) for c in repr(task.factors)) % 2 == 0)
    if name in {"orthogonal_train", "id_eval"}:
        return train
    if name == "ood_recombination":
        train_set = set(train)
        return tuple(task.task_id for task in CATALOG if task.task_id not in train_set)
    raise KeyError(f"Unknown task split: {name}")


def prompt_vocabulary() -> tuple[str, ...]:
    words = {"<pad>", "<unk>"}
    for task in CATALOG:
        for prompt in task.prompts:
            words.update(prompt.lower().replace(".", "").replace(",", "").split())
    return tuple(sorted(words))


VOCABULARY = prompt_vocabulary()
VOCAB_TO_ID = {word: index for index, word in enumerate(VOCABULARY)}


def tokenize_prompt(prompt: str, max_length: int = 32) -> tuple[int, ...]:
    """Tokenize catalog prompts without depending on an external language model."""

    words = prompt.lower().replace(".", "").replace(",", "").split()
    ids = [VOCAB_TO_ID.get(word, VOCAB_TO_ID["<unk>"]) for word in words[:max_length]]
    return tuple(ids + [VOCAB_TO_ID["<pad>"]] * (max_length - len(ids)))
