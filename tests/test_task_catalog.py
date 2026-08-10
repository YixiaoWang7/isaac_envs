from cg_isaac_envs.tasks.desk_service.task_catalog import CATALOG, TASK_BY_ID, split_task_ids, tokenize_prompt


def test_catalog_is_stable_and_complete():
    assert len(CATALOG) == 36
    assert tuple(task.task_id for task in CATALOG) == tuple(range(36))
    assert set(TASK_BY_ID) == set(range(36))
    assert {task.family for task in CATALOG} == {"tea", "candy", "combo", "takeaway"}


def test_splits_are_disjoint_and_cover_catalog():
    train = set(split_task_ids("orthogonal_train"))
    ood = set(split_task_ids("ood_recombination"))
    assert train
    assert ood
    assert not train & ood
    assert train | ood == set(split_task_ids("all"))


def test_every_task_has_valid_prompt_and_goal_alternative():
    for task in CATALOG:
        assert len(task.prompts) == 2
        assert all(prompt.endswith(".") for prompt in task.prompts)
        assert task.alternatives
        assert all(alternative for alternative in task.alternatives)
        assert len(tokenize_prompt(task.canonical_prompt)) == 32


def test_candy_tasks_retain_two_container_solutions():
    candy_tasks = [task for task in CATALOG if task.family == "candy"]
    assert candy_tasks
    assert all(len(task.alternatives) == 2 for task in candy_tasks)
    for task in candy_tasks:
        targets = {
            goal.target
            for alternative in task.alternatives
            for goal in alternative
            if goal.relation == "inside" and goal.required
        }
        assert {"mug", "bowl"}.issubset(targets)

