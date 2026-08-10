from cg_isaac_envs.tasks.desk_service.task_catalog import CATALOG


def test_all_positive_goal_targets_exist_or_are_destinations():
    known = {
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
        "left",
        "right",
    }
    for task in CATALOG:
        for alternative in task.alternatives:
            for goal in alternative:
                assert goal.subject in known
                assert goal.target in known
                assert goal.subject != goal.target


def test_every_task_ends_at_its_prompt_destination():
    for task in CATALOG:
        destination = task.factor_dict["destination"]
        for alternative in task.alternatives:
            delivery = [goal for goal in alternative if goal.relation == "at" and goal.required]
            assert len(delivery) == 1
            assert delivery[0].target == destination

