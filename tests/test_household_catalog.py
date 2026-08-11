from cg_isaac_envs.tasks.household.task_catalog import CATALOG, MUG_LABELS, OBJECT_NAMES, ORTHOGONAL9_CODES, STAGE_1, STAGE_2, STAGE_3, STATION_LABELS, TASK_BY_CODE, split_task_ids, task_ids_for_set, tokenize_prompt

def test_catalog_is_complete_3x3x3():
    assert len(CATALOG)==27
    assert len(TASK_BY_CODE)==27
    assert {task.code for task in CATALOG}=={(a,b,c) for a in STAGE_1 for b in STAGE_2 for c in STAGE_3}

def test_orthogonal_task_set_matches_oa9_design():
    assert ORTHOGONAL9_CODES==((1,1,1),(1,2,2),(1,3,3),(2,1,2),(2,2,3),(2,3,1),(3,1,3),(3,2,1),(3,3,2))
    assert task_ids_for_set("orthogonal")==tuple(TASK_BY_CODE[code].task_id for code in ORTHOGONAL9_CODES)
    assert task_ids_for_set("orthogonal")== (0,4,8,10,14,15,20,21,25)

def test_all_combinations_have_daily_semantics():
    assert {t.selected_object for t in CATALOG}=={"red_candy","blue_candy","green_candy"}
    assert {t.destination for t in CATALOG}=={"hot_serving_place","cold_serving_place","storage_place"}
    assert {t.associated_object for t in CATALOG}=={"mug_a","mug_b","mug_c"}
    assert all(t.family=="candy_mug_service" for t in CATALOG)

def test_objects_and_stage_language():
    assert OBJECT_NAMES==("red_candy","blue_candy","green_candy","mug_a","mug_b","mug_c")
    for task in CATALOG:
        assert len(task.stage_instructions)==3
        assert len(task.prompts)==2
        assert len(tokenize_prompt(task.canonical_prompt))==32
        relation=task.alternatives[0][1].relation
        assert relation=="on"

def test_splits_are_disjoint_and_complete():
    train=set(split_task_ids("orthogonal_train")); ood=set(split_task_ids("ood_recombination"))
    assert not train & ood
    assert train | ood==set(range(27))

def test_final_state_has_exactly_inside_and_on_relations():
    for task in CATALOG:
        relations={goal.relation for goal in task.alternatives[0]}
        assert relations=={"inside", "on"}

def test_prompts_use_short_visual_labels():
    for task in CATALOG:
        candy_label=task.selected_object.replace("_", " ")
        assert all(candy_label in prompt.lower() for prompt in task.prompts)
        assert all(MUG_LABELS[task.associated_object] in prompt for prompt in task.prompts)
        assert all(STATION_LABELS[task.destination] in prompt for prompt in task.prompts)
        assert MUG_LABELS[task.associated_object] in task.stage_instructions[2]
        assert STATION_LABELS[task.destination] in task.stage_instructions[2]
        assert all(("Use the handle." in prompt)==(task.destination=="hot_serving_place") for prompt in task.prompts)
