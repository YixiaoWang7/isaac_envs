from cg_isaac_envs.tasks.household.task_catalog import CATALOG, OBJECT_NAMES, STAGE_1, STAGE_2, STAGE_3, TASK_BY_CODE, split_task_ids, tokenize_prompt

def test_catalog_is_complete_3x3x3():
    assert len(CATALOG)==27
    assert len(TASK_BY_CODE)==27
    assert {task.code for task in CATALOG}=={(a,b,c) for a in STAGE_1 for b in STAGE_2 for c in STAGE_3}

def test_all_combinations_have_daily_semantics():
    assert {t.selected_object for t in CATALOG}=={"bowl","plate","serving_pan"}
    assert {t.destination for t in CATALOG}=={"left_place","right_place","packing_place"}
    assert {t.associated_object for t in CATALOG}=={"apple","banana","snack_package"}
    assert all(t.family=="meal_setup" for t in CATALOG)

def test_objects_and_stage_language():
    assert OBJECT_NAMES==("bowl","plate","serving_pan","apple","banana","snack_package")
    for task in CATALOG:
        assert len(task.stage_instructions)==3
        assert len(task.prompts)==2
        assert len(tokenize_prompt(task.canonical_prompt))==32
        relation=task.alternatives[0][1].relation
        assert relation==("inside" if task.selected_object=="bowl" else "on")

def test_splits_are_disjoint_and_complete():
    train=set(split_task_ids("orthogonal_train")); ood=set(split_task_ids("ood_recombination"))
    assert not train & ood
    assert train | ood==set(range(27))
