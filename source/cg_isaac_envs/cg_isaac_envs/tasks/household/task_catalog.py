"""Fully valid 3 x 3 x 3 daily meal-placement task catalog."""
from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from typing import Literal
RelationKind = Literal["inside", "on", "beside", "handle_lift"]
@dataclass(frozen=True)
class RelationGoal:
    subject: str
    relation: RelationKind
    target: str
    required: bool = True
@dataclass(frozen=True)
class TaskSpec:
    task_id: int
    code: tuple[int,int,int]
    family: str
    selected_object: str
    destination: str
    associated_object: str
    prompts: tuple[str,str]
    stage_instructions: tuple[str,str,str]
    alternatives: tuple[tuple[RelationGoal,...],...]
    manipulation_subjects: tuple[str,str]
    @property
    def side(self): return {"hot_serving_place":"hot","cold_serving_place":"cold","storage_place":"storage"}[self.destination]
    @property
    def factor_dict(self):
        return {"family":self.family,"stage_code":"-".join(map(str,self.code)),"selected_object":self.selected_object,"destination":self.destination,"associated_object":self.associated_object,"side":self.side,"item":self.associated_object}
    @property
    def canonical_prompt(self): return self.prompts[0]
STAGE_1={1:"red_candy",2:"blue_candy",3:"green_candy"}
STAGE_2={1:"mug_a",2:"mug_b",3:"mug_c"}
STAGE_3={1:"hot_serving_place",2:"cold_serving_place",3:"storage_place"}
MUG_LABELS={"mug_a":"white mug","mug_b":"blue mug","mug_c":"red mug"}
STATION_LABELS={"hot_serving_place":"red station","cold_serving_place":"blue station","storage_place":"green station"}
STAGE_NAMES=("pick_candy","place_candy_in_mug","move_mug_to_destination")
OBJECT_NAMES=("red_candy","blue_candy","green_candy","mug_a","mug_b","mug_c")
TARGET_NAMES=("hot_serving_place","cold_serving_place","storage_place")
FAMILIES=("candy_mug_service",)
def build_catalog():
    tasks=[]
    for s1,s2,s3 in product(STAGE_1,STAGE_2,STAGE_3):
        candy,mug,destination=STAGE_1[s1],STAGE_2[s2],STAGE_3[s3]
        candy_label=candy.replace('_',' ')
        mug_label=MUG_LABELS[mug]
        station_label=STATION_LABELS[destination]
        handle_note=" Use the handle." if destination=="hot_serving_place" else ""
        tasks.append(TaskSpec(
            task_id=len(tasks),code=(s1,s2,s3),family="candy_mug_service",selected_object=candy,destination=destination,associated_object=mug,
            prompts=(f"Put the {candy_label} in the {mug_label}, then move it to the {station_label}.{handle_note}",f"{candy_label.capitalize()}, {mug_label}, {station_label}.{handle_note}"),
            stage_instructions=(f"Pick up the {candy_label}.",f"Put it in the {mug_label}.",f"Move the {mug_label} to the {station_label}."+handle_note),
            alternatives=((RelationGoal(candy,"inside",mug),RelationGoal(mug,"on",destination)),),
            manipulation_subjects=(candy,mug)))
    return tuple(tasks)
CATALOG=build_catalog(); TASK_BY_ID={t.task_id:t for t in CATALOG}; TASK_BY_CODE={t.code:t for t in CATALOG}
ORTHOGONAL9_CODES=((1,1,1),(1,2,2),(1,3,3),(2,1,2),(2,2,3),(2,3,1),(3,1,3),(3,2,1),(3,3,2))
TASK_SETS={"all":tuple(task.code for task in CATALOG),"orthogonal":ORTHOGONAL9_CODES}
def task_ids_for_set(name):
    if name not in TASK_SETS: raise KeyError(name)
    return tuple(TASK_BY_CODE[code].task_id for code in TASK_SETS[name])
def split_task_ids(name):
    if name=="all": return tuple(range(len(CATALOG)))
    train=tuple(t.task_id for t in CATALOG if sum(t.code)%2==0)
    if name in {"orthogonal_train","id_eval"}: return train
    if name=="ood_recombination": return tuple(t.task_id for t in CATALOG if t.task_id not in set(train))
    raise KeyError(name)
def prompt_vocabulary():
    words={"<pad>","<unk>"}
    for task in CATALOG:
        for prompt in (*task.prompts,*task.stage_instructions): words.update(prompt.lower().replace(".","").replace(",","").split())
    return tuple(sorted(words))
VOCABULARY=prompt_vocabulary(); VOCAB_TO_ID={w:i for i,w in enumerate(VOCABULARY)}
def tokenize_prompt(prompt,max_length=32):
    words=prompt.lower().replace(".","").replace(",","").split(); ids=[VOCAB_TO_ID.get(w,VOCAB_TO_ID["<unk>"]) for w in words[:max_length]]; return tuple(ids+[VOCAB_TO_ID["<pad>"]]*(max_length-len(ids)))
