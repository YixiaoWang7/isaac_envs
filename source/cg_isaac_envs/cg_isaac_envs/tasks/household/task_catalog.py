"""Fully valid 3 x 3 x 3 daily meal-placement task catalog."""
from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from typing import Literal
RelationKind = Literal["inside", "on"]
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
    def side(self): return {"left_place":"left","right_place":"right","packing_place":"packing"}[self.destination]
    @property
    def factor_dict(self):
        return {"family":self.family,"stage_code":"-".join(map(str,self.code)),"selected_object":self.selected_object,"destination":self.destination,"associated_object":self.associated_object,"side":self.side,"item":self.associated_object}
    @property
    def canonical_prompt(self): return self.prompts[0]
STAGE_1={1:"bowl",2:"plate",3:"serving_pan"}
STAGE_2={1:"left_place",2:"right_place",3:"packing_place"}
STAGE_3={1:"apple",2:"banana",3:"snack_package"}
STAGE_NAMES=("pick_container","place_container","place_food")
OBJECT_NAMES=("bowl","plate","serving_pan","apple","banana","snack_package")
TARGET_NAMES=("left_place","right_place","packing_place")
FAMILIES=("meal_setup",)
def build_catalog():
    tasks=[]
    for s1,s2,s3 in product(STAGE_1,STAGE_2,STAGE_3):
        container,destination,food=STAGE_1[s1],STAGE_2[s2],STAGE_3[s3]
        relation="inside" if container=="bowl" else "on"
        place={"left_place":"left dining place","right_place":"right dining place","packing_place":"packing place"}[destination]
        prep="in" if relation=="inside" else "on"
        tasks.append(TaskSpec(
            task_id=len(tasks),code=(s1,s2,s3),family="meal_setup",selected_object=container,destination=destination,associated_object=food,
            prompts=(f"Place the {container.replace('_',' ')} at the {place} and put the {food.replace('_',' ')} {prep} it.",f"Set the {container.replace('_',' ')} at the {place}, then add the {food.replace('_',' ')}."),
            stage_instructions=(f"Pick up the {container.replace('_',' ')}.",f"Place it at the {place}.",f"Put the {food.replace('_',' ')} {prep} it."),
            alternatives=((RelationGoal(container,"on",destination),RelationGoal(food,relation,container)),),
            manipulation_subjects=(container,food)))
    return tuple(tasks)
CATALOG=build_catalog(); TASK_BY_ID={t.task_id:t for t in CATALOG}; TASK_BY_CODE={t.code:t for t in CATALOG}
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
