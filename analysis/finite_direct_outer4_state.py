"""Atomic block transactions, with explicit restart-uncommitted semantics."""
from copy import deepcopy
from pathlib import Path
import json
import torch
from analysis.nested_equilibrium_utils import atomic_torch_save,atomic_write_json,stable_json_hash
from data.finite_direct_sampling import file_hash


class StageJournal:
    def __init__(self,root,contract,backend,*,resume=False):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True)
        self.contract=contract; self.backend=backend; self.completed=[]; self.results={}
        self.pointer=self.root/'latest.json'
        if self.pointer.exists():
            if not resume: raise RuntimeError('existing run requires --resume')
            self.restore()
        elif resume: raise RuntimeError('resume requires a committed checkpoint')
        elif list(self.root.iterdir()): raise RuntimeError('uncommitted run directory is not empty')
        else: self._publish('INITIALIZED',{})

    def _publish(self,stage,result):
        generation=len(self.completed); target=self.root/f'checkpoint_{generation:05d}.pt'
        payload=dict(contract=self.contract,completed=list(self.completed),results=deepcopy(self.results),
                     backend=self.backend.snapshot(),stage=stage,result=result,
                     resume_semantics='RESTART_UNCOMMITTED_BLOCK')
        atomic_torch_save(target,payload)
        # Validate the durable file before publishing the sole commit pointer.
        read=torch.load(target,map_location='cpu',weights_only=False)
        if read['contract']!=self.contract or read['completed']!=self.completed: raise RuntimeError('checkpoint verification failed')
        digest=file_hash(target)
        meta=self.root/f'checkpoint_{generation:05d}.json'
        atomic_write_json(meta,dict(stage=stage,completed=list(self.completed),sha256=digest,
                                   contract_hash=stable_json_hash(self.contract),result=result))
        atomic_write_json(self.pointer,dict(checkpoint=target.name,sha256=digest,metadata=meta.name))

    def restore(self):
        pointer=json.loads(self.pointer.read_text()); target=self.root/pointer['checkpoint']
        if target.parent!=self.root or file_hash(target)!=pointer['sha256']: raise RuntimeError('corrupt committed checkpoint')
        payload=torch.load(target,map_location='cpu',weights_only=False)
        if payload['contract']!=self.contract: raise RuntimeError('code/economics/data contract mismatch on resume')
        self.completed=payload['completed']; self.results=payload['results']
        self.backend.restore(payload['backend'])

    def run(self,stage,action):
        if stage in self.completed: return self.results[stage]
        try:
            result=action() or {}
            # JSON-serializable metrics; models/state are in backend snapshot.
            self.results[stage]=result; self.completed.append(stage)
            self._publish(stage,result)
            return result
        except BaseException as e:
            atomic_write_json(self.root/'last_interruption.json',dict(stage=stage,error=repr(e),
                recovery='RESTART_UNCOMMITTED_BLOCK',latest_pointer=str(self.pointer)))
            raise
