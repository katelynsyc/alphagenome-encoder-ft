
import os
import json
import tempfile

from ray import tune
from ray.tune.schedulers import ASHAScheduler
import optuna
from ray.tune.search.optuna import OptunaSearch
from typing import Dict, Optional, Any


LR_LIST = sorted([
    mult * (10**exp)
    for exp in range(-7, -2)       # 10^-7 up to 10^-3
    for mult in [1, 3, 5, 8]
])

BATCH_SIZES = [16, 32, 64, 128, 256, 512, 1024]
LINEAR_SIZES = [128, 256, 512, 1024, 2048, 2560, 4096]
WEIGHT_DECAYS = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4]


def define_by_run_func(trial) -> Optional[Dict[str, Any]]: #handles hyperparameter search space defs, no actual training
    """Define-by-run function to construct a conditional search space.
    Args:
        trial: Optuna Trial object
        
    Returns:
        Dict containing constant parameters or None
    """

    lr1_idx = trial.suggest_int("lr1_idx", 0, len(LR_LIST) - 1) #indexing allows sampler to learn higher indices correlated to higher values
    lr1 = LR_LIST[lr1_idx]
    if lr1_idx == 0: #no smaller value for lr2 possible
        raise optuna.exceptions.TrialPruned() 
    lr2_idx = trial.suggest_int("lr2_idx", 0, lr1_idx - 1)
    lr2 = LR_LIST[lr2_idx]
    
    num_layers = trial.suggest_int("num_layers", 1, 2)

    layer1_idx = trial.suggest_int("layer1_idx", 0, len(LINEAR_SIZES) - 1)
    hidden_sizes = [LINEAR_SIZES[layer1_idx]]

    if num_layers == 2:
        # <=  layer1_idx (not layer1_idx - 1) since you want "larger or equal", not strictly larger
        layer2_idx = trial.suggest_int("layer2_idx", 0, layer1_idx)
        hidden_sizes.append(LINEAR_SIZES[layer2_idx])

    batch_size = trial.suggest_int("batch_size", 0, len(BATCH_SIZES) - 1) #using indexes allows for better learning?
    s1_dropout = trial.suggest_float("s1_dropout", 0.0, 0.6, step=0.05)
    s2_dropout = trial.suggest_float("s2_dropout", 0.0, 0.6, step=0.05)
    weight_decay = trial.suggest_int("weight_decay", 0, len(WEIGHT_DECAYS) - 1)
   
    return None #search space is fully defined by the trial.suggest_* calls above; nothing constant to add

#model
#accuracy, for tune.report({"accuracy: "}) ??

sampler = optuna.samplers.TPESampler(seed=42, multivariate=True, group=True)

optuna_search = OptunaSearch( #optuna only has single final dictionary from tune.report() @ end of trial, not checkpoints
    space=define_by_run_func,
    metric = "val_pearson",
    mode="max",
    sampler=sampler,
)

def trainable(config): #config is dictionary populated by raytune with corresponding hyperparameters selected from search space
    lr1_idx = config["lr1_idx"] #get from ray's config dictionary, bc optuna passed them there
    lr2_idx = config["lr2_idx"]
    lr1 = LR_LIST[lr1_idx] #translate back to actual floats from indices
    lr2 = LR_LIST[lr2_idx]

    num_layers = config["num_layers"]
    hidden_sizes = [LINEAR_SIZES[config["layer1_idx"]]]
    if num_layers == 2:
        hidden_sizes.append(LINEAR_SIZES[config["layer2_idx"]])

    batch_size = BATCH_SIZES[config["batch_size"]]
    weight_decay = WEIGHT_DECAYS[config["weight_decay"]]
    s1_dropout = config["s1_dropout"] #already a real float, no index to decode
    s2_dropout = config["s2_dropout"]


    #initialize model and optimizer with the real floats here
    my_model = 
    
    # checkpoint loading
    checkpoint = tune.get_checkpoint()
    start = 1
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            with open(os.path.join(checkpoint_dir, "checkpoint.json"), "r") as f:
                state = json.load(f)
        start = state["epoch"] + 1

    for epoch in range(start, config["num_epochs"]):
        # Do some training...
        #need to compute val_pearson here

        # Checkpoint saving
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            with open(os.path.join(temp_checkpoint_dir, "checkpoint.json"), "w") as f:
                json.dump({"epoch": epoch}, f)
            tune.report(
                {"epoch": epoch, "val_pearson": val_pearson}, #key must match metric="val_pearson" passed to OptunaSearch/ASHAScheduler
                checkpoint=tune.Checkpoint.from_directory(temp_checkpoint_dir),
            ) #tune.report() is not meant to transfer models, can slow down a lot --> but then how would i checkpoint to restore a model??



#use the ASHA scheduler for pruning runs that don't look promising
asha_scheduler = ASHAScheduler( #says it will take many many days, is this 
    time_attr='epoch', #edit these for what i want
    metric='val_pearson', #want to use this with val pearson
    mode='max', #if pearson then change to max
    max_t=250, #max epochs a single trial can run, time it takes to converge??
    grace_period=10, #min epochs trial must run before ASHA kicks it, need to be careful about this bc batch sive
    reduction_factor=4, #keep top 25% of the bracket
    brackets=1,
)

#run ray tune 

#where do i then put this logic to resume if it needs to resume

storage_path = os.path.expanduser("~/ray_results")
exp_name = "name!!! of the unique path" #experiment name
path = os.path.join(storage_path, exp_name)

if tune.Tuner.can_restore(path): #restores experiment that already has happened
    tuner = tune.Tuner.restore(path, trainable=trainable, resume_errored=True) #ARE there any large object references that raytune would have put in ray.put` them in the Ray Object Store between trials?
    #if so need to tune over those object refs in param_space
else:
    tuner = tune.Tuner(
        trainable,
        param_space={
            these are supposed to be now the static, unsearched constants, so maybe get these from config file?
        },
        run_config=tune.RunConfig(storage_path=storage_path, name=exp_name),
        tune_config=tune.TuneConfig(search_alg=optuna_search, scheduler=asha_scheduler, num_samples=1000), #want a thousand samples
        failure_config=tune.FailureConfig(max_failures=4)
    )
results = tuner.fit() #generates hyperparameter configurations, wraps into trial objects
df_results = results.get_dataframe()
print(results.get_best_result(metric="val_pearson", mode="max").config)

#retrieving the best results
best_config = results.get_best_result(metric="val_pearson", mode="max").config
