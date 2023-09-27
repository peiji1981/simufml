

'''
usage example:
>> python hetero_fm.py --epochs 10 --no-shuffle --no-cipher --seed 7
'''

from IPython.core import debugger as idb

from simufml.hetero_ml.hetero_fm import HeteroFM_ACT, HeteroFM_PAS, HeteroFM_ASS
from simufml.common.optimizer import SGD
from simufml.hetero_ml.learner.party_learner import PartyLearner
from simufml.hetero_ml.learner.fed_learner import FedLearner
from simufml.utils.util import RANDOM_SEED
from simufml.utils.util import Role
from simufml.demo.util import get_args, csv2dl, auc_func, abs_data_path


data_path = abs_data_path(__file__)
ACT_csv=data_path/'breast_hetero_guest.csv'
PAS_csv=data_path/'breast_hetero_host.csv'

def hetero_fm_demo(
    skip_cipher, epochs,
    train_bs, valid_bs, rand_seed, lr, wd, shuffle
):
    #==== data ====
    ACT_train_dl, ACT_valid_dl, PAS_train_dl, PAS_valid_dl, ACT_fs, PAS_fs = csv2dl(
        active_csv=ACT_csv,
        passive_csv=PAS_csv,
        train_bs=train_bs,
        valid_bs=valid_bs,
        rand_seed=rand_seed
    )


    #==== Model ====
    embed_size = 16
    ACT_model = HeteroFM_ACT(
        feature_size=ACT_fs,
        embed_size=embed_size,
        total_features=ACT_fs+PAS_fs
    )
    PAS_model = HeteroFM_PAS(
        feature_size=PAS_fs,
        embed_size=embed_size,
        total_features=ACT_fs+PAS_fs
    )
    ASS_model = HeteroFM_ASS()


    #==== Optimizer ====
    ACT_opt = SGD(lr=lr, wd=wd)
    PAS_opt = SGD(lr=lr, wd=wd)


    #==== Party Learner ====
    ACT_learner = PartyLearner(
        role=Role.active,
        model=ACT_model,
        train_dataloader=ACT_train_dl,
        valid_dataloader=ACT_valid_dl,
        optimizer=ACT_opt
    )
    PAS_learner = PartyLearner(
        role=Role.passive,
        model=PAS_model, 
        train_dataloader=PAS_train_dl,
        valid_dataloader=PAS_valid_dl,
        optimizer=PAS_opt
    )
    ASS_learner = PartyLearner(
        role=Role.assistant,
        model=ASS_model
    )


    #==== metrics ====
    metric_funcs = {"auc": auc_func}


    #==== Learner ====
    learner = FedLearner(
        parties=[ACT_learner, PAS_learner, ASS_learner]
    )
    learner.fit(epochs, metric_funcs, shuffle, skip_cipher, True)

    return ACT_learner.epoch_loss, ACT_learner.metric_results[0]


if __name__=='__main__':
    args = get_args()
    RANDOM_SEED.seed = args.seed

    hetero_fm_demo(
        skip_cipher=args.skip_cipher, epochs=args.epochs,
        train_bs=args.train_bs, valid_bs=args.valid_bs, rand_seed=args.seed, lr=args.lr, wd=args.wd, shuffle=args.shuffle
    )
