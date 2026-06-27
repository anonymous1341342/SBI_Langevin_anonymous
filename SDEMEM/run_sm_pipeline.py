from train_sm_noscale import main as train_sm_noscale_main
from train_sm_init import main as train_sm_init_main
from train_sm_fisher import main as train_sm_fisher_main
from DebReg_fisher import main as debreg_fisher_main
import sys


def main(task_id):
    train_sm_noscale_main(task_id)
    train_sm_init_main(task_id)
    train_sm_fisher_main(task_id)
    debreg_fisher_main(task_id)


if __name__ == "__main__":
    task_id = int(sys.argv[1])
    main(task_id)
