from train_single_init import main as train_single_init_main
from train_single_fisher import main as train_single_fisher_main
from DebReg_fisher import main as DebReg_fisher_main
import sys


def main(task_id):
    train_single_init_main(task_id)
    train_single_fisher_main(task_id)
    DebReg_fisher_main(task_id)


if __name__ == "__main__":
    task_id = int(sys.argv[1])
    main(task_id)