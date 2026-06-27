from train_nmodel_init import main as train_nmodel_init_main
from train_nmodel_fisher import main as train_nmodel_fisher_main
import sys


def main(task_id, simu_budget):
    train_nmodel_init_main(task_id, simu_budget)
    train_nmodel_fisher_main(task_id, simu_budget)


if __name__ == "__main__":
    task_id = int(sys.argv[1])
    simu_budget = sys.argv[2]
    main(task_id, simu_budget)
