import numpy as np
from warm_start.warm_start import WarmStart
from warm_start.experience import Experience, ExperienceStore

def test_warm_start():
    X_train = ["doc uno", "doc dos", "doc tres"]
    y_train = ["pos", "pos", "neg"]

    warm = WarmStart(
        positive_min_threshold=0.2,
        k_pos=2,
        k_neg=2,
        max_alpha=0.05,
        min_alpha=-0.02,
    )

    warm.pre_warm_up(X_train, y_train)

    def my_generator_fn(sampler):
        classifier = sampler.choice(["NaiveBayes", "LogisticRegression"])
        c_param = sampler.continuous(0.01, 10)
        use_pca = sampler.boolean()

    # Fake experiences
    fake_exp_good = Experience(
        dataset_feature_extractor_name=warm.dataset_feature_extractor_class.__name__,
        system_feature_extractor_name=warm.system_feature_extractor_class.__name__,
        dataset_features=np.ones(10),
        system_features=np.ones(5),
        f1=0.8,
        evaluation_time=1.5,
        alias="myalias_good",
    )
    fake_exp_good2 = Experience(
        dataset_feature_extractor_name=warm.dataset_feature_extractor_class.__name__,
        system_feature_extractor_name=warm.system_feature_extractor_class.__name__,
        dataset_features=np.ones(10),
        system_features=np.ones(5),
        f1=0.9,
        evaluation_time=2.0,
        alias="myalias_good2",
    )
    fake_exp_bad = Experience(
        dataset_feature_extractor_name=warm.dataset_feature_extractor_class.__name__,
        system_feature_extractor_name=warm.system_feature_extractor_class.__name__,
        dataset_features=np.ones(10),
        system_features=np.ones(5),
        f1=None,
        evaluation_time=None,
        alias="myalias_bad",
    )

    # Vaciar algorithms
    fake_exp_good.algorithms = []
    fake_exp_good2.algorithms = []
    fake_exp_bad.algorithms = []

    def dummy_load(*args, **kwargs):
        fake_exp_good.algorithms = []
        fake_exp_good2.algorithms = []
        fake_exp_bad.algorithms = []
        return [fake_exp_good, fake_exp_good2, fake_exp_bad]

    ExperienceStore.load_all_experiences = dummy_load

    print("fake_exp_good.algorithms:", fake_exp_good.algorithms)
    print("fake_exp_good2.algorithms:", fake_exp_good2.algorithms)
    print("fake_exp_bad.algorithms:", fake_exp_bad.algorithms)

    initial_model = warm.warm_up(my_generator_fn)
    print("initial_model:", initial_model)


if __name__ == "__main__":
    test_warm_start()
