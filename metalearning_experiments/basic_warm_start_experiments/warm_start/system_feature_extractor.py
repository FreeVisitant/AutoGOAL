def my_generator_fn(sampler):
    transformer_name = sampler.choice(["TfidfVectorizer", "CountVectorizer"])
    max_features = sampler.discrete(100, 5000)
    use_pca = sampler.boolean()
    if use_pca:
        n_components = sampler.discrete(2, 100)

    classifier_name = sampler.choice(["NaiveBayes", "RandomForest", "LogisticRegression"])
    # No retornamos nada; se registran en sampler.updates.
